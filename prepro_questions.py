import json
import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('question_list_path')
parser.add_argument('multiple_choices_list_path')
parser.add_argument('answer_list_path')
parser.add_argument('target_path')
parser.add_argument('--vocab_dict_path', default='')

ARGS = parser.parse_args()


def prepro_questions(args):
    question_list_path = args.question_list_path
    multiple_choices_list_path = args.multiple_choices_list_path
    answer_list_path = args.answer_list_path
    vocab_dict_path = args.vocab_dict_path
    target_path = args.target_path

    print "Loading json files ..."
    question_list = json.load(open(question_list_path, 'rb'))
    multiple_choices_list = json.load(open(multiple_choices_list_path, 'rb'))
    answer_list = json.load(open(answer_list_path, 'rb'))
    if vocab_dict_path:
        vocab_dict = json.load(open(vocab_dict_path, 'rb'))
    else:
        vocab_set = set()

    tok_sents = []
    labels = []

    print "Preprocessing questions ..."
    for i, (raw_question, raw_mcs, raw_answer) in enumerate(zip(question_list, multiple_choices_list, answer_list)):
        tok_question = _tokenize(raw_question)
        tok_mcs = [_tokenize(raw_mc) for raw_mc in raw_mcs]
        tok_answer = _tokenize(raw_answer)

        tok_sent = [_append_answer(tok_question, tok_mc) for tok_mc in tok_mcs]
        label = [tok_answer == tok_mc for tok_mc in tok_mcs]
        tok_sents.append(tok_sent)
        labels.append(label)

        if not vocab_dict_path:
            vocab_set |= set(tok_question)
            for tok_mc in tok_mcs:
                vocab_set |= set(tok_mc)
            vocab_set |= set(tok_answer)

    if not vocab_dict_path:
        vocab_dict = {word: idx for idx, word in enumerate(list(sorted(vocab_set)))}

    sents = [[vocab_dict[word] for word in tok_sent] for tok_sent in tok_sents]

    sents_path = os.path.join(target_path, "sents.json")
    labels_path = os.path.join(target_path, "labels.json")
    vocab_dict_path = os.path.join(target_path, "vocab_dict.json")

    print "Dumping json files ..."
    json.dump(sents, open(sents_path, 'wb'))
    json.dump(labels, open(labels_path, 'wb'))
    json.dump(vocab_dict_path, open(vocab_dict_path, 'wb'))


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    tokens = [token.lower() for token in tokens]
    return tokens


def _append_answer(question, answer):
    return question + answer

if __name__ == "__main__":
    prepro_questions(ARGS)

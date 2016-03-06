import json
import os
import argparse
import re
from collections import Counter

import progressbar as pb
import h5py
import numpy as np
from nltk import PorterStemmer

parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--vocab_dict_path', default='')
parser.add_argument('--vocab_min_count', type=int, default=5)
parser.add_argument('--qa2hypo', default=0)

ARGS = parser.parse_args()

stemmer = PorterStemmer()

def prepro_questions(args):
    root_dir = args.root_dir
    question_list_path = os.path.join(root_dir, 'question.json')
    multiple_choices_list_path = os.path.join(root_dir, 'multiple_choice.json')
    answer_list_path = os.path.join(root_dir, 'answer.json')

    vocab_dict_path = args.vocab_dict_path
    vocab_min_count = args.vocab_min_count

    print("Loading json files ...")
    question_list = json.load(open(question_list_path, 'rb'))
    multiple_choices_list = json.load(open(multiple_choices_list_path, 'rb'))
    answer_list = json.load(open(answer_list_path, 'rb'))
    if vocab_dict_path:
        print("vocab dict specified: %s" % vocab_dict_path)
        vocab_dict = json.load(open(vocab_dict_path, 'rb'))
        create_vocab = False
    else:
        print("No vocab dict specified. Will create vocab dict.")
        vocab_counter = Counter()
        create_vocab = True

    tok_sents = []
    labels = []

    max_sent_len = 0

    print("Preprocessing questions ...")
    num_questions = len(question_list)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=num_questions).start()
    for i, (raw_question, raw_mcs, raw_answer) in enumerate(zip(question_list, multiple_choices_list, answer_list)):
        tok_question = _tokenize(raw_question)
        tok_mcs = [_tokenize(raw_mc) for raw_mc in raw_mcs] # mc: multiple choices
        tok_answer = _tokenize(raw_answer)

        # Generate hypotheses
        tok_sent = [_append_answer(tok_question, tok_mc) for tok_mc in tok_mcs]

        label = [int(tok_answer == tok_mc) for tok_mc in tok_mcs]
        tok_sents.append(tok_sent)
        labels.append(label)

        tok_sent_len = max(len(each_tok_sent) for each_tok_sent in tok_sent)
        if tok_sent_len > max_sent_len:
            max_sent_len = tok_sent_len

        if not vocab_dict_path:
            for tok in tok_question: _vadd(vocab_counter, tok)
            for tok_mc in tok_mcs:
                for tok in tok_mc: _vadd(vocab_counter, tok)
            for tok in tok_answer: _vadd(vocab_counter, tok)

        pbar.update(i+1)
    pbar.finish()

    if create_vocab:
        print("creating vocab dict ...")
        vocab_list, _ = zip(*sorted([pair for pair in vocab_counter.items() if pair[1] > vocab_min_count],
                                    key=lambda x: -x[1]))

        vocab_dict = {word: idx+1 for idx, word in enumerate(sorted(vocab_list))}
        vocab_dict['UNK'] = 0
        print("vocab size: %d" % len(vocab_dict))


    print("Converting to numpy array ...")
    lens = [[len(each_tok_sent) for each_tok_sent in tok_sent] for tok_sent in tok_sents]
    sents = [[[_vget(vocab_dict, word) for word in each_tok_sent] + [0] * (max_sent_len - len(each_tok_sent))
              for each_tok_sent in tok_sent] for tok_sent in tok_sents]
    sents = np.array(sents, dtype='int32')

    assert len(sents.shape) == 3

    sent_path = os.path.join(root_dir, "sent.h5")
    label_path = os.path.join(root_dir, "label.json")
    vocab_dict_path = os.path.join(root_dir, "vocab_dict.json")
    len_path = os.path.join(root_dir, "len.json")


    print("Dumping h5 file ...")
    f = h5py.File(sent_path, 'w')
    f['data'] = sents
    f.close()

    print("Dumping json files ...")
    json.dump(labels, open(label_path, 'wb'))
    if create_vocab:
        json.dump(vocab_dict, open(vocab_dict_path, 'wb'))
    json.dump(lens, open(len_path, 'wb'))


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    tokens = [token.lower() for token in tokens]
    return tokens


def _vadd(vocab_counter, word):
    word = word.lower()
    word = stemmer.stem(word)
    vocab_counter[word] += 1


def _vget(vocab_dict, word):
    word = word.lower()
    word = stemmer.stem(word)
    return vocab_dict[word] if word in vocab_dict else 0


def _append_answer(question, ans):
    assert isinstance(question, list)
    assert isinstance(ans, list)
    return question + ans

if __name__ == "__main__":
    prepro_questions(ARGS)

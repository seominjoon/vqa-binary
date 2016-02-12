import json
import os
import argparse
import re
from collections import Counter

import progressbar as pb
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('question_list_path')
parser.add_argument('multiple_choices_list_path')
parser.add_argument('answer_list_path')
parser.add_argument('target_path')
parser.add_argument('--vocab_dict_path', default='')
parser.add_argument('--vocab_min_count', type=int, default=5)

ARGS = parser.parse_args()


def prepro_questions(args):
    question_list_path = args.question_list_path
    multiple_choices_list_path = args.multiple_choices_list_path
    answer_list_path = args.answer_list_path
    vocab_dict_path = args.vocab_dict_path
    target_path = args.target_path
    vocab_min_count = args.vocab_min_count

    print "Loading json files ..."
    question_list = json.load(open(question_list_path, 'rb'))
    multiple_choices_list = json.load(open(multiple_choices_list_path, 'rb'))
    answer_list = json.load(open(answer_list_path, 'rb'))
    if vocab_dict_path:
        vocab_dict = json.load(open(vocab_dict_path, 'rb'))
    else:
        vocab_counter = Counter()

    tok_sents = []
    labels = []

    max_sent_len = 0

    print "Preprocessing questions ..."
    num_questions = len(question_list)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=num_questions).start()
    for i, (raw_question, raw_mcs, raw_answer) in enumerate(zip(question_list, multiple_choices_list, answer_list)):
        tok_question = _tokenize(raw_question)
        tok_mcs = [_tokenize(raw_mc) for raw_mc in raw_mcs]
        tok_answer = _tokenize(raw_answer)

        tok_sent = [_append_answer(tok_question, tok_mc) for tok_mc in tok_mcs]
        label = [int(tok_answer == tok_mc) for tok_mc in tok_mcs]
        tok_sents.append(tok_sent)
        labels.append(label)

        tok_sent_len = max(len(each_tok_sent) for each_tok_sent in tok_sent)
        if tok_sent_len > max_sent_len:
            max_sent_len = tok_sent_len

        if not vocab_dict_path:
            for tok in tok_question: vocab_counter[tok] += 1
            for tok_mc in tok_mcs:
                for tok in tok_mc: vocab_counter[tok] += 1
            for tok in tok_answer: vocab_counter[tok] += 1

        pbar.update(i+1)
    pbar.finish()

    if not vocab_dict_path:
        print "creating vocab dict ..."
        vocab_list = zip(*sorted([pair for pair in vocab_counter.iteritems() if pair[1] > vocab_min_count],
                                 key=lambda x: -x[1]))[0]

        vocab_dict = {word: idx+1 for idx, word in enumerate(sorted(vocab_list))}
        vocab_dict['UNK'] = 0
        print "vocab size: %d" % len(vocab_dict)

    def _get(word_):
        return vocab_dict[word] if word in vocab_dict else 0

    print "Converting to numpy array ..."
    sents = [[[_get(word) for word in each_tok_sent] + [0] * (max_sent_len - len(each_tok_sent))
              for each_tok_sent in tok_sent] for tok_sent in tok_sents]
    sents = np.array(sents, dtype='int32')

    assert len(sents.shape) == 3

    sent_path = os.path.join(target_path, "sent.h5")
    label_path = os.path.join(target_path, "label.json")
    vocab_dict_path = os.path.join(target_path, "vocab_dict.json")


    print "Dumping h5 file ..."
    f = h5py.File(sent_path, 'w')
    f['data'] = sents
    f.close()

    print "Dumping json files ..."
    json.dump(labels, open(label_path, 'wb'))
    json.dump(vocab_dict, open(vocab_dict_path, 'wb'))


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    tokens = [token.lower() for token in tokens]
    return tokens


def _append_answer(question, answer):
    return question + answer

if __name__ == "__main__":
    prepro_questions(ARGS)

import json
import os
import argparse
import re
from collections import Counter

#import progressbar as pb
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--vocab_dict_path', default='')
parser.add_argument('--vocab_min_count', type=int, default=5)
parser.add_argument('--qa2hypo', default=0)

ARGS = parser.parse_args()



# auxiliary verbs, from https://en.wikipedia.org/wiki/Auxiliary_verb
AUX_V = ['am', 'is', 'are', 'can', 'could', 'dare', 'do', 'does', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would']
AUX_V_REGEX = '('+'|'.join(['('+AUX_V[i]+')' for i in range(len(AUX_V))])+')'
AUX_V_BE = ['am', 'is', 'are']
AUX_V_BE_REGEX = '('+'|'.join(['('+AUX_V_BE[i]+')' for i in range(len(AUX_V_BE))])+')'
AUX_V_DOES = ['can', 'could', 'dare', 'does', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would']
AUX_V_DOES_REGEX = '('+'|'.join(['('+AUX_V_DOES[i]+')' for i in range(len(AUX_V_DOES))])+')'
AUX_V_DO_REGEX = '(do)'


# global variables
QUESTION_TYPES = ['__+', \
'(when '+AUX_V_REGEX+'.*)|(when\?)', \
'(where '+AUX_V_REGEX+'.*)|(where\?)', \
'what', \
'which', \
'(whom '+AUX_V_REGEX+'.*)|(who '+AUX_V_REGEX+'.*)|(who\?)|(whom\?)', \
'why', \
'(how many)|(how much)', \
'(\Ahow [^(many)(much)])|(\W+how [^(many)(much)])', \
'(name)|(choose)|(identify)'
]


# SAMPLE_TYPE:
# -1: don't sample randomly, sample by question type
# 0: sample the inverse of all the question types
# not -1 or 0: sample by question type
# SAMPLE_TYPE = 200
# used when SAMPLE_TYPE == -1
# QUESTION_TYPE = 8



def prepro_questions(args):
    root_dir = args.root_dir
    question_list_path = os.path.join(root_dir, 'question.json')
    multiple_choices_list_path = os.path.join(root_dir, 'multiple_choice.json')
    answer_list_path = os.path.join(root_dir, 'answer.json')

    vocab_dict_path = args.vocab_dict_path
    vocab_min_count = args.vocab_min_count
    ifqa2hypo = args.qa2hypo

    print "Loading json files ..."
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

    print "Preprocessing questions ..."
    num_questions = len(question_list)
    #pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=num_questions).start()
    for i, (raw_question, raw_mcs, raw_answer) in enumerate(zip(question_list, multiple_choices_list, answer_list)):
        tok_question = _tokenize(raw_question)
        tok_mcs = [_tokenize(raw_mc) for raw_mc in raw_mcs] # mc: multiple choices
        tok_answer = _tokenize(raw_answer)

        # Generate hypotheses
        tok_sent = None
        if ifqa2hypo:
            tok_sent = [_qa2hypo(tok_question, tok_mc) for tok_mc in tok_mcs]
            print "qa2hypo is used!"
	else:
            tok_sent = [_append_answer(tok_question, tok_mc) for tok_mc in tok_mcs]
            print "append_answer is used!"
        
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

        #pbar.update(i+1)
    #pbar.finish()

    if create_vocab:
        print "creating vocab dict ..."
        vocab_list = zip(*sorted([pair for pair in vocab_counter.iteritems() if pair[1] > vocab_min_count],
                                 key=lambda x: -x[1]))[0]

        vocab_dict = {word: idx+1 for idx, word in enumerate(sorted(vocab_list))}
        vocab_dict['UNK'] = 0
        print "vocab size: %d" % len(vocab_dict)

    def _get(word_):
        return vocab_dict[word] if word in vocab_dict else 0

    print "Converting to numpy array ..."
    lens = [[len(each_tok_sent) for each_tok_sent in tok_sent] for tok_sent in tok_sents]
    sents = [[[_get(word) for word in each_tok_sent] + [0] * (max_sent_len - len(each_tok_sent))
              for each_tok_sent in tok_sent] for tok_sent in tok_sents]
    sents = np.array(sents, dtype='int32')

    assert len(sents.shape) == 3

    sent_path = os.path.join(root_dir, "sent.h5")
    label_path = os.path.join(root_dir, "label.json")
    vocab_dict_path = os.path.join(root_dir, "vocab_dict.json")
    len_path = os.path.join(root_dir, "len.json")


    print "Dumping h5 file ..."
    f = h5py.File(sent_path, 'w')
    f['data'] = sents
    f.close()

    print "Dumping json files ..."
    json.dump(labels, open(label_path, 'wb'))
    if create_vocab:
        json.dump(vocab_dict, open(vocab_dict_path, 'wb'))
    json.dump(lens, open(len_path, 'wb'))


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    tokens = [token.lower() for token in tokens]
    return tokens


def _append_answer(question, ans):
    assert isinstance(question, list)
    assert isinstance(ans, list)
    return question + ans

def _qa2hypo(question, ans):
	question = question.lower()
    ans = ans.lower().strip('.')

    q_type = get_question_type(question)
    # test_patterns([q_type], question)
    sent = rule_based_transform(question, ans, q_type)

    return sent

#######################################################
# helper functions below
#######################################################

# determine the question type
def get_question_type(question):
    for q_type in QUESTION_TYPES:
        if re.search(q_type, question):
            return q_type
    return 'none of these'

# rule based qa2hypo transformation
def rule_based_transform(question, ans, q_type):
    if q_type == QUESTION_TYPES[0]:
        s, e = test_pattern(q_type, question)
        hypo = replace(question, s, e, ans)
    else:
        if q_type == QUESTION_TYPES[1]:
            s, e = test_pattern('when', question)
            if re.search('when '+AUX_V_DOES_REGEX, question):
                s2, e2 = test_pattern('when '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            elif re.search('when '+AUX_V_DO_REGEX, question):
                s3, e3 = test_pattern('when '+AUX_V_DO_REGEX, question)
                hypo = replace(question, s3, e3, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[2]:
            s, e = test_pattern('where', question)
            if re.search('where '+AUX_V_DOES_REGEX, question):
                s2, e2 = test_pattern('where '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' at '+ans
            elif re.search('where '+AUX_V_DO_REGEX, question):
                s3, e3 = test_pattern('where '+AUX_V_DO_REGEX, question)
                hypo = replace(question, s3, e3, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[3]:
            s, e = test_pattern('what', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[4]:
            s, e = test_pattern('which', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[5]:
            s, e = test_pattern('(who)|(whom)', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[6]:
            s, e = test_pattern('why', question)
            hypo = strip_question_mark(question)+', '+ans
            if not re.search('because', ans, re.IGNORECASE):
                hypo = strip_question_mark(question)+', because '+ans

        elif q_type == QUESTION_TYPES[7]:
            s, e = test_pattern('(how many)|(how much)', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[8]:
            s, e = test_pattern('(\Ahow )|(\W+how )', question)
            hypo = replace(question, s, e, ' '+ans+' is how ')

        elif q_type == QUESTION_TYPES[9]:
            s, e = test_pattern('(name)|(choose)|(identify)', question)
            hypo = replace(question, s, e, ans+' is')

        else:
            hypo = strip_nonalnum_re(question)+' '+ans

    hypo = strip_question_mark(hypo)
    return hypo

# strip the question mark
def strip_question_mark(sent):
    if sent.endswith('?') or sent.endswith(':'):
        return sent[:-1]
    else:
        return sent

# strip any non alnum characters in the end
def strip_nonalnum_re(sent):
    return re.sub(r"^\W+|\W+$", "", sent)

# replace 
def replace(text, start, end, replacement):
    text_left = text[:start]
    text_right = text[end:]
    return text_left+replacement+text_right

# for print purpose
def test_patterns(patterns, text):
    """Given source text and a list of patterns, look for
    matches for each pattern within the text and print
    them to stdout.
    """
    # Show the character positions and input text
    # print
    # print ''.join(str(i/10 or ' ') for i in range(len(text)))
    # print ''.join(str(i%10) for i in range(len(text)))
    # print text

    # Look for each pattern in the text and print the results
    for pattern in patterns:
        print
        print 'Matching "%s"' % pattern
        # --- regex ---
        for match in re.finditer(pattern, text):
            s = match.start()
            e = match.end()
            print '  %2d : %2d = "%s"' % \
                (s, e-1, text[s:e])
            print '    Groups:', match.groups()
            if match.groupdict():
                print '    Named groups:', match.groupdict()
            print
    return

# for return purpose
def test_pattern(pattern, text):
    match = re.search(pattern, text)
    s = match.start()
    e = match.end()
    # print '  %2d : %2d = "%s"' % (s, e-1, text[s:e])
    return s, e

#######################################################
# helper functions above
#######################################################


if __name__ == "__main__":
    prepro_questions(ARGS)

import numpy as np
import json
import argparse
import os
import random
import re

parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
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
SAMPLE_TYPE = 200
# used when SAMPLE_TYPE == -1
QUESTION_TYPE = 8



#############################################################################################

# prepare a qa_pairs json object and write it to disk
def prepro_questions_annotations(args):
	root_dir = args.root_dir
	questions_path = os.path.join(root_dir, 'MultipleChoice_shining3_all_questions.json')
	annotations_path = os.path.join(root_dir, 'shining3_all_annotations.json')

	print "Loading json files ..."
	questions = json.load(open(questions_path, 'rb'))
	annotations = json.load(open(annotations_path, 'rb'))
	
	#return questions, annotations

	qa_path = os.path.join(root_dir, 'qa_pairs.json')
	qa_pairs = {}
	qa_pairs['qa_pairs'] = []
	for i in range(len(annotations['annotations'])):
		qa_pair = {}
		qa_pair['image_id'] = annotations['annotations'][i]['image_id']
		qa_pair['ans'] = annotations['annotations'][i]['multiple_choice_answer']
		qa_pair['question_id'] = annotations['annotations'][i]['question_id']
		qa_pair['question'] = questions['questions'][i]['question']
		qa_pairs['qa_pairs'].append(qa_pair)

	print "Dumping json files ..."
	json.dump(qa_pairs, open(qa_path, 'wb'))

#############################################################################################


# turn qa_pairs into hypotheses
def qa2hypo(args):
	root_dir = args.root_dir
	qa_path = os.path.join(root_dir, 'qa_pairs.json')
	qa_res_path = os.path.join(root_dir, 'qa_res.json')

	print "Loading json files ..."
	qa_pairs = json.load(open(qa_path, 'rb'))
	qa_pairs_list = qa_pairs['qa_pairs']

	# number of samples and the types of questions to sample
	k = SAMPLE_TYPE
	
	# execute the sampling (for the purpose of examining the result)
	q_type = QUESTION_TYPES[QUESTION_TYPE]
	qa_pairs_list = sample_qa(qa_pairs_list, k, q_type) # set the case lower in the function for questions
	
	# result file
	res = []

	ctr = 0
	for item in qa_pairs_list:
		question = item['question']
		ans = item['ans']
		question = question.lower()
		ans = ans.lower().strip('.')

		# determine the question type:
		if k != -1:
			q_type = get_question_type(question)

		print 'Question:', question
		print 'Answer:', ans

		test_patterns([q_type], question)
		sent = rule_based_transform(question, ans, q_type)
		
		print 'Result:', sent
		res.append({'Question':question, 'Answer':ans, 'Result':sent})

		ctr += 1
		print "--------------------------------------"
	
	print ctr
	print "Dumping json files ..."
	json.dump(res, open(qa_res_path, 'wb'))

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

# sample sentences
def sample_qa(qa_pairs_list, k, q_type):
	l = range(len(qa_pairs_list))
	l_sampled = []

	# random sampling
	if k != -1 and k != 0:
		l_sampled = random.sample(l, k)

	# inverse sampling
	elif k == 0:
		return sample_qa_inverse(qa_pairs_list)

	# sample by question type (k == -1)
	else:
		for num in l:
			q = qa_pairs_list[num]['question'].lower() # use the lower case for all
			# --- regex ---
			if re.search(q_type, q):
				l_sampled.append(num) 

	return [qa_pairs_list[i] for i in l_sampled]

# sample sentences -- the inverse set; this is a helper to sample_qa
def sample_qa_inverse(qa_pairs_list):
	l = range(len(qa_pairs_list))
	l_sampled = []

	for num in l:
		q = qa_pairs_list[num]['question'].lower() # use the lower case for all
		flag = 0
		for q_type in QUESTION_TYPES:
			# --- regex ---
			if re.search(q_type, q) != None:
				flag = 1
				break
		if flag == 0:
			l_sampled.append(num)

	return [qa_pairs_list[i] for i in l_sampled]


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

if __name__ == "__main__":
	#prepro_questions_annotations(ARGS)
	qa_pairs = qa2hypo(ARGS)
	# print len(qa_pairs['qa_pairs'])


	#questions, annotations = read_questions_annotations(ARGS)

	# for key in questions:
	# 	print key

	#print questions['num_choices']
	# print len(questions['questions'])

	# print questions['questions'][0]
	# print questions['questions'][1]

	# question_id1 = {}
	# for q in questions['questions']:
	# 	question_id = q['question_id']
	# 	if question_id in question_id1:
	# 		question_id1[question_id] += 1
	# 	else:
	# 		question_id1[question_id] = 0
	# print len(question_id1)

	#for key in annotations:
	#	print key

	# print len(annotations['annotations'])

	# print annotations['annotations'][0]
	# print annotations['annotations'][1]

	# annotations_id1 = {}
	# for a in annotations['annotations']:
	# 	annotations_id = a['question_id']
	# 	if annotations_id in annotations_id1:
	# 		annotations_id1[annotations_id] += 1
	# 	else:
	# 		annotations_id1[annotations_id] = 0
	# print len(annotations_id1)
import numpy as np
import json
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
ARGS = parser.parse_args()

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

# turn qa_pairs into hypotheses
def qa2hypo(args):
	root_dir = args.root_dir
	qa_path = os.path.join(root_dir, 'qa_pairs.json')

	print "Loading json files ..."
	qa_pairs = json.load(open(qa_path, 'rb'))
	qa_pairs_list = qa_pairs['qa_pairs']

	# number of samples
	k = 5
	qa_pairs_list_sampled = sample_qa(qa_pairs_list, k)
	for i in qa_pairs_list_sampled:
		print i


# sample sentences
def sample_qa(qa_pairs_list, k):
	l = range(len(qa_pairs_list))
	# random sampling
	l_sampled = random.sample(l, k)
	return [qa_pairs_list[i] for i in l_sampled]




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
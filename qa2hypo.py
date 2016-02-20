import numpy as np
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
ARGS = parser.parse_args()


def qa2hypo(args):
	root_dir = args.root_dir
	questions_path = os.path.join(root_dir, 'MultipleChoice_shining3_all_questions.json')
	annotations_path = os.path.join(root_dir, 'shining3_all_annotations.json')

	print "Loading json files ..."
	#questions = json.load(open(questions_path, 'rb'))
	annotations = json.load(open(annotations_path, 'rb'))
	
	#return questions, annotations

	qa_path = os.path.join(root_dir, 'qa_pairs.json')
	qa_pairs = {}

	print "Dumping json files ..."
	json.dump(qa_pairs, open(qa_path, 'wb'))





if __name__ == "__main__":
	qa2hypo(ARGS)
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
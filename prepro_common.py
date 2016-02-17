import json
import os
import argparse
import progressbar as pb

parser = argparse.ArgumentParser()
parser.add_argument('question_json_path')
parser.add_argument('annotation_json_path')
parser.add_argument('images_dir')
parser.add_argument('target_dir')
parser.add_argument('--prefix')
parser.add_argument('--ext')
parser.add_argument('--zfill_width', default=12, type=int)

ARGS = parser.parse_args()


def prepro_common(args):
    question_json_path = args.question_json_path
    annotation_json_path = args.annotation_json_path
    images_dir = args.images_dir
    id_len = args.id_len
    prefix = args.prefix
    ext = args.ext
    target_dir = args.target_dir

    print "Loading %s ..." % question_json_path
    question_json = json.load(open(question_json_path, 'rb'))
    print "Loading %s ..." % annotation_json_path
    annotation_json = json.load(open(annotation_json_path, 'rb'))
    question_list = []
    multiple_choices_list = []
    answer_list = []
    image_index_list = []
    image_path_list = []
    image_path_dict = {}

    num_questions = len(question_json['questions'])

    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=num_questions).start()

    for idx, (question_dict, annotation_dict) in enumerate(zip(question_json['questions'], annotation_json['annotations'])):
        question_id = question_dict['question_id']
        assert question_id == annotation_dict['question_id']
        image_id = question_dict['image_id']
        assert image_id == annotation_dict['image_id']
        question = question_dict['question']
        multiple_choices = question_dict['multiple_choices']
        answer = annotation_dict['multiple_choice_answer']
        image_id_str = str(image_id).zfill(id_len)
        image_path = os.path.join(images_dir, "%s%s%s" % (prefix, image_id_str, ext))

        question_list.append(question)
        multiple_choices_list.append(multiple_choices)
        answer_list.append(answer)
        if image_path in image_path_dict:
            image_index_list.append(image_path_dict[image_path])
        else:
            image_index_list.append(len(image_path_list))
            image_path_dict[image_path] = len(image_path_list)
            image_path_list.append(image_path)

        pbar.update(idx + 1)

    pbar.finish()

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    print "Dumping json files ..."
    json.dump(question_list, open(os.path.join(target_dir, "question.json"), 'w'))
    json.dump(multiple_choices_list, open(os.path.join(target_dir, "multiple_choice.json"), 'w'))
    json.dump(answer_list, open(os.path.join(target_dir, "answer.json"), 'w'))
    json.dump(image_path_list, open(os.path.join(target_dir, "image_path.json"), 'w'))
    json.dump(image_index_list, open(os.path.join(target_dir, "image_idx.json"), 'w'))

if __name__ == "__main__":
    prepro_common(ARGS)

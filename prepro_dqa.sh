# training
# python prepro_common.py /home/ubuntu/vqa-data/MultipleChoice_shining3_all_questions.json /home/ubuntu/vqa-data/shining3_all_annotations.json /home/ubuntu/vqa-data/all train/ --prefix COCO_all_
# python prepro_questions.py train/question.json train/multiple_choice.json train/answer.json train
# th prepro_images.lua --image_path_json train/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path train/image_rep.h5 --backend nn

python prepro_common.py /home/ubuntu/dqa-data/vqa-format/shining3/MultipleChoice_shining3_all_questions.json /home/ubuntu/dqa-data/vqa-format/shining3/shining3_all_annotations.json /home/ubuntu/dqa-data/vqa-format/shining3/all dqa_all/ --prefix shining3_all_ --ext .png
python prepro_questions.py dqa_all/question.json dqa_all/multiple_choice.json dqa_all/answer.json dqa_all --vocab_dict_path train/vocab_dict.json
th prepro_images.lua --image_path_json dqa_all/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path dqa_all/image_rep.h5 --backend nn

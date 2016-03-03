#!/usr/bin/env bash
# training
#python prepro_common.py ../vqa-data/MultipleChoice_mscoco_train2014_questions.json ../vqa-data/mscoco_train2014_annotations.json ../vqa-data/train2014 train/ --prefix COCO_train2014_
python prepro_questions.py train --qa2hypo 1 #train/question.json train/multiple_choice.json train/answer.json train
# th prepro_images.lua --image_path_json train/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path train/image_rep.h5 --backend nn

#python prepro_common.py ../vqa-data/MultipleChoice_mscoco_val2014_questions.json ../vqa-data/mscoco_val2014_annotations.json ../vqa-data/val2014 val/ --prefix COCO_val2014_
python prepro_questions.py val --vocab_dict_path train/vocab_dict.json --qa2hypo 1 #val/question.json val/multiple_choice.json val/answer.json
# th prepro_images.lua --image_path_json val/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path val/image_rep.h5 --backend nn

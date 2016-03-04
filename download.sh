#!/usr/bin/env bash
CM_DIR="~/caffe-models"
DQA_DATA_DIR="~/data/dqa"

mkdir $CM_DIR
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel $DL_PATH
wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt $DL_PATH

mkdir $DQA_DATA_DIR
wget https://s3-us-west-2.amazonaws.com/dqa-data/shining3-1500r-vqa.zip $DQA_DATA_DIR
unzip $DQA_DATA_DIR/shining3-1500r-vqa.zip $DQA_DATA_DIR
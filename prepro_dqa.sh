# Replace: PREFIX, EXT, DATA_DIR, DATA_TYPE, SUBTYPE, ROOT_DIR
PREFIX="shining3_all_"
EXT=".png"
DATA_DIR="dqa-data/vqa-format/shining3"
DATA_TYPE="shining3"
SUBTYPE="train"
IMAGES_DIR="all"
ROOT_DIR="dqa-train"
python prepro_common.py /home/ubuntu/$DATA_DIR/MultipleChoice_"$DATA_TYPE"_"$SUBTYPE"_questions.json /home/ubuntu/$DATA_DIR/"$DATA_TYPE"_"$SUBTYPE"_annotations.json /home/ubuntu/$DATA_DIR/$IMAGES_DIR/ $ROOT_DIR --prefix $PREFIX --ext $EXT
python prepro_questions.py $ROOT_DIR
th prepro_images.lua --image_path_json $ROOT_DIR/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path $ROOT_DIR/image_rep.h5 --backend nn
VOCAB_DICT_PATH="$ROOTDIR/vocab_dict.json"
PREFIX="shining3_all_"
EXT=".png"
DATA_DIR="dqa-data/vqa-format/shining3"
DATA_TYPE="shining3"
SUBTYPE="test"
IMAGES_DIR="all"
ROOT_DIR="dqa-test"
python prepro_common.py /home/ubuntu/$DATA_DIR/MultipleChoice_"$DATA_TYPE"_"$SUBTYPE"_questions.json /home/ubuntu/$DATA_DIR/"$DATA_TYPE"_"$SUBTYPE"_annotations.json /home/ubuntu/$DATA_DIR/$IMAGES_DIR/ $ROOT_DIR --prefix $PREFIX --ext $EXT
python prepro_questions.py $ROOT_DIR --vocab_dict_path $VOCAB_DICT_PATH
th prepro_images.lua --image_path_json $ROOT_DIR/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path $ROOT_DIR/image_rep.h5 --backend nn

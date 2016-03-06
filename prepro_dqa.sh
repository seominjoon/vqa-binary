# Replace: PREFIX, EXT, DATA_DIR, DATA_TYPE, SUBTYPE, ROOT_DIR
USER_DIR="$HOME"
PREFIX="shining3-1500r_train_"
EXT=".png"
DATA_DIR="data/dqa/shining3-1500r-train-vqa"
DATA_TYPE="shining3-1500r"
SUBTYPE="train"
IMAGES_DIR="train"
ROOT_DIR="dqa-train"
python prepro_common.py $USER_DIR/$DATA_DIR/MultipleChoice_"$DATA_TYPE"_"$SUBTYPE"_questions.json $USER_DIR/$DATA_DIR/"$DATA_TYPE"_"$SUBTYPE"_annotations.json $USER_DIR/$DATA_DIR/$IMAGES_DIR/ $ROOT_DIR --prefix $PREFIX --ext $EXT
python prepro_questions.py $ROOT_DIR --vocab_min_count 1
# th prepro_images.lua --image_path_json $ROOT_DIR/image_path.json --cnn_proto $USER_DIR/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model $USER_DIR/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path $ROOT_DIR/image_rep.h5 --backend nn 
VOCAB_DICT_PATH="$ROOT_DIR/vocab_dict.json"
PREFIX="shining3-1500r_test_"
EXT=".png"
DATA_DIR="data/dqa/shining3-1500r-test-vqa"
DATA_TYPE="shining3-1500r"
SUBTYPE="test"
IMAGES_DIR="test"
ROOT_DIR="dqa-test"
python prepro_common.py $USER_DIR/$DATA_DIR/MultipleChoice_"$DATA_TYPE"_"$SUBTYPE"_questions.json $USER_DIR/$DATA_DIR/"$DATA_TYPE"_"$SUBTYPE"_annotations.json $USER_DIR/$DATA_DIR/$IMAGES_DIR/ $ROOT_DIR --prefix $PREFIX --ext $EXT
python prepro_questions.py $ROOT_DIR --vocab_dict_path $VOCAB_DICT_PATH
# th prepro_images.lua --image_path_json $ROOT_DIR/image_path.json --cnn_proto $USER_DIR/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model $USER_DIR/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path $ROOT_DIR/image_rep.h5 --backend nn

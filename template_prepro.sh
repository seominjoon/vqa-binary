# Replace: PREFIX, EXT, DATA_DIR, DATA_TYPE, SUBTYPE, ROOT_DIR
PREFIX=""
EXT=""
DATA_DIR=""
DATA_TYPE=""
SUBTYPE=""
ROOT_DIR=""
python prepro_common.py /home/ubuntu/$DATA_DIR/MultipleChoice_"$DATA_TYPE"_"$SUBTYPE"_questions.json /home/ubuntu/$DATA_DIR/"$DATA_TYPE"_"$SUBTYPE"_annotations.json /home/ubuntu/$DATA_DIR/$SUBTYPE/ $ROOT_DIR --prefix $PREFIX --ext $EXT
python prepro_questions.py $ROOT_DIR
th prepro_images.lua --image_path_json $ROOT_DIR/image_path.json --cnn_proto /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model /home/ubuntu/caffe-models/VGG_ILSVRC_19_layers.caffemodel --out_path $ROOT_DIR/image_rep.h5 --backend nn

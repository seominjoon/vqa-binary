import json

import tensorflow as tf
import progressbar as pb

from data import read_vqa

flags = tf.app.flags

flags.DEFINE_integer("train_batch_size", 100, "Batch size during training [100]")
flags.DEFINE_string("train_image_rep_h5", "train/image_rep.h5", "image_rep.h5 file path for training [train/image_rep.h5]")
flags.DEFINE_string("train_image_idx", "train/image_idx.json", "image_idx.json file path for training [train/image_idx.json]")
flags.DEFINE_string("train_sent_h5", "train/sent.h5", "sent.h5 file path for training [train/sent.h5]")
flags.DEFINE_string("train_label", "train/label.json", "label.json file path for training [train/label.json]")
flags.DEFINE_string("vocab_dict", "train/vocab_dict.json", "vocab_dict.json file path [train/vocab_dict.json]")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs [100]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm during trainig [40]")

FLAGS = flags.FLAGS

def main(_):
    vocab_dict = json.load(open(FLAGS.vocab_dict, 'rb'))
    FLAGS.vocab_size = len(vocab_dict)

    train_data_set = read_vqa(FLAGS.train_batch_size, FLAGS.train_image_rep_h5, FLAGS.train_image_idx, FLAGS.train_sent_h5, FLAGS.train_label)
    FLAGS.image_rep_dim = train_data_set.image_rep_dim
    FLAGS.max_sent_size = train_data_set.max_sent_size
    FLAGS.num_mcs = train_data_set.num_mcs

    # pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=train_data_set.num_batches).start()
    tf_graph = tf.Graph()


if __name__ == "__main__":
    tf.app.run()

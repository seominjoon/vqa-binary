import json
import os

import tensorflow as tf
from pprint import pprint

from data import read_vqa
from model import Model

flags = tf.app.flags

# All the file directories... should get rid of these!
flags.DEFINE_string("train_image_rep_h5", "train/image_rep.h5", "image_rep.h5 file path for training [train/image_rep.h5]")
flags.DEFINE_string("train_image_idx", "train/image_idx.json", "image_idx.json file path for training [train/image_idx.json]")
flags.DEFINE_string("train_sent_h5", "train/sent.h5", "sent.h5 file path for training [train/sent.h5]")
flags.DEFINE_string("train_label", "train/label.json", "label.json file path for training [train/label.json]")
flags.DEFINE_string("train_len", "train/len.json", "len.json file path for training [train/len.json]")
flags.DEFINE_string("val_image_rep_h5", "val/image_rep.h5", "image_rep.h5 file path for validation [val/image_rep.h5]")
flags.DEFINE_string("val_image_idx", "val/image_idx.json", "image_idx.json file path for validation [val/image_idx.json]")
flags.DEFINE_string("val_sent_h5", "val/sent.h5", "sent.h5 file path for validation [val/sent.h5]")
flags.DEFINE_string("val_label", "val/label.json", "label.json file path for validation [val/label.json]")
flags.DEFINE_string("val_len", "val/len.json", "len.json file path for valing [val/len.json]")
flags.DEFINE_string("vocab_dict", "val/vocab_dict.json", "vocab_dict.json file path [val/vocab_dict.json]")

# training parameters
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs [100]")
flags.DEFINE_integer("train_batch_size", 100, "Batch size during training [100]")
flags.DEFINE_integer("val_batch_size", 100, "Batch size during validation [100]")
flags.DEFINE_integer("rnn_num_layers", 3, "Number of RNN (LSTM) layers [3]")
flags.DEFINE_integer("rnn_hidden_size", 512, "Hidden size of RNN (LSTM) [512]")
flags.DEFINE_integer("common_size", 1024, "Common size [1024]")
flags.DEFINE_float("learning_rate", 1e-2, "Learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm during trainig [40]")

# training and testing options
flags.DEFINE_boolean("is_train", False, "Train? [False]")
flags.DEFINE_integer("eval_period", 3, "Evaluation period [3]")
flags.DEFINE_integer("eval_num_batches", 50, "Number of batches to evaluate during training [50]")
flags.DEFINE_integer("save_period", 1, "Save period [1]")
flags.DEFINE_string("save_dir", "save", "Save path [save]")
flags.DEFINE_string("log_dir", "log", "Log path [log]")

# for debugging
flags.DEFINE_boolean("draft", False, "Quick iteration of epochs? [False]")

FLAGS = flags.FLAGS

def main(_):
    vocab_dict = json.load(open(FLAGS.vocab_dict, 'rb'))
    FLAGS.vocab_size = len(vocab_dict)
    pprint(FLAGS.__dict__)

    train_data_set = read_vqa(FLAGS.train_batch_size, FLAGS.train_image_rep_h5, FLAGS.train_image_idx,
                              FLAGS.train_sent_h5, FLAGS.train_len, FLAGS.train_label)
    FLAGS.image_rep_size = train_data_set.image_rep_size
    FLAGS.max_sent_size = train_data_set.max_sent_size
    FLAGS.num_mcs = train_data_set.num_mcs
    val_data_set = read_vqa(FLAGS.val_batch_size, FLAGS.val_image_rep_h5, FLAGS.val_image_idx, FLAGS.val_sent_h5, FLAGS.val_len, FLAGS.val_label)

    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    # Time-sensitive parameters. Will be altered if draft.
    if FLAGS.draft:
        FLAGS.train_num_batches = 5
        FLAGS.eval_num_batches = 1
        FLAGS.val_num_batches = 5
        FLAGS.num_epochs = 5
        FLAGS.eval_period = 1

    tf_graph = tf.Graph()
    writer = tf.train.SummaryWriter(FLAGS.log_dir, tf_graph.as_graph_def())
    model = Model(tf_graph, FLAGS, writer)
    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.is_train:
            model.train(sess, train_data_set, FLAGS.learning_rate, val_data_set=val_data_set)
        else:
            model.load(sess)

        print "-" * 80
        print "training complete."
        print "testing %d examples (train data) ..." % train_data_set.num_examples
        model.test(sess, train_data_set, num_batches=FLAGS.train_num_batches)

        print "testing %d examples (val data) ..." % val_data_set.num_examples
        model.test(sess, val_data_set, num_batches=FLAGS.val_num_batches)

if __name__ == "__main__":
    tf.app.run()

import json

import tensorflow as tf
import progressbar as pb

from data import read_vqa
from model import Model

flags = tf.app.flags

flags.DEFINE_integer("train_batch_size", 100, "Batch size during training [100]")
flags.DEFINE_integer("val_batch_size", 100, "Batch size during validation [100]")
flags.DEFINE_string("train_image_rep_h5", "train/image_rep.h5", "image_rep.h5 file path for training [train/image_rep.h5]")
flags.DEFINE_string("train_image_idx", "train/image_idx.json", "image_idx.json file path for training [train/image_idx.json]")
flags.DEFINE_string("train_sent_h5", "train/sent.h5", "sent.h5 file path for training [train/sent.h5]")
flags.DEFINE_string("train_label", "train/label.json", "label.json file path for training [train/label.json]")
flags.DEFINE_string("val_image_rep_h5", "val/image_rep.h5", "image_rep.h5 file path for validation [val/image_rep.h5]")
flags.DEFINE_string("val_image_idx", "val/image_idx.json", "image_idx.json file path for validation [val/image_idx.json]")
flags.DEFINE_string("val_sent_h5", "val/sent.h5", "sent.h5 file path for validation [val/sent.h5]")
flags.DEFINE_string("val_label", "val/label.json", "label.json file path for validation [val/label.json]")
flags.DEFINE_string("vocab_dict", "val/vocab_dict.json", "vocab_dict.json file path [val/vocab_dict.json]")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs [100]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm during trainig [40]")
flags.DEFINE_integer("num_layers", 1, "Number of LSTM layers [1]")
flags.DEFINE_integer("hidden_size", 200, "Hidden size of LSTM [200]")

FLAGS = flags.FLAGS

def main(_):
    vocab_dict = json.load(open(FLAGS.vocab_dict, 'rb'))
    FLAGS.vocab_size = len(vocab_dict)

    train_data_set = read_vqa(FLAGS.train_batch_size, FLAGS.train_image_rep_h5, FLAGS.train_image_idx, FLAGS.train_sent_h5, FLAGS.train_label)
    FLAGS.image_rep_size = train_data_set.image_rep_size
    FLAGS.max_sent_size = train_data_set.max_sent_size
    FLAGS.num_mcs = train_data_set.num_mcs
    val_data_set = read_vqa(FLAGS.val_batch_size, FLAGS.val_image_rep_h5, FLAGS.val_image_idx, FLAGS.val_sent_h5, FLAGS.val_label)

    # pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=train_data_set.num_batches).start()
    tf_graph = tf.Graph()
    train_model = Model(tf_graph, FLAGS, 'train')
    tf.get_variable_scope().reuse_variables()
    # test_model = Model(tf_graph, FLAGS, 'test')
    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.initialize_all_variables())
        for epoch_idx in xrange(FLAGS.num_epochs):
            print "epoch %d" % (epoch_idx + 1)
            train_model.train(sess, train_data_set, FLAGS.learning_rate)
            train_model.test(sess, train_data_set)
            # test_model.test(sess, val_data_set)


if __name__ == "__main__":
    tf.app.run()

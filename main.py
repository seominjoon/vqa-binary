import json
import os

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
flags.DEFINE_string("train_len", "train/len.json", "len.json file path for training [train/len.json]")
flags.DEFINE_string("val_image_rep_h5", "val/image_rep.h5", "image_rep.h5 file path for validation [val/image_rep.h5]")
flags.DEFINE_string("val_image_idx", "val/image_idx.json", "image_idx.json file path for validation [val/image_idx.json]")
flags.DEFINE_string("val_sent_h5", "val/sent.h5", "sent.h5 file path for validation [val/sent.h5]")
flags.DEFINE_string("val_label", "val/label.json", "label.json file path for validation [val/label.json]")
flags.DEFINE_string("val_len", "val/len.json", "len.json file path for valing [val/len.json]")
flags.DEFINE_string("vocab_dict", "val/vocab_dict.json", "vocab_dict.json file path [val/vocab_dict.json]")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs [100]")
flags.DEFINE_float("learning_rate", 1e-2, "Learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm during trainig [40]")
flags.DEFINE_integer("num_layers", 3, "Number of LSTM layers [3]")
flags.DEFINE_integer("hidden_size", 512, "Hidden size of LSTM [512]")
flags.DEFINE_string("save_dir", "save", "Save path [save]")
flags.DEFINE_string("log_dir", "log", "Summary path [log]")
flags.DEFINE_boolean("bool_train", False, "Train? [False]")
flags.DEFINE_boolean("draft", False, "Quick iteration of epochs? [False]")
flags.DEFINE_integer("eval_num_batches", 50, "Number of batches to evaluate during training [50]")
flags.DEFINE_integer("eval_period", 3, "Evaluation period [3]")
flags.DEFINE_integer("common_size", 1024, "Common size [1024]")

FLAGS = flags.FLAGS

def main(_):
    vocab_dict = json.load(open(FLAGS.vocab_dict, 'rb'))
    FLAGS.vocab_size = len(vocab_dict)
    print "vocab size: %d" % len(vocab_dict)

    train_data_set = read_vqa(FLAGS.train_batch_size, FLAGS.train_image_rep_h5, FLAGS.train_image_idx,
                              FLAGS.train_sent_h5, FLAGS.train_len, FLAGS.train_label)
    FLAGS.image_rep_size = train_data_set.image_rep_size
    FLAGS.max_sent_size = train_data_set.max_sent_size
    FLAGS.num_mcs = train_data_set.num_mcs
    val_data_set = read_vqa(FLAGS.val_batch_size, FLAGS.val_image_rep_h5, FLAGS.val_image_idx, FLAGS.val_sent_h5, FLAGS.val_len, FLAGS.val_label)

    save_dir = FLAGS.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Time-sensitive parameters. Will be altered if draft.
    train_num_batches = train_data_set.num_batches
    eval_num_batches = FLAGS.eval_num_batches
    val_num_batches = val_data_set.num_batches
    num_epochs = FLAGS.num_epochs
    eval_period = FLAGS.eval_period
    if FLAGS.draft:
        train_num_batches = 5
        eval_num_batches = 1
        val_num_batches = 5
        num_epochs = 5
        eval_period = 1

    tf_graph = tf.Graph()
    train_model = Model(tf_graph, FLAGS, 'train')
    test_model = Model(tf_graph, FLAGS, 'test')
    writer = tf.train.SummaryWriter(FLAGS.log_dir, tf_graph.as_graph_def())
    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.bool_train:
            print "training %d epochs ..." % num_epochs
            for epoch_idx in xrange(num_epochs):
                train_model.train(sess, train_data_set, FLAGS.learning_rate, writer=writer, num_batches=train_num_batches)
                if (epoch_idx + 1) % eval_period == 0:
                    print "evaluating %d x %d examples (train data) ..." % (eval_num_batches, train_data_set.batch_size)
                    test_model.test(sess, train_data_set, num_batches=eval_num_batches)
                    print "evaluating %d x %d examples (val data) ..." % (eval_num_batches, val_data_set.batch_size)
                    test_model.test(sess, val_data_set, writer=writer, num_batches=eval_num_batches)
            print "saving model ..."
            train_model.save(sess, FLAGS.save_dir)
        else:
            print "loading model ..."
            train_model.load(sess, FLAGS.save_dir)

        print "testing %d examples (val data) ..." % val_data_set.num_examples
        test_model.test(sess, val_data_set, num_batches=val_num_batches)

        print "testing %d examples (train data) ..." % train_data_set.num_examples
        test_model.test(sess, train_data_set, num_batches=train_num_batches)




if __name__ == "__main__":
    tf.app.run()

import json
import tensorflow as tf

from pprint import pprint

from data import read_vqa, read_vqa_from_dir
from models.binary_model import BinaryModel
from models.multi_model import MultiModel

flags = tf.app.flags

# All the file directories... should get rid of these!
flags.DEFINE_string("train_dir", "train", "train data directory [train]")
flags.DEFINE_string("val_dir", "val", "validation data directory [val]")
flags.DEFINE_string("test_dir", "val", "test data directory [val]")
flags.DEFINE_string("vocab_dict_path", "train/vocab_dict.json", "vocab dict path [train/vocab_dict.json]")

# training parameters
flags.DEFINE_integer("num_epochs", 300, "Total number of epochs [300]")
flags.DEFINE_integer("batch_size", 100, "Batch size [100]")
flags.DEFINE_integer("rnn_num_layers", 3, "Number of RNN (LSTM) layers [3]")
flags.DEFINE_integer("rnn_hidden_size", 300, "Hidden size of RNN (LSTM) [300]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm during trainig [40]")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs [1]")

# training and testing options
flags.DEFINE_boolean("train", False, "Train? Test if False [False]")
flags.DEFINE_boolean("load", False, "Load from last stop? [False]")
flags.DEFINE_string("model", "multi", "Type of model? 'multi' or 'binary' [multi]")
flags.DEFINE_integer("eval_period", 3, "Evaluation period [3]")
flags.DEFINE_integer("eval_num_batches", 10, "Number of batches to evaluate during training [10]")
flags.DEFINE_integer("save_period", 1, "Save period [1]")
flags.DEFINE_string("save_dir", "save", "Save path [save]")
flags.DEFINE_string("log_dir", "log", "Log path [log]")

# for debugging
flags.DEFINE_boolean("draft", False, "Quick iteration of epochs? [False]")

FLAGS = flags.FLAGS

def main(_):
    vocab_dict = json.load(open(FLAGS.vocab_dict_path, 'rb'))
    FLAGS.vocab_size = len(vocab_dict)
    batch_size = FLAGS.batch_size
    if FLAGS.train:
        train_data_dir = FLAGS.train_dir
        val_data_dir = FLAGS.val_dir
        train_data_set = read_vqa_from_dir(batch_size, train_data_dir, name='train')
        val_data_set = read_vqa_from_dir(batch_size, val_data_dir, name='val')
        FLAGS.image_rep_size = train_data_set.image_rep_size
        FLAGS.max_sent_size = max(train_data_set.max_sent_size, val_data_set.max_sent_size)
        FLAGS.train_num_batches = train_data_set.num_batches
        assert FLAGS.eval_num_batches < min(train_data_set.num_batches, val_data_set.num_batches), "num batches should be less"
        FLAGS.eval_num_batches = FLAGS.eval_num_batches
        FLAGS.num_mcs = train_data_set.num_mcs
        if not os.path.exists(FLAGS.save_dir):
            os.mkdir(FLAGS.save_dir)
    else:
        test_data_dir = FLAGS.test_dir
        test_data_set = read_vqa_from_dir(batch_size, test_data_dir, name='test')
        FLAGS.image_rep_size = test_data_set.image_rep_size
        FLAGS.max_sent_size = test_data_set.max_sent_size
        FLAGS.test_num_batches = test_data_set.num_batches
        FLAGS.num_mcs = test_data_set.num_mcs

    # Time-sensitive parameters. Will be altered if draft.
    if FLAGS.draft:
        FLAGS.train_num_batches = 1
        FLAGS.eval_num_batches = 1
        FLAGS.test_num_batches = 1
        FLAGS.val_num_batches = 1
        FLAGS.num_epochs = 1
        FLAGS.eval_period = 1

    pprint(FLAGS.__dict__)

    model_dict = {'binary': BinaryModel,
                  'multi': MultiModel}
    model_class = model_dict[FLAGS.model]

    tf_graph = tf.Graph()
    model = model_class(tf_graph, FLAGS)
    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.train:
            writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph_def)
            if FLAGS.load:
                model.load(sess)
            model.train(sess, writer, train_data_set, FLAGS.learning_rate, val_data_set=val_data_set)
        else:
            model.load(sess)
            model.test(sess, test_data_set, num_batches=FLAGS.test_num_batches)

if __name__ == "__main__":
    tf.app.run()

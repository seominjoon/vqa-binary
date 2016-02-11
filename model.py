import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
import progressbar as pb

from data import DataSet


class Model(object):
    def __init__(self, tf_graph, params, mode):
        self.tf_graph = tf_graph
        self.params = params
        self.mode = mode
        self._build_tf_graph()

    def _build_tf_graph(self):
        params = self.params
        num_layers = params.num_layers
        batch_size = params.batch_size
        hidden_size = params.hidden_size
        max_sent_size = params.max_sent_size
        image_rep_size = params.image_rep_size
        vocab_size = params.vocab_size

        # placeholders
        with tf.name_scope("ph"):
            learning_rate = tf.placeholder('float', name='lr')
            input_sent_batch = tf.placeholder(tf.int32, [batch_size, max_sent_size], 'sent')
            input_image_rep_batch = tf.placeholder(tf.float32, [batch_size, image_rep_size], 'image_rep')
            target_batch = tf.placeholder(tf.int32, [batch_size, 2])

        # input sent embedding
        with tf.variable_scope("emb"):
            emb_mat = tf.get_variable("emb_mat", [vocab_size, hidden_size])
            x_batch = tf.nn.embedding_lookup(emb_mat, input_sent_batch)  # [N, d]

        single_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
        init_hidden_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnn'):
            x_split_batch = [tf.squeeze(x_each_batch, [1])
                             for x_each_batch in tf.split(1, max_sent_size, x_batch)]
            o_split_batch, h_last_batch = rnn.rnn(cell, x_split_batch, init_hidden_state)

        with tf.variable_scope('trans'):
            trans_mat = tf.get_variable("trans_mat", [image_rep_size, hidden_size])
            trans_bias = tf.get_variable("trans_bias", [1, hidden_size])
            m_batch = tf.matmul(input_image_rep_batch, trans_mat) + trans_bias

        # concatenate sent emb and image rep
        with tf.variable_scope('out'):
            logit_batch = h_last_batch * m_batch

        with tf.variable_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logit_batch, target_batch)
            avg_loss = tf.reduce_mean(losses)

        self.input_sent_batch = input_sent_batch
        self.input_image_batch = input_image_rep_batch
        self.target_batch = target_batch
        self.learning_rate = learning_rate
        self.avg_loss = avg_loss

        if self.mode == 'train':
            global_step = tf.get_variable("global_step", trainable=False)
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = opt.compute_gradients(losses)
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)
            self.opt_op = opt_op
            self.global_step = global_step

    def train_batch(self, sess, image_rep_batch, sent_batch, target_batch, learning_rate):
        assert self.mode == 'train', "This model is not for training!"
        feed_dict = {self.input_image_batch: image_rep_batch,
                     self.input_sent_batch: sent_batch,
                     self.target_batch: target_batch,
                     self.learning_rate: learning_rate}
        sess.run(self.opt_op, feed_dict=feed_dict)
        return None

    def train(self, sess, train_data_set, learning_rate):
        assert self.mode == 'train', 'This model is not for training!'
        assert isinstance(train_data_set, DataSet)
        params = self.params
        batch_size = params.train_batch_size
        max_sent_size = params.max_sent_size

        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=train_data_set.num_batches).start()
        for num_batches_completed in xrange(train_data_set.num_batches):
            image_rep_batch, mc_sent_batch, mc_label_batch = train_data_set.get_next_labeled_batch()
            sent_batch, label_batch = np.zeros([batch_size, max_sent_size]), np.zeros([batch_size, max_sent_size])
            for i, (mc_sent, mc_label) in enumerate(zip(mc_sent_batch, mc_label_batch)):
                correct_idx = np.argmax(mc_label)
                if np.random.randint(2) > 0:
                    sent, label = mc_sent[correct_idx], mc_label[correct_idx]
                else:
                    delta_idx = np.random.randint(params.num_mcs-1) + 1
                    new_idx = correct_idx - delta_idx
                    sent, label = mc_sent[new_idx], mc_label[new_idx]
                sent_batch[i, :] = sent
                label_batch[i, :] = label
            result = self.train_batch(sess, image_rep_batch, sent_batch, label_batch, learning_rate)
            pbar.update(num_batches_completed)
        pbar.finish()

        train_data_set.complete_epoch()
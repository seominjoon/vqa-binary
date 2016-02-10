import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn


class Model(object):
    def __init__(self, tf_graph, params, mode):
        self.tf_graph = tf_graph
        self.params = params
        self._build_tf_graph(mode)

    def _build_tf_graph(self, mode):
        params = self.params
        num_layers = params.num_layers
        batch_size = params.batch_size
        hidden_size = params.hidden_size
        max_sent_size = params.max_sent_size
        image_rep_size = params.image_rep_size
        vocab_size = params.vocab_size

        # placeholders
        with tf.name_scope("ph"):
            input_sent_batch = tf.placeholder(tf.int32, [batch_size, max_sent_size])
            input_image_batch = tf.placeholder(tf.float32, [batch_size, image_rep_size])
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
            m_batch = tf.matmul(input_image_batch, trans_mat) + trans_bias

        # concatenate sent emb and image rep
        with tf.variable_scope('out'):
            logit_batch = h_last_batch * m_batch

        with tf.variable_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logit_batch, target_batch)
            avg_loss = tf.reduce_mean(losses)

        self.input_sent_batch = input_sent_batch
        self.input_image_batch = input_image_batch
        self.target_batch = target_batch
        self.avg_loss = avg_loss

    def train(self, input_image_batch, input_sent_batch, target_batch):
        pass
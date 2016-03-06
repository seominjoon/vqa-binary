import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn
from tensorflow.python.ops import rnn_cell

from models.base_model import BaseModel


class MultiModel(BaseModel):
    def _build_tower(self):
        # Params
        params = self.params
        rnn_num_layers = params.rnn_num_layers
        rnn_hidden_size = params.rnn_hidden_size  # d
        max_sent_size = params.max_sent_size  # S
        image_rep_size = params.image_rep_size  # I
        vocab_size = params.vocab_size  # V
        batch_size = params.batch_size  # B
        num_mcs = params.num_mcs  # C

        # placeholders
        self.learning_rate = tf.placeholder('float', name='lr')
        self.mc_sent_batch = tf.placeholder(tf.int32, [batch_size, num_mcs, max_sent_size], 'sent')
        self.image_rep_batch = tf.placeholder(tf.float32, [batch_size, image_rep_size], 'image_rep')
        self.mc_len_batch = tf.placeholder(tf.int8, [batch_size, num_mcs], 'len')
        self.target_batch = tf.placeholder(tf.int32, [batch_size, num_mcs], 'target')

        summaries = []

        # input sent embedding
        with tf.variable_scope("emb"):
            with tf.device("/cpu:0"):
                emb_mat = tf.get_variable("emb_mat", [vocab_size, rnn_hidden_size])
            x_batch = tf.nn.embedding_lookup(emb_mat, self.mc_sent_batch, "emb")  # [B, C, S, d]

        with tf.variable_scope('rnn') as scope:
            single_cell = rnn_cell.BasicLSTMCell(rnn_hidden_size, forget_bias=0.0)
            cell = rnn_cell.MultiRNNCell([single_cell] * rnn_num_layers)
            init_hidden_state = cell.zero_state(batch_size * num_mcs, tf.float32)

            flat_x_batch = tf.reshape(x_batch, [batch_size * num_mcs, max_sent_size, rnn_hidden_size])
            flat_len_batch = tf.reshape(self.mc_len_batch, [batch_size * num_mcs])

            flat_x_split_batch = [tf.squeeze(flat_x_each_batch, [1])
                                  for flat_x_each_batch in tf.split(1, max_sent_size, flat_x_batch)]
            flat_o_split_batch, flat_h_last_batch = rnn.rnn(cell, flat_x_split_batch, init_hidden_state,
                sequence_length=flat_len_batch, scope=scope)

            h_last_batch = tf.reshape(flat_h_last_batch, [batch_size, num_mcs, 2 * rnn_num_layers * rnn_hidden_size])
            mc_s_batch = tf.identity(tf.split(2, 2*rnn_num_layers, h_last_batch)[2*rnn_num_layers-1], name='s')  # [B, C, d]

        with tf.variable_scope('trans'):
            image_trans_mat = tf.get_variable("image_trans_mat", [image_rep_size, rnn_hidden_size])
            image_trans_bias = tf.get_variable("image_trans_bias", [1, rnn_hidden_size])
            """
            sent_trans_mat = tf.get_variable("sent_trans_mat", [rnn_hidden_size, common_size])
            sent_trans_bias = tf.get_variable("sent_trans_bias", [1, common_size])
            """
            m_batch = tf.tanh(tf.matmul(self.image_rep_batch, image_trans_mat) + image_trans_bias, name='m')  # [B, d]
            aug_m_batch = tf.expand_dims(m_batch, 2, name='aug_m')  # [B, d, 1]
            # s_batch = tf.tanh(tf.matmul(sent_rep_batch, sent_trans_mat) + sent_trans_bias, 'sent_trans')
            # logit_batch = tf.squeeze(tf.batch_matmul(mc_s_batch, aug_m_batch), [2], name='logit')  # [B, C]
            logit_batch = tf.reduce_sum(mc_s_batch, 2)
            p_batch = tf.nn.softmax(logit_batch, 'p')

        with tf.name_scope('loss') as loss_scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_batch, tf.cast(self.target_batch, 'float'), name='cross_entropy')
            avg_cross_entropy = tf.reduce_mean(cross_entropy, 0, name='avg_cross_entropy')
            tf.add_to_collection('losses', avg_cross_entropy)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            losses = tf.get_collection('losses', loss_scope)
            ema = tf.train.ExponentialMovingAverage(0.9, name='ema')
            ema_op = ema.apply(losses + [total_loss])

        with tf.name_scope('opt'):
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            # loss_averages_op = ema.apply(losses + [total_loss])
            # with tf.control_dependencies([loss_averages_op]):
            # opt = tf.train.AdagradOptimizer(learning_rate)
            grads_and_vars = opt.compute_gradients(cross_entropy)
            # clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)

        with tf.name_scope('acc'):
            correct_vec = tf.equal(tf.argmax(p_batch, 1), tf.argmax(self.target_batch, 1))
            num_corrects = tf.reduce_sum(tf.cast(correct_vec, 'int32'), name='num_corrects')
            acc = tf.reduce_mean(tf.cast(correct_vec, 'float'), name='acc')


        # summaries
        summaries.append(tf.scalar_summary("%s (raw)" % total_loss.op.name, total_loss))
        summaries.append(tf.scalar_summary(total_loss.op.name, ema.average(total_loss)))
        for grad, var in grads_and_vars:
            if grad: summaries.append(tf.histogram_summary("%s/grad" % var.op.name, grad))
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        merged_summary = tf.merge_summary(summaries)

        # Storing tensors
        self.opt_op = opt_op
        self.logit_batch = logit_batch
        self.total_loss = total_loss
        self.acc = acc
        self.correct_vec = correct_vec
        self.num_corrects = num_corrects
        self.merged_summary = merged_summary

    def train_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, target_batch, learning_rate):
        feed_dict = self._get_feed_dict(image_rep_batch, mc_sent_batch, mc_len_batch, target_batch=target_batch)
        feed_dict[self.learning_rate] = learning_rate
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def _pad(self, array, inc):
        assert len(array.shape) > 0, "Array must be at least 1D!"
        if len(array.shape) == 1:
            return np.concatenate([array, np.zeros([inc])], 0)
        else:
            return np.concatenate([array, np.zeros([inc, array.shape[1]])], 0)

    def test_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, target_batch):
        params = self.params
        batch_size = params.batch_size

        actual_batch_size = image_rep_batch.shape[0]
        if actual_batch_size < batch_size:
            diff = batch_size - actual_batch_size
            image_rep_batch = self._pad(image_rep_batch, diff)
            mc_sent_batch = self._pad(mc_sent_batch, diff)
            mc_len_batch = self._pad(mc_len_batch, diff)
            target_batch = self._pad(target_batch, diff)

        feed_dict = self._get_feed_dict(image_rep_batch, mc_sent_batch, mc_len_batch, target_batch=target_batch)
        correct_vec, total_loss, summary_str, global_step = \
            sess.run([self.correct_vec, self.total_loss, self.merged_summary, self.global_step], feed_dict=feed_dict)
        num_corrects = np.sum(correct_vec[:actual_batch_size])

        return num_corrects, total_loss, summary_str, global_step

    def _get_feed_dict(self, image_rep_batch, sent_batch, len_batch, target_batch=None):
        if target_batch is None:
            target_batch = np.zeros([image_rep_batch.shape[0], 2])
        feed_dict = {self.image_rep_batch: image_rep_batch,
                     self.mc_sent_batch: sent_batch,
                     self.mc_len_batch: len_batch,
                     self.target_batch: target_batch}
        return feed_dict

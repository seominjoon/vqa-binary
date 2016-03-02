import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np

from models.base_model import BaseModel


class BinaryModel(BaseModel):
    def _build_tower(self):
        # Params
        params = self.params
        rnn_num_layers = params.rnn_num_layers
        rnn_hidden_size = params.rnn_hidden_size
        max_sent_size = params.max_sent_size
        image_rep_size = params.image_rep_size
        vocab_size = params.vocab_size
        batch_size = params.batch_size

        # placeholders
        self.learning_rate = tf.placeholder('float', name='lr')
        self.input_sent_batch = tf.placeholder(tf.int32, [batch_size, max_sent_size], 'sent')
        self.input_image_rep_batch = tf.placeholder(tf.float32, [batch_size, image_rep_size], 'image_rep')
        self.input_len_batch = tf.placeholder(tf.int8, [batch_size], 'len')
        self.target_batch = tf.placeholder(tf.int32, [batch_size, 2], 'target')

        summaries = []

        # input sent embedding
        with tf.variable_scope("emb"):
            with tf.device("/cpu:0"):
                emb_mat = tf.get_variable("emb_mat", [vocab_size, rnn_hidden_size])
            x_batch = tf.nn.embedding_lookup(emb_mat, self.input_sent_batch, "emb")  # [N, d]

        with tf.variable_scope('rnn') as scope:
            single_cell = rnn_cell.BasicLSTMCell(rnn_hidden_size, forget_bias=0.0)
            cell = rnn_cell.MultiRNNCell([single_cell] * rnn_num_layers)
            init_hidden_state = cell.zero_state(batch_size, tf.float32)
            x_split_batch = [tf.squeeze(x_each_batch, [1])
                             for x_each_batch in tf.split(1, max_sent_size, x_batch)]
            o_split_batch, h_last_batch = rnn.rnn(cell, x_split_batch, init_hidden_state,
                                                  sequence_length=self.input_len_batch, scope=scope)

        with tf.variable_scope('trans'):
            image_trans_mat = tf.get_variable("image_trans_mat", [image_rep_size, rnn_hidden_size])
            image_trans_bias = tf.get_variable("image_trans_bias", [1, rnn_hidden_size])
            """
            sent_trans_mat = tf.get_variable("sent_trans_mat", [rnn_hidden_size, common_size])
            sent_trans_bias = tf.get_variable("sent_trans_bias", [1, common_size])
            """
            m_batch = tf.tanh(tf.matmul(self.input_image_rep_batch, image_trans_mat) + image_trans_bias, 'image_trans')
            sent_rep_batch = tf.split(1, 2*rnn_num_layers, h_last_batch)[2*rnn_num_layers-1]
            # s_batch = tf.tanh(tf.matmul(sent_rep_batch, sent_trans_mat) + sent_trans_bias, 'sent_trans')

        # concatenate sent emb and image rep
        with tf.variable_scope("class"):
            class_mat = tf.get_variable("class_mat", [rnn_hidden_size, 2])
            logit_batch = tf.matmul(sent_rep_batch * m_batch, class_mat, name='logit')

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
            grads_and_vars = opt.compute_gradients(total_loss)
            # clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)

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
        self.avg_cross_entropy = avg_cross_entropy
        self.merged_summary = merged_summary

    def train_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch, learning_rate):
        params = self.params
        batch_size = params.batch_size
        max_sent_size = params.max_sent_size
        sent_batch, len_batch, target_batch = np.zeros([batch_size, max_sent_size]), np.zeros([batch_size]), np.zeros([batch_size, 2])
        for i, (mc_sent, mc_len, mc_label) in enumerate(zip(mc_sent_batch, mc_len_batch, mc_label_batch)):
            correct_idx = np.argmax(mc_label)
            val = np.random.randint(2)
            if val > 0.5:
                sent, len_, label = mc_sent[correct_idx], mc_len[correct_idx], mc_label[correct_idx]
            else:
                delta_idx = np.random.randint(params.num_mcs-1) + 1
                new_idx = correct_idx - delta_idx
                sent, len_, label = mc_sent[new_idx], mc_len[new_idx], mc_label[new_idx]
            target = np.array([0, 1]) if label else np.array([1, 0])
            sent_batch[i, :] = sent
            len_batch[i] = len_
            target_batch[i, :] = target
        feed_dict = self._get_feed_dict(image_rep_batch, sent_batch, len_batch, target_batch=target_batch)
        feed_dict[self.learning_rate] = learning_rate
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def _pad(self, array, inc):
        assert len(array.shape) > 0, "Array must be at least 1D!"
        if len(array.shape) == 1:
            return np.concatenate([array, np.zeros([inc])], 0)
        else:
            return np.concatenate([array, np.zeros([inc, array.shape[1]])], 0)


    def test_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch):
        params = self.params
        batch_size = params.batch_size
        num_mcs = params.num_mcs

        actual_batch_size = image_rep_batch.shape[0]
        if actual_batch_size < batch_size:
            diff = batch_size - actual_batch_size
            image_rep_batch = self._pad(image_rep_batch, diff)
            mc_sent_batch = self._pad(mc_sent_batch, diff)
            mc_len_batch = self._pad(mc_len_batch, diff)
            mc_label_batch = self._pad(mc_label_batch, diff)

        mc_prob_batch = np.zeros([batch_size, num_mcs])

        losses = []
        for mc_idx in range(num_mcs):
            sent_batch = mc_sent_batch[:, mc_idx]
            len_batch = mc_len_batch[:, mc_idx]
            feed_dict = self._get_feed_dict(image_rep_batch, sent_batch, len_batch)
            logit_batch, cur_loss, summary_str, global_step = sess.run([self.logit_batch, self.avg_cross_entropy, self.merged_summary, self.global_step], feed_dict=feed_dict)
            mc_prob_batch[:, mc_idx] = logit_batch[:, 1]
            losses.append(cur_loss)
        mc_pred_batch = np.argmax(mc_prob_batch, 1)
        mc_true_batch = np.argmax(mc_label_batch, 1)
        num_corrects = np.sum((mc_pred_batch == mc_true_batch)[:actual_batch_size])
        loss = np.mean(losses)

        return num_corrects, loss, summary_str, global_step

    def _get_feed_dict(self, image_rep_batch, sent_batch, len_batch, target_batch=None):
        if target_batch is None:
            target_batch = np.zeros([image_rep_batch.shape[0], 2])
        feed_dict = {self.input_image_rep_batch: image_rep_batch,
                     self.input_sent_batch: sent_batch,
                     self.input_len_batch: len_batch,
                     self.target_batch: target_batch}
        return feed_dict


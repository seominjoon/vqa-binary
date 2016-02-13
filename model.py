import os

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
        with tf_graph.as_default():
            self._build_tf_graph()
            self.saver = tf.train.Saver()

    def _build_tf_graph(self):
        params = self.params
        num_layers = params.num_layers
        hidden_size = params.hidden_size
        max_sent_size = params.max_sent_size
        image_rep_size = params.image_rep_size
        vocab_size = params.vocab_size
        common_size = params.common_size

        summaries = []
        global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_step = global_step

        if self.mode == 'train':
            batch_size = params.train_batch_size
        elif self.mode == 'test':
            batch_size = params.num_mcs

        # placeholders
        with tf.name_scope("%s/ph" % self.mode):
            if self.mode == 'train':
                learning_rate = tf.placeholder('float', name='lr')
            input_sent_batch = tf.placeholder(tf.int32, [batch_size, max_sent_size], 'sent')
            input_image_rep_batch = tf.placeholder(tf.float32, [batch_size, image_rep_size], 'image_rep')
            input_len_batch = tf.placeholder(tf.int8, [batch_size], 'len')
            target_batch = tf.placeholder(tf.int32, [batch_size, 2])

        # input sent embedding
        with tf.variable_scope("emb", reuse=self.mode=='test'):
            emb_mat = tf.get_variable("emb_mat", [vocab_size, hidden_size])
            x_batch = tf.nn.embedding_lookup(emb_mat, input_sent_batch)  # [N, d]

        single_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
        init_hidden_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnn', reuse=self.mode=='test'):
            x_split_batch = [tf.squeeze(x_each_batch, [1])
                             for x_each_batch in tf.split(1, max_sent_size, x_batch)]
            o_split_batch, h_last_batch = rnn.rnn(cell, x_split_batch, init_hidden_state, sequence_length=input_len_batch)

        with tf.variable_scope('trans', reuse=self.mode=='test'):
            image_trans_mat = tf.get_variable("image_trans_mat", [image_rep_size, common_size])
            image_trans_bias = tf.get_variable("image_trans_bias", [1, common_size])
            m_batch = tf.tanh(tf.matmul(input_image_rep_batch, image_trans_mat) + image_trans_bias)

            sent_trans_mat = tf.get_variable("sent_trans_mat", [hidden_size, common_size])
            sent_trans_bias = tf.get_variable("sent_trans_bias", [1, common_size])
            s_batch = tf.tanh(tf.matmul(tf.split(1, 2*num_layers, h_last_batch)[2*num_layers-1], sent_trans_mat) + sent_trans_bias)

        # concatenate sent emb and image rep
        with tf.variable_scope('out', reuse=self.mode=='test'):
            # logit_batch = h_last_batch * m_batch
            class_mat = tf.get_variable("class_mat", [common_size, 2])
            logit_batch = tf.matmul(s_batch * m_batch, class_mat)

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logit_batch, tf.cast(target_batch, 'float'))
            avg_loss = tf.reduce_mean(losses)
            summaries.append(tf.scalar_summary('%s_avg_loss' % self.mode, avg_loss))

        self.input_sent_batch = input_sent_batch
        self.input_image_batch = input_image_rep_batch
        self.input_len_batch = input_len_batch
        self.target_batch = target_batch
        self.avg_loss = avg_loss

        if self.mode == 'train':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = opt.compute_gradients(losses)
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)
            self.opt_op = opt_op
            self.learning_rate = learning_rate
        elif self.mode == 'test':
            prob_batch = tf.reshape(tf.slice(logit_batch, [0, 1], [-1, 1]), [-1])
            label_batch = tf.reshape(tf.slice(target_batch, [0, 1], [-1, 1]), [-1])
            correct = tf.equal(tf.argmax(prob_batch, 0), tf.argmax(label_batch, 0))
            self.correct = correct
            self.prob_batch =prob_batch
            self.label_batch = label_batch

        self.summary = tf.merge_summary(summaries)

    def _get_feed_dict(self, image_rep_batch, sent_batch, len_batch, target_batch=None):
        feed_dict = {self.input_image_batch: image_rep_batch,
                     self.input_sent_batch: sent_batch,
                     self.input_len_batch: len_batch}
        if target_batch is not None:
            feed_dict[self.target_batch] = target_batch
        return feed_dict

    def train_batch(self, sess, image_rep_batch, sent_batch, len_batch, target_batch, learning_rate):
        assert self.mode == 'train', "This model is not for training!"
        feed_dict = self._get_feed_dict(image_rep_batch, sent_batch, len_batch, target_batch)
        feed_dict[self.learning_rate] = learning_rate
        return sess.run([self.opt_op, self.summary, self.global_step], feed_dict=feed_dict)

    def train(self, sess, train_data_set, learning_rate, writer=None, num_batches=None):
        if num_batches is None:
            num_batches = train_data_set.num_batches
        assert self.mode == 'train', 'This model is not for training!'
        assert isinstance(train_data_set, DataSet)
        params = self.params
        batch_size = params.train_batch_size
        max_sent_size = params.max_sent_size

        pbar = pb.ProgressBar(widgets=["epoch %d:" % (train_data_set.num_epochs_completed + 1),
                                       pb.Percentage(), pb.Bar(), pb.ETA()], maxval=train_data_set.num_batches)
        pbar.start()
        for num_batches_completed in xrange(num_batches):
            image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch = train_data_set.get_next_labeled_batch()
            sent_batch, len_batch, target_batch = np.zeros([batch_size, max_sent_size]), np.zeros([batch_size]), np.zeros([batch_size, 2])
            for i, (mc_sent, mc_len, mc_label) in enumerate(zip(mc_sent_batch, mc_len_batch, mc_label_batch)):
                correct_idx = np.argmax(mc_label)
                if np.random.randint(2) > 0:
                    sent, len_, label = mc_sent[correct_idx], mc_len[correct_idx], mc_label[correct_idx]
                else:
                    delta_idx = np.random.randint(params.num_mcs-1) + 1
                    new_idx = correct_idx - delta_idx
                    sent, len_, label = mc_sent[new_idx], mc_len[new_idx], mc_label[new_idx]
                target = np.array([0, 1]) if label else np.array([1, 0])
                sent_batch[i, :] = sent
                len_batch[i] = len_
                target_batch[i, :] = target
            result = self.train_batch(sess, image_rep_batch, sent_batch, len_batch, target_batch, learning_rate)
            summary_str, global_step = result[1], result[2]
            if writer: writer.add_summary(summary_str, global_step)
            pbar.update(num_batches_completed)
        pbar.finish()

        train_data_set.complete_epoch()

    def test(self, sess, test_data_set, num_batches=None, writer=None):
        assert isinstance(test_data_set, DataSet)

        if num_batches is None:
            num_batches = test_data_set.num_batches

        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches).start()
        num_corrects = 0
        total_avg_loss = 0
        for num_batches_completed in xrange(num_batches):
            image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch = test_data_set.get_next_labeled_batch()
            for image_rep, mc_sent, mc_len, mc_label in zip(image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch):
                mc_image_rep = np.tile(image_rep, [len(mc_sent), 1])
                mc_target = np.array([[0, 1] if label else [1, 0] for label in mc_label])
                feed_dict = self._get_feed_dict(mc_image_rep, mc_sent, mc_len, mc_target)
                correct, each_avg_loss, summary_str, global_step= sess.run([self.correct, self.avg_loss, self.summary, self.global_step], feed_dict=feed_dict)
                num_corrects += correct
                total_avg_loss += each_avg_loss
            pbar.update(num_batches_completed)
        pbar.finish()
        test_data_set.reset()
        total = num_batches * test_data_set.batch_size
        acc = float(num_corrects)/total
        avg_loss = total_avg_loss/total
        if writer: writer.add_summary(summary_str, global_step)
        """
        summary = tf.scalar_summary("eval_%s_avg_loss" % self.mode, avg_loss)
        if writer: writer.add_summary(summary.eval(session=sess), global_step)
        """
        print "%d/%d = %.4f, loss=%.4f" % (num_corrects, total, acc, avg_loss)

    def save(self, sess, save_dir):
        save_path = os.path.join(save_dir, 'model')
        self.saver.save(sess, save_path, self.global_step)

    def load(self, sess, save_dir):
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

import os

import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
import progressbar as pb

from data import DataSet


class Model(object):
    def __init__(self, tf_graph, params, writer, name=None):
        self.tf_graph = tf_graph
        self.params = params
        self.save_dir = params.save_dir
        self.name = name if name else self.__class__.__name__
        self.writer = writer
        with tf_graph.as_default():
            self._build_tf_graph()
            self.saver = tf.train.Saver()


    def _build_tf_graph(self):
        print "building graph ..."
        # Params
        params = self.params
        rnn_num_layers = params.rnn_num_layers
        rnn_hidden_size = params.rnn_hidden_size
        max_sent_size = params.max_sent_size
        image_rep_size = params.image_rep_size
        vocab_size = params.vocab_size
        common_size = params.common_size
        batch_size = params.batch_size
        num_mcs = params.num_mcs

        with tf.variable_scope(self.name) as model_scope:
            global_step = tf.Variable(0, name='global_step', trainable=False)
            summaries = []

            # placeholders
            with tf.name_scope("ph"):
                learning_rate = tf.placeholder('float', name='lr')
                input_sent_batch = tf.placeholder(tf.int32, [batch_size, num_mcs, max_sent_size], 'sent')
                input_image_rep_batch = tf.placeholder(tf.float32, [batch_size, num_mcs, image_rep_size], 'image_rep')
                input_len_batch = tf.placeholder(tf.int8, [batch_size, num_mcs], 'len')
                target_batch = tf.placeholder(tf.int32, [batch_size, 2], 'target')

            # input sent embedding
            with tf.variable_scope("emb"):
                with tf.device("/cpu:0"):
                    emb_mat = tf.get_variable("emb_mat", [vocab_size, rnn_hidden_size])
                x_batch = tf.nn.embedding_lookup(emb_mat, input_sent_batch, "emb")  # [N, d]

            with tf.variable_scope('rnn') as scope:
                single_cell = rnn_cell.BasicLSTMCell(rnn_hidden_size, forget_bias=0.0)
                cell = rnn_cell.MultiRNNCell([single_cell] * rnn_num_layers)
                init_hidden_state = cell.zero_state(batch_size, tf.float32)
                x_split_batch = [tf.squeeze(x_each_batch, [1])
                                 for x_each_batch in tf.split(1, max_sent_size, x_batch)]
                o_split_batch, h_last_batch = rnn.rnn(cell, x_split_batch, init_hidden_state,
                                                      sequence_length=input_len_batch, scope=scope)

            with tf.variable_scope('trans'):
                image_trans_mat = tf.get_variable("image_trans_mat", [image_rep_size, common_size])
                image_trans_bias = tf.get_variable("image_trans_bias", [1, common_size])
                sent_trans_mat = tf.get_variable("sent_trans_mat", [rnn_hidden_size, common_size])
                sent_trans_bias = tf.get_variable("sent_trans_bias", [1, common_size])
                m_batch = tf.tanh(tf.matmul(input_image_rep_batch, image_trans_mat) + image_trans_bias, 'image_trans')
                sent_rep_batch = tf.split(1, 2*rnn_num_layers, h_last_batch)[2*rnn_num_layers-1]
                s_batch = tf.tanh(tf.matmul(sent_rep_batch, sent_trans_mat) + sent_trans_bias, 'sent_trans')

            # concatenate sent emb and image rep
            with tf.variable_scope("class"):
                class_mat = tf.get_variable("class_mat", [common_size, 2])
                logit_batch = tf.matmul(s_batch * m_batch, class_mat, name='logit')

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_batch, tf.cast(target_batch, 'float'), name='cross_entropy')
                avg_cross_entropy = tf.reduce_mean(cross_entropy, name='avg_cross_entropy')
                tf.add_to_collection('losses', avg_cross_entropy)
                total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
                losses = tf.get_collection('losses')
                ema = tf.train.ExponentialMovingAverage(0.9, name='ema')
                ema_op = ema.apply(losses + [total_loss])
                summaries.append(tf.scalar_summary(total_loss.op.name, total_loss))
                summaries.append(tf.scalar_summary(total_loss.op.name + '(moving)', ema.average(total_loss)))

            with tf.name_scope('opt'):
                # loss_averages_op = ema.apply(losses + [total_loss])
                # with tf.control_dependencies([loss_averages_op]):
                # opt = tf.train.GradientDescentOptimizer(learning_rate)
                opt = tf.train.AdagradOptimizer(learning_rate)
                grads_and_vars = opt.compute_gradients(total_loss)
                # clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
                opt_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

            merged_summary = tf.merge_summary(summaries)

            # Storing variables
            self.global_step = global_step

            # Storing tensors
            self.opt_op = opt_op
            self.learning_rate = learning_rate
            self.logit_batch = logit_batch
            self.input_sent_batch = input_sent_batch
            self.input_image_batch = input_image_rep_batch
            self.input_len_batch = input_len_batch
            self.target_batch = target_batch
            self.total_loss = total_loss
            self.merged_summary = merged_summary

    def _get_feed_dict(self, image_rep_batch, sent_batch, len_batch, target_batch=None):
        feed_dict = {self.input_image_batch: image_rep_batch,
                     self.input_sent_batch: sent_batch,
                     self.input_len_batch: len_batch}
        if target_batch is not None:
            feed_dict[self.target_batch] = target_batch
        return feed_dict

    def train_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch, learning_rate):
        params = self.params
        batch_size = params.batch_size
        max_sent_size = params.max_sent_size
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
        feed_dict = self._get_feed_dict(image_rep_batch, sent_batch, len_batch, target_batch)
        feed_dict[self.learning_rate] = learning_rate
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def train(self, sess, train_data_set, learning_rate, val_data_set=None):
        assert isinstance(train_data_set, DataSet)
        params = self.params
        num_batches = params.train_num_batches
        batch_size = params.batch_size

        print "training %d epochs ..." % params.num_epochs
        for epoch_idx in xrange(params.num_epochs):
            pbar = pb.ProgressBar(widgets=["epoch %d:" % (train_data_set.num_epochs_completed + 1),
                                           pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
            pbar.start()
            for num_batches_completed in xrange(num_batches):
                image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch = train_data_set.get_next_labeled_batch()
                _, summary_str, global_step = self.train_batch(sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_len_batch, learning_rate)
                self.writer.add_summary(summary_str, global_step)
                pbar.update(num_batches_completed)
            pbar.finish()
            train_data_set.complete_epoch()

            if val_data_set and (epoch_idx + 1) % params.eval_period == 0:
                print "evaluating %d x %d examples (train data) ..." % (params.eval_num_batches, batch_size)
                self.test(sess, train_data_set, num_batches=params.eval_num_batches)
                print "evaluating %d x %d examples (val data) ..." % (params.eval_num_batches, batch_size)
                self.test(sess, val_data_set, num_batches=params.eval_num_batches)

            if (epoch_idx + 1) % params.save_period == 0:
                print "saving model ..."
                self.save(sess)

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

        if image_rep_batch.shape[0] < batch_size:
            early_stop = image_rep_batch.shape[0]
            diff = batch_size - image_rep_batch.shape[0]
            image_rep_batch = self._pad(image_rep_batch, diff)
            mc_sent_batch = self._pad(mc_sent_batch, diff)
            mc_len_batch = self._pad(mc_len_batch, diff)
            mc_label_batch = self._pad(mc_label_batch, diff)
        else:
            early_stop = None

        mc_prob_batch = np.zeros([batch_size, num_mcs])

        for mc_idx in xrange(num_mcs):
            sent_batch = mc_sent_batch[:, mc_idx]
            len_batch = mc_len_batch[:, mc_idx]
            feed_dict = self._get_feed_dict(image_rep_batch, sent_batch, len_batch)
            logit_batch, summary_str, global_step = sess.run([self.logit_batch, self.merged_summary, self.global_step], feed_dict=feed_dict)
            mc_prob_batch[:, mc_idx] = logit_batch[:, 1]
        mc_pred_batch = np.argmax(mc_prob_batch, 1)
        mc_true_batch = np.argmax(mc_label_batch, 1)
        if early_stop:
            mc_pred_batch = mc_pred_batch[:early_stop]
            mc_true_batch = mc_pred_batch[:early_stop]
        num_corrects = np.sum(mc_pred_batch, mc_true_batch)

        return num_corrects, summary_str, global_step

    def test(self, sess, test_data_set, num_batches=None):
        num_batches = num_batches if num_batches else test_data_set.num_batches
        num_corrects, total = 0, 0
        print "testing %d batches x %d examples (%s)... (last batch might contain less examples)" % \
              (num_batches, test_data_set.batch_size, test_data_set.name)
        pbar = pb.ProgressBar(widgets=["epoch %d:" % (test_data_set.num_epochs_completed + 1),
                                       pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
        for num_batches_completed in num_batches:
            image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch = test_data_set.get_next_labeled_batch()
            cur_num_corrects, _, global_step = self.test_batch(sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch)
            num_corrects += cur_num_corrects
            total += len(image_rep_batch)
            pbar.update(num_batches_completed + 1)
        pbar.finish()
        test_data_set.complete_epoch()

        print "acc at %d: %.2f%% = %d / %d" % (global_step, 100 * float(num_corrects)/total, num_corrects, total)

    def save(self, sess):
        print "saving model ..."
        save_path = os.path.join(self.save_dir, self.name)
        self.saver.save(sess, save_path, self.global_step)

    def load(self, sess):
        print "loading model ..."
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

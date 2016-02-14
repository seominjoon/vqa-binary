import os

import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
import progressbar as pb

from data import DataSet


class Model(object):
    class Tensors(object):
        pass

    def __init__(self, tf_graph, params, writer, name=None):
        self.tf_graph = tf_graph
        self.params = params
        self.save_dir = params.save_dir
        self.name = name if name else self.__class__.__name__
        self.writer = writer
        with tf_graph.as_default():
            self.train_tensors = self._build_tf_graph('train')
            self.saver = tf.train.Saver()
            self.test_tensors = self._build_tf_graph('test')


    def _build_tf_graph(self, mode):
        # Params
        params = self.params
        rnn_num_layers = params.rnn_num_layers
        rnn_hidden_size = params.rnn_hidden_size
        max_sent_size = params.max_sent_size
        image_rep_size = params.image_rep_size
        vocab_size = params.vocab_size
        common_size = params.common_size
        if mode == 'train':
            batch_size = params.train_batch_size
        elif mode == 'test':
            batch_size = params.num_mcs
        else:
            raise Exception("Unknown mode: %s" % mode)

        if mode == 'train': global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('variable', reuse=(mode=='test')) as shared_scope:
            emb_mat = tf.get_variable("emb_mat", [vocab_size, rnn_hidden_size])
            image_trans_mat = tf.get_variable("image_trans_mat", [image_rep_size, common_size])
            image_trans_bias = tf.get_variable("image_trans_bias", [1, common_size])
            sent_trans_mat = tf.get_variable("sent_trans_mat", [rnn_hidden_size, common_size])
            sent_trans_bias = tf.get_variable("sent_trans_bias", [1, common_size])
            class_mat = tf.get_variable("class_mat", [common_size, 2])

        with tf.name_scope(mode) as mode_scope:
            summaries = []

            # placeholders
            with tf.name_scope("placeholder"):
                if mode == 'train': learning_rate = tf.placeholder('float', name='lr')
                input_sent_batch = tf.placeholder(tf.int32, [batch_size, max_sent_size], 'sent')
                input_image_rep_batch = tf.placeholder(tf.float32, [batch_size, image_rep_size], 'image_rep')
                input_len_batch = tf.placeholder(tf.int8, [batch_size], 'len')
                target_batch = tf.placeholder(tf.int32, [batch_size, 2], 'target')

            # input sent embedding
            x_batch = tf.nn.embedding_lookup(emb_mat, input_sent_batch, "emb")  # [N, d]

            with tf.name_scope('rnn'):
                single_cell = rnn_cell.BasicLSTMCell(rnn_hidden_size, forget_bias=0.0)
                cell = rnn_cell.MultiRNNCell([single_cell] * rnn_num_layers)
                init_hidden_state = cell.zero_state(batch_size, tf.float32)
                x_split_batch = [tf.squeeze(x_each_batch, [1])
                                 for x_each_batch in tf.split(1, max_sent_size, x_batch)]
                o_split_batch, h_last_batch = rnn.rnn(cell, x_split_batch, init_hidden_state,
                                                      sequence_length=input_len_batch, scope=shared_scope)

            with tf.name_scope('trans'):
                m_batch = tf.tanh(tf.matmul(input_image_rep_batch, image_trans_mat) + image_trans_bias, 'image_trans')
                sent_rep_batch = tf.split(1, 2*rnn_num_layers, h_last_batch)[2*rnn_num_layers-1]
                s_batch = tf.tanh(tf.matmul(sent_rep_batch, sent_trans_mat) + sent_trans_bias, 'sent_trans')

            # concatenate sent emb and image rep
            logit_batch = tf.matmul(s_batch * m_batch, class_mat, name='logit')

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_batch, tf.cast(target_batch, 'float'), name='cross_entropy')
                avg_cross_entropy = tf.reduce_mean(cross_entropy, name='avg_cross_entropy')
                tf.add_to_collection('losses', avg_cross_entropy)
                total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
                losses = tf.get_collection('losses')
                ema = tf.train.ExponentialMovingAverage(0.9, name='ema')
                summaries.append(tf.scalar_summary(total_loss.op.name, total_loss))
                summaries.append(tf.scalar_summary(total_loss.op.name + '(moving)', ema.average(total_loss)))

            if mode == 'train':
                with tf.name_scope('opt'):
                    # loss_averages_op = ema.apply(losses + [total_loss])
                    # with tf.control_dependencies([loss_averages_op]):
                    # opt = tf.train.GradientDescentOptimizer(learning_rate)
                    opt = tf.train.AdagradOptimizer(learning_rate)
                    grads_and_vars = opt.compute_gradients(losses)
                    # clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
                    opt_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

            elif mode == 'test':
                with tf.name_scope('eval'):
                    prob_batch = tf.reshape(tf.slice(logit_batch, [0, 1], [-1, 1]), [-1])
                    label_batch = tf.reshape(tf.slice(target_batch, [0, 1], [-1, 1]), [-1])
                    correct = tf.equal(tf.argmax(prob_batch, 0), tf.argmax(label_batch, 0))

            else:
                raise Exception("Unknown mode: %s" % mode)

            merged_summary = tf.merge_summary(summaries)

            # Storing variables
            if mode == 'train':
                self.global_step = global_step

            # Storing tensors
            tensors = Model.Tensors()
            if mode == 'train':
                tensors.opt_op = opt_op
                tensors.learning_rate = learning_rate
            elif mode == 'test':
                tensors.correct = correct
                tensors.prob_batch =prob_batch
                tensors.label_batch = label_batch
            else:
                raise Exception("Unknown mode: %s" % mode)
            tensors.input_sent_batch = input_sent_batch
            tensors.input_image_batch = input_image_rep_batch
            tensors.input_len_batch = input_len_batch
            tensors.target_batch = target_batch
            tensors.avg_loss = avg_loss
            tensors.merged_summary = merged_summary

    def _get_feed_dict(self, tensors, image_rep_batch, sent_batch, len_batch, target_batch=None):
        feed_dict = {tensors.input_image_batch: image_rep_batch,
                     tensors.input_sent_batch: sent_batch,
                     tensors.input_len_batch: len_batch}
        if target_batch is not None:
            feed_dict[tensors.target_batch] = target_batch
        return feed_dict

    def train_batch(self, sess, image_rep_batch, sent_batch, len_batch, target_batch, learning_rate):
        tensors = self.train_tensors
        feed_dict = self._get_feed_dict(tensors, image_rep_batch, sent_batch, len_batch, target_batch)
        feed_dict[tensors.learning_rate] = learning_rate
        return sess.run([tensors.opt_op, tensors.merged_summary, self.global_step], feed_dict=feed_dict)

    def train(self, sess, train_data_set, learning_rate, val_data_set=None):
        assert isinstance(train_data_set, DataSet)
        params = self.params
        num_batches = params.train_num_batches
        batch_size = params.train_batch_size
        max_sent_size = params.max_sent_size

        print "training %d epochs ..." % params.num_epochs
        for epoch_idx in xrange(params.num_epochs):
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
                self.writer.add_summary(summary_str, global_step)
                pbar.update(num_batches_completed)
            pbar.finish()
            train_data_set.complete_epoch()

            if val_data_set and (epoch_idx + 1) % params.eval_period == 0:
                print "evaluating %d x %d examples (train data) ..." % (params.eval_num_batches, train_data_set.batch_size)
                self.test(sess, train_data_set, num_batches=params.eval_num_batches)
                print "evaluating %d x %d examples (val data) ..." % (params.eval_num_batches, val_data_set.batch_size)
                self.test(sess, val_data_set, writer=writer, num_batches=params.eval_num_batches)

            if (epoch_idx + 1) % params.save_period == 0:
                print "saving model ..."
                self.save(sess)

    def test(self, sess, test_data_set, num_batches=None):
        assert isinstance(test_data_set, DataSet)

        tensors = self.test_tensors

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
                feed_dict = self._get_feed_dict(tensors, mc_image_rep, mc_sent, mc_len, mc_target)
                correct, each_avg_loss, summary_str, global_step \
                    = sess.run([tensors.correct, tensors.avg_loss, tensors.merged_summary, self.global_step], feed_dict=feed_dict)
                num_corrects += correct
                total_avg_loss += each_avg_loss
            pbar.update(num_batches_completed)
        pbar.finish()
        test_data_set.reset()
        total = num_batches * test_data_set.batch_size
        acc = float(num_corrects)/total
        avg_loss = total_avg_loss/total
        self.writer.add_summary(summary_str, self.global_step)
        print "%d/%d = %.4f, loss=%.4f" % (num_corrects, total, acc, avg_loss)

    def save(self, sess):
        print "saving model ..."
        save_path = os.path.join(self.save_dir, self.name)
        self.saver.save(sess, save_path, self.global_step)

    def load(self, sess):
        print "loading model ..."
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

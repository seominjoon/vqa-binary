import os

import tensorflow as tf
import progressbar as pb
import numpy as np

from data import DataSet


class BaseModel(object):
    def __init__(self, tf_graph, params, name=None):
        self.tf_graph = tf_graph
        self.params = params
        self.save_dir = params.save_dir
        self.name = name if name else self.__class__.__name__
        self.initializer = tf.random_normal_initializer(0, 0.1)
        with tf_graph.as_default(), tf.variable_scope(self.name, initializer=self.initializer):
            print("building %s graph ..." % self.name)
            self.global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0), trainable=False)
            self._build_tower()
            self.saver = tf.train.Saver()

    def _build_tower(self):
        raise Exception("Implement this function!")

    def train_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch, learning_rate):
        raise Exception("Implement this function!")

    def test_batch(self, sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch):
        raise Exception("Implement this function!")

    def train(self, sess, writer, train_data_set, learning_rate, val_data_set=None):
        assert isinstance(train_data_set, DataSet)
        params = self.params
        num_batches = params.train_num_batches
        batch_size = params.batch_size
        max_sent_size = params.max_sent_size
        num_mcs = params.num_mcs

        print("training %d epochs ..." % params.num_epochs)
        for epoch_idx in xrange(params.num_epochs):
            pbar = pb.ProgressBar(widgets=["epoch %d|" % (train_data_set.num_epochs_completed + 1),
                                           pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
            pbar.start()
            for num_batches_completed in xrange(num_batches):
                image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch = train_data_set.get_next_labeled_batch()
                new_size = np.array([batch_size, num_mcs, max_sent_size])
                if mc_sent_batch.shape[2] < max_sent_size:
                    mc_sent_batch = BaseModel._pad(mc_sent_batch, 2, max_sent_size)
                    mc_len_batch = BaseModel._pad(mc_len_batch, 2, max_len_size)
                    mc_label_batch = BaseModel._pad(mc_label_batch, 2, max_label_size)
                _, summary_str, global_step = self.train_batch(sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch, learning_rate)
                writer.add_summary(summary_str, global_step)
                pbar.update(num_batches_completed)
            pbar.finish()
            train_data_set.complete_epoch()

            if val_data_set and (epoch_idx + 1) % params.eval_period == 0:
                print("evaluating on train data ...")
                self.test(sess, train_data_set, num_batches=params.eval_num_batches)
                print("evaluating on val data ...")
                self.test(sess, val_data_set, num_batches=params.eval_num_batches)

            if (epoch_idx + 1) % params.save_period == 0:
                self.save(sess)
        print("training done.")

    @staticmethod
    def _pad(array, dim, new_len):
        p = np.zeros([len(array.shape), 2])
        diff = new_len -array.shape[dim]
        if diff > 0:
            p[dim][1] = diff
            array = np.pad(array, p)
        assert array.shape[dim] == new_len
        return array

    def _pad(self, array, inc):
        assert len(array.shape) > 0, "Array must be at least 1D!"
        if len(array.shape) == 1:
            return np.concatenate([array, np.zeros([inc])], 0)
        else:
            return np.concatenate([array, np.zeros([inc, array.shape[1]])], 0)

    def test(self, sess, test_data_set, num_batches=None):
        num_batches = num_batches if num_batches else test_data_set.num_batches
        num_corrects, total = 0, 0
        string = "N=%d|" % (test_data_set.batch_size * num_batches)
        pbar = pb.ProgressBar(widgets=[string, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
        pbar.start()
        losses = []
        for num_batches_completed in xrange(num_batches):
            image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch = test_data_set.get_next_labeled_batch()
            cur_num_corrects, cur_loss, _, global_step = self.test_batch(sess, image_rep_batch, mc_sent_batch, mc_len_batch, mc_label_batch)
            num_corrects += cur_num_corrects
            total += len(image_rep_batch)
            losses.append(cur_loss)
            pbar.update(num_batches_completed)
        pbar.finish()
        test_data_set.reset()
        loss = np.mean(losses)

        print("at %d: acc = %.2f%% = %d / %d, loss = %.4f" %
              (global_step, 100 * float(num_corrects)/total, num_corrects, total, loss))

    def save(self, sess):
        print("saving model ...")
        save_path = os.path.join(self.save_dir, self.name)
        self.saver.save(sess, save_path, self.global_step)

    def load(self, sess):
        print("loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

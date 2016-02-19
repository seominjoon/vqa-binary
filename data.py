import os

import h5py
import numpy as np
import json


class DataSet(object):
    def __init__(self, batch_size, idxs, image_rep_ds, image_idxs, sent_ds, lens, labels=None, include_leftover=False, name=""):
        """

        :param config:
        :param idxs:
        :param image_rep_dataset: dataset created by h5py.File.create_dataset
        :param sents:
        :param labels: may be unspecified if the dataset is for test-std
        :return:
        """
        self.idxs = idxs
        self.image_rep_ds = image_rep_ds
        self.image_rep_size = image_rep_ds.shape[1]
        self.image_idxs = image_idxs
        self.batch_size = batch_size
        self.sent_ds = sent_ds
        self.lens = lens
        self.num_mcs = sent_ds.shape[1]
        self.max_sent_size = sent_ds.shape[2]
        self.labels = labels
        self.idx_in_epoch = 0
        self.num_epochs_completed = 0
        self.num_examples = len(idxs)
        self.num_batches = self.num_examples / self.batch_size + int(include_leftover)
        self.include_leftover = include_leftover
        self.name = name
        self.reset()

    @staticmethod
    def _pad(array, new_len):
        diff = new_len - array.shape[2]
        if diff > 0:
            p = ((0,0), (0,0), (0,diff))
            array = np.pad(array, p, mode='constant')
        return array

    def get_next_labeled_batch(self, sent_size=None):
        assert self.has_next_batch(), "End of epoch. Call 'complete_epoch()' to rewind."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if self.include_leftover and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        cur_image_idxs = self.image_idxs[cur_idxs]
        image_rep_batch = np.array([self.image_rep_ds[cur_image_idx] for cur_image_idx in cur_image_idxs])
        sent_batch = np.array([self.sent_ds[cur_idx] for cur_idx in cur_idxs])
        len_batch = np.array(self.lens[cur_idxs])
        label_batch = np.array(self.labels[cur_idxs])
        self.idx_in_epoch += self.batch_size
        if sent_size:
            assert sent_size >= self.max_sent_size, "sent size must be bigger than this data's max sent size."
            sent_batch = DataSet._pad(sent_batch, sent_size)
        return image_rep_batch, sent_batch, len_batch, label_batch

    def has_next_batch(self):
        if self.include_leftover:
            return self.idx_in_epoch + 1 < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        np.random.shuffle(self.idxs)


def read_vqa(batch_size, image_rep_h5_path, image_idx_path, sent_h5_path, len_path, labels_path=None, name=""):
    image_rep_h5 = h5py.File(image_rep_h5_path, 'r')
    image_rep_ds = image_rep_h5['data']
    sent_h5 = h5py.File(sent_h5_path, 'r')
    sent_ds = sent_h5['data']
    lens = np.array(json.load(open(len_path, 'rb')))
    image_idxs = np.array(json.load(open(image_idx_path, 'rb')))
    if labels_path:
        labels = np.array(json.load(open(labels_path, 'rb')))
    else:
        labels = None
    idxs = range(len(labels))
    data_set = DataSet(batch_size, idxs, image_rep_ds, image_idxs, sent_ds, lens, labels=labels, name=name)
    return data_set


def read_vqa_from_dir(batch_size, data_dir, name=""):
    image_rep_h5_path = os.path.join(data_dir, 'image_rep.h5')
    image_idx_path = os.path.join(data_dir, 'image_idx.json')
    sent_h5_path = os.path.join(data_dir, 'sent.h5')
    len_path = os.path.join(data_dir, 'len.json')
    label_path = os.path.join(data_dir, 'label.json')
    if not os.path.exists(label_path): label_path = None
    data_set = read_vqa(batch_size, image_rep_h5_path, image_idx_path, sent_h5_path, len_path, label_path, name=name)
    return data_set


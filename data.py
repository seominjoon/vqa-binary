import tensorflow as tf
import h5py
import numpy as np
import json


class DataSet(object):
    def __init__(self, batch_size, idxs, image_rep_ds, image_idxs, sent_ds, lens, labels=None):
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
        self.num_batches = self.num_examples / self.batch_size
        np.random.shuffle(self.idxs)


    def get_next_labeled_batch(self):
        assert self.has_next_batch(), "End of epoch. Call 'complete_epoch()' to rewind."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        cur_idxs = self.idxs[from_:to]
        cur_image_idxs = self.image_idxs[cur_idxs]
        image_rep_batch = np.array([self.image_rep_ds[cur_image_idx] for cur_image_idx in cur_image_idxs])
        sent_batch = np.array([self.sent_ds[cur_idx] for cur_idx in cur_idxs])
        len_batch = self.lens[cur_idxs]
        label_batch = self.labels[cur_idxs]
        self.idx_in_epoch += self.batch_size
        return image_rep_batch, sent_batch, len_batch, label_batch

    def has_next_batch(self):
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.idx_in_epoch = 0
        self.num_epochs_completed += 1
        np.random.shuffle(self.idxs)

def read_vqa(batch_size, image_rep_h5_path, image_idx_path, sent_h5_path, len_path, labels_path=None):
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
    data_set = DataSet(batch_size, idxs, image_rep_ds, image_idxs, sent_ds, lens, labels=labels)
    return data_set


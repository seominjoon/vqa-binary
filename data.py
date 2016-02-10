import tensorflow as tf
import h5py
import numpy as np
import json


class DataSet(object):
    def __init__(self, config, idxs, image_rep_ds, sents, labels=None):
        """

        :param config:
        :param idxs:
        :param image_rep_dataset: dataset created by h5py.File.create_dataset
        :param sents:
        :param labels: may be unspecified if the dataset is for test-std
        :return:
        """
        self.config = config
        self.image_rep_ds = image_rep_ds
        self.idxs = idxs
        self.batch_size = config.batch_size
        self.sents = sents
        self.labels = labels
        self.idx_in_epoch = 0
        self.num_epochs_completed = 0
        self.num_examples = len(idxs)

    def get_next_labeled_batch(self):
        assert self.has_next_batch(), "End of epoch. Call 'complete_epoch()' to rewind."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        cur_idxs = self.idxs[from_:to]
        image_rep_batch = self.image_rep_ds[cur_idxs]
        sent_batch = self.sents[cur_idxs]
        label_batch = self.labels[cur_idxs]
        return image_rep_batch, sent_batch, label_batch

    def has_next_batch(self):
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.idx_in_epoch = 0
        self.num_epochs_completed += 1
        np.random.shuffle(self.idxs)

def read_vqa(image_rep_h5_path, question_json_path, annotation_json_path=None, data_dir='data'):
    question_dict = json.load(open(question_json_path, 'rb'))
    annotation_dict = json.load(open(annotation_json_path, 'rb'))
    image_rep_h5 = h5py.File(image_rep_h5_path, 'r')
    image_rep_ds = image_rep_h5['data']

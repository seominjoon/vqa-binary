import tensorflow as tf
import progressbar as pb

from data import read_vqa

flags = tf.app.flags

flags.DEFINE_integer("train_batch_size", 32, "Batch size during training [32]")
flags.DEFINE_string("train_image_rep_h5", "train/image_rep.h5", "image_rep.h5 file path for training [train/image_rep.h5]")
flags.DEFINE_string("train_image_idx", "train/image_idx.json", "image_idx.json file path for training [train/image_idx.json]")
flags.DEFINE_string("train_sent_h5", "train/sent.h5", "sent.h5 file path for training [train/sent.h5]")
flags.DEFINE_string("train_label", "train/label.json", "label.json file path for training [train/label.json]")

FLAGS = flags.FLAGS

def main(_):
    data_set = read_vqa(FLAGS.train_batch_size, FLAGS.train_image_rep_h5, FLAGS.train_image_idx, FLAGS.train_sent_h5, FLAGS.train_label)

    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.Timer()], maxval=data_set.num_batches).start()
    i = 1
    while data_set.has_next_batch():
        image_rep_batch, sent_batch, label_batch = data_set.get_next_labeled_batch()
        pbar.update(i)
        i += 1
    pbar.finish()
    print sent_batch.shape


if __name__ == "__main__":
    tf.app.run()

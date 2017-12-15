#!/usr/bin/env python
# Model specification and training, validation pipeline to train PilotNet, an
# Nvidia designed model for predicting steering angles from a single front
# facing camera on a car in motion.

import os
import tensorflow as tf
import numpy as np
from config import config


# custom library
# from viewer import *

# TODO:
# 1. include add_to_collections if we want to use model for inference.
# 2. Implement def evaluation() to evaluate on test inputs.
# 3. Implement visualization portion.
# 4. Convert image from RGB to YSV


class PilotNet(object):

    def __init__(self, sess, log_dir, ckpt_dir, n_epochs, batch_size):
        self._sess = sess
        self._log_dir = log_dir
        self._ckpt_dir = ckpt_dir
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._build_graph()

    def _model(self, x):
        """ Model specification of PilotNet. """
        assert(x[0].shape == (480, 640, 3))
        out = tf.layers.batch_normalization(x)
        out = tf.layers.conv2d(x, 24, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 36, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 48, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = tf.reshape(out, [-1, 64 * 53 * 73])
        out = tf.layers.dense(out, 100, tf.nn.relu)
        out = tf.layers.dense(out, 50, tf.nn.relu)
        out = tf.layers.dense(out, 10, tf.nn.relu)
        out = tf.layers.dense(out, 1)
        return out

    def _build_graph(self):
        """ Build graph and define placeholders and variables. """
        self._inputs = tf.placeholder("float", [None, 480, 640, 3])
        self._targets = tf.placeholder("float", [None, 1])
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self._predict = self._model(self._inputs)
        self._loss = tf.losses.mean_squared_error(labels=self._targets, predictions=self._predict)
        self._train = tf.train.AdamOptimizer().minimize(self._loss, global_step=self._global_step)

    def _train_admin_setup(self):
        """ Setup writers, savers, and summary ops. """
        # Writers
        self._train_writer = tf.summary.FileWriter(self._log_dir + '/train', self._sess.graph)

        # Saver
        self._saver = tf.train.Saver()

        # Summaries
        tf.summary.scalar("loss", self._loss)
        self._all_summaries = tf.summary.merge_all()

    def _prepare_data(self, file_path, train=True):
        """ Create data iterator """
        dataset = input_fn(file_path)
        batch_size = 1 if train else self._batch_size
        dataset = dataset.batch(batch_size)
        print("Data loaded: {}".format(file_path))
        batch_generator = dataset.make_initializable_iterator()
        next_element = batch_generator.get_next()
        return next_element, batch_generator

    def _generate_collections(self):
        """ Specificy collections so model can be reloaded later. """
        tf.add_collections("inputs", self._inputs)
        tf.add_collections("predict", self._predict)

    def train(self, train_file_path, valid_file_path):
        """ Train and validate. """
        self._train_admin_setup()
        tf.global_variables_initializer().run()

        # Check if there is a previously saved checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._ckpt_dir))
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt))
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

        # Prepare train and valididation data
        train_next, train_gen = self._prepare_data(train_file_path)
        valid_next, valid_gen = self._prepare_data(valid_file_path)
        best_valid_loss = None

        for epoch in range(self._n_epochs):
            self._sess.run(train_gen.initializer)
            epoch_loss = []
            while True:
                try:
                    img_batch, label_batch = self._sess.run(train_next)
                    loss, _ = self._sess.run([self._loss, self._train],
                                             feed_dict={
                        self._inputs: img_batch['image'],
                        self._targets: label_batch}
                    )
                    epoch_loss.append(loss)
                except tf.errors.OutOfRangeError:
                    break

            # Validation after each epoch
            self._sess.run(valid_gen.initializer)
            valid_loss = []
            while True:
                try:
                    img_batch, label_batch = self._sess.run(valid_next)
                    loss = self._sess.run([self._loss],
                                          feed_dict={
                        self._inputs: img_batch['image'],
                        self._targets: label_batch}
                    )
                    valid_loss.append(loss)
                except tf.errors.OutOfRangeError:
                    break
            # Add summary and save checkpoint after every epoch
            s = self._sess.run(self._all_summaries, feed_dict={
                self._inputs: img_batch['image'],
                self._targets: label_batch}
            )
            self._train_writer.add_summary(s, global_step=epoch)
            valid_loss = np.mean(valid_loss)
            print("Epoch: {} Train Loss: {} Valid Loss: {}".format(
                epoch, np.mean(epoch_loss), valid_loss)
            )

            if best_valid_loss == None:
                best_valid_loss = valid_loss
            elif valid_loss < best_valid_loss:
                self._saver.save(self._sess, self._ckpt_dir, global_step=self._global_step)
                best_valid_loss = valid_loss

        # Need to closer writers
        self._train_writer.close()

    def predict(self, file_path):
        # This is customized for this particular pipeline. Predict takes in path
        # to csv file with img path information, predicts, and dumps steering
        # prediction, ground_truth, and img to pickle file.
        import pickle
        next_element, generator = self._prepare_data(file_path, train=False)
        self._sess.run(generator.initializer)

        images = []
        steer_labels = []
        steer_preds = []
        while True:
            try:
                img, steer_label = self._sess.run(next_element)
                steer_pred = self._sess.run([self._predict],
                                feed_dict={self._inputs: img['image']}
                )
                images.append(img)
                steer_labels.append(steer_label)
                steer_preds.append(steer_pred)
            except tf.errors.OutOfRangeError:
                break

        data = {
            "image": images,
            "steer_label": steer_labels,
            "steer_pred": steer_preds
        }
        with open("predictions.pickle", 'w') as f:
            pickle.dump(data, f)
            print("Predictions pickled...")


def input_fn(file_path):
    """ Generic input function used for creating dataset iterator """
    # Customized for the Udacity train data processed by scripts to convert from
    # rosbags to csv files.
    def _decode_csv(line):
        # tf.decode_csv needs a record_default as 2nd parameter
        data = tf.decode_csv(line, list(np.array([""] * 12).reshape(12, 1)))[-8:-5]
        img_path = home + data[1]
        img_decoded = tf.to_float(tf.image.decode_image(tf.read_file(img_path)))

        # normalize between (-1,1)
        img_decoded = -1.0 + 2.0 * img_decoded / 255.0
        steer_angle = tf.string_to_number(data[2], tf.float32)

        # return camera as well to confirm location of camera
        # e.g left_camera, center_camera, right_camera...we want center_camera
        return {'image': img_decoded, 'camera': data[0]}, [steer_angle]

    # skip() header row,
    # filter() for center camera,
    # map() transform each line by applying decode_csv()
    dataset = (tf.contrib.data.TextLineDataset(file_path)
               .skip(1)
               .filter(lambda x: tf.equal(
                       tf.string_split(tf.reshape(x, (1,)), ',').values[4],
                       'center_camera'))
               .map(_decode_csv))
    return dataset


if __name__ == "__main__":
    home = config['bag4']
    train_file_path = home + "train_interpolated.csv"
    valid_file_path = home + "valid_interpolated.csv"

    tf.reset_default_graph()

    # Want to see what devices are being used.
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        model = PilotNet(sess, 'pilot_net', 'checkpoints/pilot_net', 1, 32)
        model.train(train_file_path, valid_file_path)
        model.predict(valid_file_path)

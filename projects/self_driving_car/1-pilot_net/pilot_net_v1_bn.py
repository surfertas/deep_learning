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
from data_pipeline import *

# https://arxiv.org/pdf/1704.07911.pdf
# https://arxiv.org/pdf/1708.03798.pdf
# data: https://github.com/udacity/self-driving-car/tree/master/datasets/CH2
# scripts to convert rosbags to csv:
# https://github.com/rwightman/udacity-driving-reader

# TODO:
# 3. Implement visualization portion.
# 4. Convert image from RGB to YSV


class PilotNet(object):

    def __init__(self, sess, log_dir, ckpt_dir, input_dim, n_epochs, batch_size):
        self._sess = sess
        self._log_dir = log_dir
        self._ckpt_dir = ckpt_dir
        self._img_h, self._img_w, self._img_c = input_dim
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._build_graph()

    def _model(self, x):
        """ Model specification of PilotNet. """
        def conv2d_bn(x, kernels, f, s, p, activation):
            x = tf.layers.conv2d(x, kernels, f, s, p, activation=activation)
            x = tf.layers.batch_normalization(x)
            return x

        assert(x[0].shape == (self._img_h, self._img_w, self._img_c))
        # normalize between (-1,1)
        out = tf.subtract(x, 0.5)
        out = tf.multiply(out, 2.0)

        # out = tf.layers.conv2d(out, 24, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = conv2d_bn(out, 24, [5, 5], (2, 2), "valid", activation=tf.nn.relu)

        # out = tf.layers.conv2d(out, 36, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = conv2d_bn(out, 36, [5, 5], (2, 2), "valid", activation=tf.nn.relu)

        # out = tf.layers.conv2d(out, 48, [5, 5], (2, 2), "valid", activation=tf.nn.relu)
        out = conv2d_bn(out, 48, [5, 5], (2, 2), "valid", activation=tf.nn.relu)

        # out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = conv2d_bn(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)

        # out = tf.layers.conv2d(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)
        out = conv2d_bn(out, 64, [3, 3], (1, 1), "valid", activation=tf.nn.relu)

        out = tf.reshape(out, [-1, 64 * 18 * 73])
        out = tf.layers.dense(out, 100, tf.nn.relu)
        out = tf.layers.dense(out, 50, tf.nn.relu)
        out = tf.layers.dense(out, 10, tf.nn.relu)
        out = tf.layers.dense(out, 1)
        return out

    def _build_graph(self):
        """ Build graph and define placeholders and variables. """
        self._inputs = tf.placeholder("float", [None, self._img_h, self._img_w, self._img_c])
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

    def _generate_collections(self):
        """ Specificy collections so model can be reloaded later. """
        tf.add_collections("inputs", self._inputs)
        tf.add_collections("predict", self._predict)

    def train(self, train_iterator, valid_iterator):
        """ Train and validate. """
        self._train_admin_setup()
        tf.global_variables_initializer().run()

        # Check if there is a previously saved checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._ckpt_dir))
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt))
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

        # Prepare train and valididation data
        train_next, train_iter = train_iterator
        valid_next, valid_iter = valid_iterator
        best_valid_loss = None

        for epoch in range(self._n_epochs):
            self._sess.run(train_iter.initializer)
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

            # Validation every other epoch
            self._sess.run(valid_iter.initializer)
            valid_loss = []
            while True:
                try:
                    img_batch, label_batch = self._sess.run(valid_next)
                    loss, step = self._sess.run([self._loss, self._global_step],
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
            print("Epoch: {} Step: {} Train Loss: {} Valid Loss: {}".format(
                epoch, step, np.sum(epoch_loss), np.sum(valid_loss))
            )

            if best_valid_loss == None:
                best_valid_loss = valid_loss
            elif valid_loss < best_valid_loss:
                self._saver.save(self._sess, self._ckpt_dir, global_step=self._global_step)
                best_valid_loss = valid_loss

        # Need to closer writers
        self._train_writer.close()

    def predict(self, data_iterator):
        # This is customized for this particular pipeline. Predict takes in path
        # to csv file with img path information, predicts, and dumps steering
        # prediction, ground_truth, and img to pickle file.
        import pickle
        next_element, iterator = data_iterator
        self._sess.run(iterator.initializer)

        images = []
        steer_labels = []
        steer_preds = []
        while True:
            try:
                img, steer_label = self._sess.run(next_element)
                steer_pred = self._sess.run([self._predict],
                                            feed_dict={
                                            # TODO: why do we need [0] here,
                                            # when batch size = 1 works fine with validation.
                                                self._inputs: img['image'][0]}
                                            )
                # Store image path as raw image too large.
                images.append(img['image_path'])
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

if __name__ == "__main__":

    tf.reset_default_graph()

    # Want to see what devices are being used.
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        model = PilotNet(
            sess,
            'pilot_net',
            'checkpoints/pilot_net',
            input_dim=(200, 640, 3),
            n_epochs=15,
            batch_size=128
        )

        train_data = DataHandler(
            data_dir=config['data_dir'],
            file_name='train_interpolated.csv',
            bags=config['bags'],
            batch_size=128,
            train=True,
            augment=True
        )

        valid_data = DataHandler(
            data_dir=config['data_dir'],
            file_name='valid_interpolated.csv',
            bags=config['bags'],
            batch_size=1,
            train=False,
            augment=False
        )

        model.train(train_data.get_iterator(), valid_data.get_iterator())
        # TODO: upload test data so we can run test on test data instead of
        # validation data.
        model.predict(valid_data.get_iterator())

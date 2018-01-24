#!/usr/bin/env python
# @author Tasuku Miura
# @brief Classes and methods related to data pipeline

import os
import tensorflow as tf
import numpy as np
from config import config


class DataHandler(object):

    def __init__(self, data_dir, file_name, bags, batch_size, train, augment):
        self._data_dir = data_dir
        self._bags = bags
        self._file_name = file_name
        self._file_paths = self._gen_file_paths()

        self._temp_home_dir = None
        self._data_set = None

        self._train = train
        self._augment = augment
        self._batch_size = batch_size

        self._generate_data_set(self._train)

    def _gen_file_paths(self):
        file_paths = [
            os.path.join(self._data_dir, bag, self._file_name)
            for bag in self._bags
        ]
        return file_paths

    def _generate_data_set(self, train):
        def __decode_csv(line):
            # tf.decode_csv needs a record_default as 2nd parameter
            data = tf.decode_csv(line, list(np.array([""] * 12).reshape(12, 1)))[-8:-5]
            img_path = self._temp_home_dir + '/' + data[1]
            img_decoded = tf.to_float(tf.image.decode_image(tf.read_file(img_path)))

            # pre-process, normalize to 0 - 1 range, and resize image.
            x = img_decoded / 255.0
            x.set_shape([480, 640, 3])
            x = tf.image.resize_image_with_crop_or_pad(
                x,
                200,  # Height
                640,  # Width
            )

            steer_angle = tf.string_to_number(data[2], tf.float32)
            return {'image': x, 'image_path': data[1]}, [steer_angle]

        def __random_augmentation(image, target):
            # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
            x = image['image']
            x = tf.image.random_brightness(x, max_delta=32. / 255.)
            x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
            x = tf.image.random_hue(x, max_delta=0.2)
            x = tf.image.random_contrast(x, lower=0.5, upper=1.5)
            x = tf.clip_by_value(x, 0.0, 1.0)

            flip = np.random.randint(2)
            if flip == 1:
                x = tf.image.flip_left_right(x)
                target *= -1

            # return image and image path
            return {'image': x, 'image_path': image['image_path']}, target

        for file_path in self._file_paths:
            # For each path set _temp_home_dir to be used in call to _decode_csv.
            # e.g. ../1-pilot_net/data/bag4
            self._temp_home_dir = file_path.rsplit('/', 1)[0]
            dataset = (tf.contrib.data.TextLineDataset(file_path)
                       .skip(1)
                       .filter(lambda x: tf.equal(
                               tf.string_split(tf.reshape(x, (1,)), ',').values[4],
                               'center_camera'))
                       .map(__decode_csv, num_parallel_calls=4))

            if self._data_set == None:
                self._data_set = dataset
                print("Init dataset")
            else:
                self._data_set = self._data_set.concatenate(dataset)
                print("Concat dataset {}".format(self._data_set))

        # If augment is true, than augment data and concat to original dataset.
        if self._train and self._augment:
            assert(self._data_set != None)
            augmented = self._data_set.map(__random_augmentation, num_parallel_calls=4)
            self._data_set = self._data_set.concatenate(augmented)
            print("Augmentation completed...")

    def get_iterator(self):
        """ Create data iterator """
        assert(self._data_set != None)
        batch_size = 1 if not self._train else self._batch_size
        self._data_set = self._data_set.batch(batch_size)
        batch_iterator = self._data_set.make_initializable_iterator()
        next_element = batch_iterator.get_next()
        return next_element, batch_iterator

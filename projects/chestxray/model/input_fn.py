"""Create the input data pipeline using `tf.data`"""
import os
import io
import tensorflow as tf
import numpy as np

from  PIL import Image

def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label


def train_augment(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Since values have been standardized clip between -1.0, 1.0
    image = tf.clip_by_value(image, -1.0, 1.0)

    return image, label



def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs



def input_tf_records_all_fn(tf_record_dir, params):

    def _parse_record(string_record):
        feature = {
            'label': tf.FixedLenFeature([14, ], tf.int64),
            'shape': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        }

        features = tf.parse_single_example(string_record, features=feature, name='features')
        image = tf.cast(tf.decode_raw(features['image'], tf.uint8), tf.float32)
        shape = tf.decode_raw(features['shape'], tf.int32)

        image = tf.reshape(image, shape)
        # TODO: remove need to hard code dimensions
        image.set_shape([params.image_size, params.image_size, params.num_channels])

        # Standardize image
        # https://stackoverflow.com/questions/43953531/do-i-need-to-subtract-rgb-mean-value-of-imagenet-when-finetune-resnet-and-incept
        image = tf.image.per_image_standardization(image)
        label = features['label']
        return image, label


    def _get_iterator(dataset):
        # Create reinitializable iterator from dataset
        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        iterator_init_op = iterator.initializer
        inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
        return inputs

    train_augment_fn = lambda f, l: train_augment(f, l, params.use_random_flip)

    

    print([os.path.join(tf_record_dir,record) for record in os.listdir(tf_record_dir)])
    dataset = tf.data.TFRecordDataset([os.path.join(tf_record_dir,record) for record in os.listdir(tf_record_dir)])
    
    dataset = dataset.map(_parse_record,  num_parallel_calls=params.num_parallel_calls)
    # Really want to use buffer__size of full data set for proper shuffling but running into memory issues.
    dataset = dataset.shuffle(40000)

    # Split dataset into train and eval
    train_dataset = dataset.take(params.train_size)
    eval_dataset = dataset.skip(params.train_size)
    
    # Create datasets
    #augmented_dataset = train_dataset.map(train_augment_fn)
    #train_dataset = train_dataset.concatenate(augmented_dataset)
    train_dataset = (train_dataset
           .shuffle(buffer_size=30000)
           .batch(params.batch_size)
           .prefetch(1))  # make sure you always have one batch ready to serve
    
    eval_dataset = (eval_dataset
           .batch(params.batch_size)
           .prefetch(1))  # make sure you always have one batch ready to serve


    train_inputs = _get_iterator(train_dataset)
    eval_inputs = _get_iterator(eval_dataset)

    return train_inputs, eval_inputs

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from io import BytesIO, StringIO
from scipy.misc import imsave
from skimage.transform import resize
from google.cloud import storage

from model.utils import Params

# https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/2017/examples/09_tfrecord_example.py

#TODO: make it say user can specify location to dump tfrecord
#TODO: use os.makedir() to create the directory to house the trecord

# usage: python build_tfrecord_dataset.py --model_dir ./experiments/base_model/

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/gcs',
        help="Directory with data_with_gcs.csv")
parser.add_argument('--model-dir', default='./experiments/base_model',
        help="Directory where the params.json file found.")
parser.add_argument('--gcs-bucket-name', default='test-track',
        help="Directory where the params.json file found.")


def _float_feature(values):
    if not isinstance(values, (tuple, list, np.ndarray)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def images_to_tf_record(results, tfrecord_data_dir, file_name="data", batch_number=0):
    if results is None:
        results = []

    writer = tf.python_io.TFRecordWriter(
            os.path.join(tfrecord_data_dir, file_name + '_' + str(batch_number) + ".tfrecord"))

    for i in range(len(results)):
        image = np.asarray(results[i][0],np.uint8).tobytes()
        shape = np.asarray(results[i][0].shape, np.int32).tobytes()
        target = results[i][1]
        # TODO: Think about making it work with PNG
        # https://stackoverflow.com/questions/39460436/how-to-create-tfrecords-file-from-png-images
        example = tf.train.Example(features=tf.train.Features(feature={
            'target': _float_feature(target),
            'shape': _bytes_feature(shape),
            'image': _bytes_feature(image)
        }))
        writer.write(example.SerializeToString())

    writer.close()


def generate_sample(csv_path, max_count=None):
    """ Generates sample size of max_count examples. """
    df = pd.read_csv(csv_path)

    columns = ['cloud_url', 'throttle','steer']

    df_sample = df[columns][:max_count] if max_count else df[columns]
    df_sample['target'] = df_sample[columns[-2:]].values.tolist()
    df_sample = df_sample.drop(columns=['throttle','steer'])
    return df_sample

def process_item(item, image_shape, bucket_name):
    def _get_image(gs_path):
        gs_path = gs_path.split("/")    
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob("/".join(gs_path[-2:]))
        image_string = blob.download_as_string()
        image = Image.open(BytesIO(image_string))
        image = np.asarray(image)
        return image

    image = _get_image(item[0])
    image = resize(image, image_shape, anti_aliasing=0.15, mode='constant', preserve_range=True)
    return [image, item[1]]



def generate_processed_image_label_pair(gcs_bucket_name, filename, tfrecord_base, tfrecord_data_dir, image_shape, batch_size=2000, max_images=None):
    df = generate_sample(filename, max_images).values
    n_images = len(df)
    n_batches = (len(df) / batch_size) + 1

    for i in tqdm(range(int(n_batches))):
        values = [delayed(process_item)(item, image_shape, gcs_bucket_name) for item in
                df[i * batch_size:min(n_images, batch_size * (i + 1))]]
        with ProgressBar():
            results = compute(*values, scheduler='processes')

        images_to_tf_record(results, tfrecord_data_dir, file_name=tfrecord_base, batch_number=i)



if __name__ == '__main__':
    args = parser.parse_args()
    data_filename = os.path.join(args.data_dir, 'data_with_gcs.csv')
    assert os.path.isfile(data_filename), "no data_with_gcs.csv  found at {}".format(data_filename)

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "no json configuration file  found at {}".format(json_path)
    params = Params(json_path)

    tfrecord_data_dir =os.path.join('./data', params.experiment_name)
    if not os.path.exists(tfrecord_data_dir):
        os.makedirs(tfrecord_data_dir)

    # make sure that the shape is defined in params.jason correctly
    shape = (params.image_size, params.image_size, params.num_channels)

    print("Experiment name: {}".format(params.experiment_name))
    print("TFrecord data dir:: {}".format(tfrecord_data_dir))
    print("Total dataset size: {}".format(params.dataset_size))
    print("Image shape: {}".format(shape))

    # By setting max_images to dataset_size specified in params, we can ensure consistency between
    # the data generation process and train process
    generate_processed_image_label_pair(
            gcs_bucket_name=args.gcs_bucket_name,
            filename=data_filename,
            tfrecord_base=params.experiment_name,
            tfrecord_data_dir=tfrecord_data_dir,
            image_shape=shape,
            batch_size=500,
            max_images=params.dataset_size)


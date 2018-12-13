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

# usage: python build_tfrecord_dataset.py --model_dir ./experiments/xray_model_0001/

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../../data',
        help="Directory with Data_Entry_2017_with_gcs.csv")
parser.add_argument('--model_dir', default='./experiments/base_model',
        help="Directory where the params.json file found.")


def _int64_feature(values):
    if not isinstance(values, (tuple, list, np.ndarray)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def images_to_tf_record(images, tfrecord_data_dir, file_name="data", batch_number=0):
    if images is None:
        images = []

    writer = tf.python_io.TFRecordWriter(
            os.path.join(tfrecord_data_dir, file_name + '_' + str(batch_number) + ".tfrecord"))

    for i in range(len(images)):
        image = np.asarray(images[i][0],np.uint8).tobytes()
        shape = np.asarray(images[i][0].shape, np.int32).tobytes()
        label = images[i][1]
        # TODO: Think about making it work with PNG
        # https://stackoverflow.com/questions/39460436/how-to-create-tfrecords-file-from-png-images
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'shape': _bytes_feature(shape),
            'image': _bytes_feature(image)
        }))
        writer.write(example.SerializeToString())

    writer.close()


def generate_labels(csv_path, max_count=None):
    """ Converts raw labels to multi-class labels used for training """
    def _create_labels(row):
        pathology_list = [
                'Cardiomegaly',
                'Emphysema',
                'Effusion',
                'Hernia',
                'Nodule',
                'Pneumothorax',
                'Atelectasis',
                'Pleural_Thickening',
                'Mass',
                'Edema',
                'Consolidation',
                'Infiltration',
                'Fibrosis',
                'Pneumonia']


        n_pathology = len(pathology_list)
        pathology_index = dict(zip(pathology_list, list(range(n_pathology))))

        zeros = np.zeros(n_pathology)
        if "No Finding" not in row:
            i = list(pd.Series(row).map(pathology_index))
            zeros[i] = 1.0

        return zeros.astype('int')

    df = pd.read_csv(csv_path)

    if max_count:
        df_labels = df[['CloudURl', 'Finding Labels']][:max_count]
    else:
        df_labels = df[['CloudURl', 'Finding Labels']]

    df_labels['labels'] = (df_labels['Finding Labels'].str.split("|").apply(_create_labels))

    return df_labels

def process_item(item, image_shape):
    def _get_image(gs_path):
        gs_path = gs_path.split("/")    
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('xray8-images')
        blob = bucket.blob("/".join(gs_path[-3:]))
        image_string = blob.download_as_string()
        image = Image.open(BytesIO(image_string))
        image = np.asarray(image)
        return image

    image = _get_image(item[0])
    image = resize(image, image_shape, anti_aliasing=0.15, mode='constant', preserve_range=True)
    return [image, item[2]]



def generate_processed_image_label_pair(filename, tfrecord_base, tfrecord_data_dir, image_shape, batch_size=2000, max_images=None):
    df_labels = generate_labels(filename, max_images).values
    n_images = len(df_labels)
    n_batches = (len(df_labels) / batch_size) + 1

    for i in tqdm(range(int(n_batches))):
        values = [delayed(process_item)(item, image_shape) for item in
                df_labels[i * batch_size:min(n_images, batch_size * (i + 1))]]
        with ProgressBar():
            results = compute(*values, scheduler='processes')

        images_to_tf_record(results, tfrecord_data_dir, file_name=tfrecord_base, batch_number=i)



if __name__ == '__main__':
    args = parser.parse_args()
    data_filename = os.path.join(args.data_dir, 'Data_Entry_2017_with_gcs.csv')
    assert os.path.isfile(data_filename), "no data_entry_2017_with_gcs.csv  found at {}".format(data_filename)

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
            data_filename,
            tfrecord_base=params.experiment_name,
            tfrecord_data_dir=tfrecord_data_dir,
            image_shape=shape,
            batch_size=15000,
            max_images=params.dataset_size)


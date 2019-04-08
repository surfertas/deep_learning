# author@ Tasuku Miura
# date@ 12.30.2018

# Script to generate csv with url to data stored on Google Cloud Storage.
# The gcs.csv file is generated using the following on the command line: 
# $ gsutil ls [bucket_name]/[image_dir] >> gcs.csv


import os
import argparse

import pandas as pd
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/gcs',
        help="Directory with gcs.csv")
parser.add_argument('--mount-dir', default=None,
        help="Directory where Google storage was fused.")
parser.add_argument('--gcs-filename', default='gcs.csv',
        help="Name of file with gcs urls")
parser.add_argument('--percentage_to_use', default=1.0,
        help="Percentage of dataset to use")
parser.add_argument('--bucket-name', default='track-v0-night',
        help="Name of GCS bucket")


def generate_csv_with_gcs(data_dir, mount_dir, gcs_filename, bucket_name, percentage_to_use):
    """ Generate a csv file with gcs url to images stored on the cloud and
        associated targets.
        Args:
            data_dir - str: Directory with raw gcs url csv
            gcs_filename - str: file name of the csv file with gcs urls
            bucket_name - bucket name on GCS that holds the data
    """
    gcs_filename = os.path.join(data_dir, gcs_filename)    
    df = pd.read_csv(gcs_filename, header=None)
    
    # Path to pickle file should be last in file in dataframe
    pickle_file_path = df.tail(1).iloc[0].values[0]
    assert pickle_file_path.split('.')[-1] == 'pickle'

    df_image_url = df[:-1]

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
   
    blob = bucket.blob("/".join(pickle_file_path.split("/")[-2:]))    
    blob.download_to_filename("predictions.pickle")
    
    df = pd.read_pickle("predictions.pickle")

    df_images = pd.DataFrame(df['images'])
    df_targets = pd.DataFrame(df['control_commands'])

    df_sample = pd.concat([df_images,df_targets], axis=1)
    df_sample.columns = ['image', 'throttle', 'steer']

    # Use only n samples specified by "percentage_to_use". Use smaller size when experimenting.
    n_samples = len(df_sample)
    n_use = int(n_samples * percentage_to_use)
    df_sample = df_sample[:n_use]

    df_sample['key'] = df_sample['image'].apply(lambda x: x.decode("utf-8").split('/')[-1])
    df_sample = df_sample.set_index('key').drop(columns=['image'])
    
    df_gsurl = pd.read_csv(gcs_filename, header=None)
    
    # Set the directory to the directory to the google storage was fused.
    if mount_dir is not None:
        df_gsurl = df_gsurl[0].apply(lambda x: os.path.join(args.mount_dir, x.split('/')[-1])).to_frame()
    
    df_gsurl['key'] = df_gsurl[0].apply(lambda x: x.split('/')[-1])
    df_gsurl = df_gsurl.set_index('key')

    # Join on image key. Use inner join as the number of examples collected wont
    # necessarily match the examples pickled as a result of the frequency at
    # which examples are stored.
    df_final = df_gsurl.join(df_sample, how='inner')
    df_final = df_final.rename(columns={0:'url'})
    
    print("Saving data_with_gcs.csv")
    df_final.to_csv(os.path.join(data_dir,'data_with_gcs.csv'))


if __name__=="__main__":
    args = parser.parse_args()

    generate_csv_with_gcs(args.data_dir, args.mount_dir, args.gcs_filename, args.bucket_name, args.percentage_to_use)
   

# author@ Tasuku Miura
# date@ 4.21.2019

# Script to generate csv with url to data stored locally.

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/gcs',
        help="Directory with gcs.csv")
parser.add_argument('--gcs-filename', default='gcs.csv',
        help="Name of file with gcs urls")
parser.add_argument('--percentage_to_use', default=1.0,
        help="Percentage of dataset to use")

def generate_csv_with_url(data_dir, gcs_filename, percentage_to_use):
    """ Generate a csv file with gcs url to images stored locally and
        associated targets.
        Args:
            data_dir - str: Directory with raw url csv
            gcs_filename - str: file name of the csv file with gcs urls
    """
    gcs_filename = os.path.join(data_dir, gcs_filename)    
    df = pd.read_csv(gcs_filename, header=None)
    
    # Path to pickle file should be last in file in dataframe
    pickle_file_path = df.tail(1).iloc[0].values[0]
    assert pickle_file_path.split('.')[-1] == 'pickle'

    df_images = df[:-1]
    df_images.columns = ['key']
    path = '/home/tasuku/data/track-v0-night/1553947163510472059'
    df_images['url'] = df_images['key'].apply(lambda x: os.path.join(path, x))
    df_images_path = df_images.set_index('key')

    df = pd.read_pickle(os.path.join(path, "predictions.pickle"))
    df_images = pd.DataFrame(df['images'])
    df_targets = pd.DataFrame(df['control_commands'])

    df_sample = pd.concat([df_images, df_targets], axis=1)
    df_sample.columns = ['image', 'throttle', 'steer']

    # Use only n samples specified by "percentage_to_use". Use smaller size when experimenting.
    n_samples = len(df_sample)
    n_use = int(n_samples * percentage_to_use)
    df_sample = df_sample[:n_use]

    df_sample['key'] = df_sample['image'].apply(lambda x: x.decode("utf-8").split('/')[-1])
    df_sample = df_sample.set_index('key').drop(columns=['image'])
    
    # Join on image key. Use inner join as the number of examples collected wont
    # necessarily match the examples pickled as a result of the frequency at
    # which examples are stored.
    df_final = df_images_path[:n_use].join(df_sample, how='inner')
    
    print("Saving path_to_data.csv")
    df_final.to_csv(os.path.join(data_dir,'path_to_data.csv'))


if __name__=="__main__":
    args = parser.parse_args()

    generate_csv_with_url(args.data_dir, args.gcs_filename, args.percentage_to_use)
   

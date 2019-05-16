# author@ Tasuku Miura
# date@ 4.21.2019

# Script to generate csv with paths to data stored locally.

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/gcs',
        help="Directory with gcs.csv")
parser.add_argument('--filename', default='gcs3926.csv',
        help="Name of csv file")
parser.add_argument('--image-dir', default='/home/tasuku/data/track-v0-night/1557924016413533926',
        help="Directory with gcs.csv")
parser.add_argument('--percentage_to_use', default=1.0,
        help="Percentage of dataset to use")

def generate_csv_with_paths(data_dir: str, filename: str, image_dir: str, percentage_to_use: float):
    """ Generate a csv file with path to images stored locally and
        associated targets.
        Args:
            data_dir: directory with raw csv
            filename: file name of the csv file
            percentage_to_use: percentage of the samples to use
    """
    filename = os.path.join(data_dir, filename)    
    df = pd.read_csv(filename, header=None)
    
    # Path to pickle file should be last in file in dataframe
    pickle_file_path = df.tail(1).iloc[0].values[0]
    assert pickle_file_path.split('.')[-1] == 'pickle'

    df_images = df[:-1]
    df_images.columns = ['key']
    df_images['url'] = df_images['key'].apply(lambda x: os.path.join(image_dir, x))
    df_images_path = df_images.set_index('key')

    df = pd.read_pickle(os.path.join(image_dir, "predictions.pickle"))
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
    generate_csv_with_paths(args.data_dir, args.filename, args.image_dir, args.percentage_to_use)
   

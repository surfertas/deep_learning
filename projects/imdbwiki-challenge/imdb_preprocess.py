#!/usr/bin/env python
# Script to reformat the IMDB data from matlab to numpy arrays.
# Flattens all images to gray scale and resizes to (128,128)
# Reference: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
# usage:
# ./imdb_preprocess --n_samples 1000
# (Will return <1000 samples from the training set as a result of filtering.
#
# NOTE: If --n_samples is set to 0 it will attempt to process the entire data set.

import os
import argparse
import pickle
import numpy as np
import scipy.io as sio
import scipy.misc as spm
import datetime
import matplotlib.image as plt


def reformat_date(mat_date):
    """ Extract only the year.

        Necessary for calculating the age of the individual in the image.

    Args:
        mat_date - raw date format.
    Retrurns:
        dt - adjusted date.
    """
    # Take account for difference in convention between matlab and python.
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt


def matlab_to_numpy(path_to_meta, matlab_file, path_to_images):
    """ Opens .mat file and reformats.

        Matlab struct format to dictionary of numpy arrays.

    Args:
        path_to_meta - path to dir with matlab meta file.
        matlab_file - matlab file
        path_to_images - incomplete paths to images.
    Returns:
        imdb_dict - dict of numpy arrays.
    """
    mat_struct = sio.loadmat(os.path.join(path_to_meta, matlab_file))
    data_set = [data[0] for data in mat_struct['imdb'][0, 0]]

    keys = [
        'dob',
        'photo_taken',
        'full_path',
        'gender',
        'name',
        'face_location',
        'face_score',
        'second_face_score',
        'celeb_names',
        'celeb_id'
    ]

    # Creates path to full path to image from incomplete path.
    create_path = lambda path: os.path.join(path_to_images, path[0])

    imdb_dict = dict(zip(keys, np.asarray(data_set)))
    imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
    imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

    # Add 'age' key to the dictionary
    imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

    return imdb_dict


def create_and_dump(get_paths, path_to_images, imdb_dict, n_samples):
    """ Creates dictionary of inputs and labels and pickles.
    Args:
        path_to_images - image dir.
        imdb_dict - numpy dict
        n_samples - number of samples.
    """
    raw_path = np.asarray(imdb_dict['full_path'])
    raw_age = np.asarray(imdb_dict['age'])
    raw_sface = np.asarray(imdb_dict['second_face_score'])

    if n_samples != 0:
        # Sample without replacement.
        isamples = np.random.choice(
            np.arange(len(imdb_dict['full_path'])),
            n_samples,
            replace=False,
        )

        print(isamples)
        # Extract the samples.
        raw_path = raw_path[isamples]
        raw_age = raw_age[isamples]
        raw_sface = raw_sface[isamples]

    age = []
    img_paths = []
    # Filter out unusable data.
    for i, sface in enumerate(raw_sface):
        in_valid_age_range = (raw_age[i] >= 0 and raw_age[i] <= 100)
        if not np.isnan(sface) and in_valid_age_range:
            age.append(raw_age[i])
            img_paths.append(raw_path[i])

    # Convert images path to images.
    imgs = [
        np.asarray(spm.imresize(
            spm.imread(path, flatten=1),
            (128, 128)), dtype=np.float32)
        for path in img_paths
    ]

    # Option to return paths to images or paths to images
    input_features = img_paths if get_paths else imgs
     
    data = {
        'image_inputs': np.array(input_features),
        'age_labels': np.array(age)
    }

    print("Number of samples reduced to: {}".format(len(data['image_inputs'])))
    pkl_dir = os.path.join( path_to_images, "pkl_folder")

    # If file dir doesnt exist create.
    if not os.path.isdir(pkl_dir):
        os.makedirs(pkl_dir)

    # Pickle file
    with open(os.path.join(
        pkl_dir,
        "imdb_data_{}.pkl".format(n_samples)), 'wb') as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description='IMDB data reformat script.')
    parser.add_argument('--n-samples', '-n', type=int, default=50,
                        help='The number of samples to use.')
    parser.add_argument('--path-to-meta', '-p2m', type=str, default='/media/ubuntu/Samsung_T3/workspace/imdb_data/meta/imdb',
                        help='Path to Matlab file.')
    parser.add_argument('--matlab-file', '-m', type=str, default='imdb.mat',
                        help='Matlab file.')
    parser.add_argument('--path-to-images', '-p2i', type=str, default='/media/ubuntu/Samsung_T3/workspace/imdb_data/imdb_crop/imdb_crop',
                        help='Path to where the images have been saved.')
    parser.add_argument('--get-paths', '-ip', type=bool, default=True,
                        help='True if input features returned are paths to imagess.')
   
    args = parser.parse_args()
    print(args.get_paths)
    imdb_dict = matlab_to_numpy(
        args.path_to_meta,
        args.matlab_file,
        args.path_to_images
    )
    print("Dictionary created...")

    print("Converting {} samples. (0=all samples)".format(args.n_samples))
    create_and_dump(args.get_paths, args.path_to_images, imdb_dict, args.n_samples)

    print("File dumped to imdb_data_{}.pkl.".format(args.n_samples))


if __name__ == "__main__":
    main()

# @author Tasuku Miura
# @brief PyTorch implementation of PilotNet (Assumes CUDA enabled)

import os
import pickle

from skimage import io, transform
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from pilot_net import *

import math
import copy


class DriveDataset(Dataset):

    """

    Custom dataset to handle Udacity drive data.

    """

    def __init__(self, csv_file, root_dir, bags, transform=None):
        self._csv_file = csv_file
        self._root_dir = root_dir
        self._bags = bags
        self._transform = transform
        self._frames = self._get_frames()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self._root_dir,
                                self._frames['bag'].iloc[idx],
                                self._frames['filename'].iloc[idx])
        image = io.imread(img_name)
        target = self._frames['angle'].iloc[idx]
        sample = {'image': image, 'image_path': img_name, 'steer': target}

        if self._transform is not None:
            sample['image'] = self._transform(sample['image'])

        return sample

    def _get_frames(self):
        file_paths = [
            (bag, os.path.join(self._root_dir, bag, self._csv_file))
            for bag in self._bags
        ]

        frames = []
        for bag, path in file_paths:
            df = pd.read_csv(path)
            df['bag'] = bag
            frames.append(df[df.frame_id == 'center_camera'])
        return pd.concat(frames, axis=0)


class SequenceDriveDataset(Dataset):

    """

    Custom dataset to convert Udacity drive data to inputs of sequences.

    """

    def __init__(self, csv_file, root_dir, bags, time_steps, transform=None):
        self._csv_file = csv_file
        self._root_dir = root_dir
        self._bags = bags
        self._steps = time_steps + 1  # +1 to adjust for indexing
        self._transform = transform
        self._frames = self.get_frames()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        images = []
        for step in range(1, self._steps):
            img_name = os.path.join(
                self._root_dir,
                self._frames['filename_L{}'.format(step)].iloc[idx]
            )
            images.append(io.imread(img_name))
        target = self._frames['angle'].iloc[idx]
        # 'image_path' is path to image at time t (need for testing and
        # visualization. (see test_sequence() in train.py)
        sample = {
            'image': images,
            'image_path': self._frames['filename'].iloc[idx],
            'steer': target
        }

        if self._transform is not None:
            transformed = []
            for image in sample['image']:
                transformed.append(self._transform(image))
            sample['image'] = transformed

        return sample

    def get_frames(self):
        file_paths = [
            (bag, os.path.join(self._root_dir, bag, self._csv_file))
            for bag in self._bags
        ]
        frames = []
        unused = ['width', 'height', 'lat', 'long', 'alt', 'timestamp']
        for bag, path in file_paths:
            df = pd.read_csv(path)
            # Add bag column and generate correct paths.
            df['bag'] = bag
            df.filename = df.bag + '/' + df.filename
            # Remove unused to columns.
            for col in unused:
                del df[col]

            frames.append(df[df.frame_id == 'center_camera'])

        all_df = pd.concat(frames, axis=0, ignore_index=True)

        lags_df = (pd.concat(
            [all_df.filename.shift(i) for i in range(self._steps)],
            axis=1,
            keys=['filename'] + ['filename_L%s' % i for i in range(1, self._steps)]
        ).dropna()
        )
        # To ensure that only past frames are used to predict current steering.
        # (e.g to predict target at t, we use t-1,t-2, etc.)
        lags_df = lags_df.drop(['filename'], axis=1)
        final_df = pd.merge(all_df, lags_df, left_index=True, right_index=True)
        return final_df

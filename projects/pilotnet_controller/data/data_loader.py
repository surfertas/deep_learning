import pandas as pd
import numpy as np
import random
import os

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from io import BytesIO
from PIL import Image, ImageChops
from google.cloud import storage

from .transforms import basenet_transforms


class ControllerDataset(Dataset):

    def __init__(self, cfg, split, datasets, transform, throttle_include=False):
        self.cfg = cfg
        self.data_csv = datasets[split]
        self.features = self.data_csv['url']
        self.target = self.data_csv[['throttle', 'steer']]

        self.n_samples = len(self.target)
        self.transform = transform
        self.throttle_include = throttle_include

        self.debug = self.cfg.MODEL.DEBUG
        self.augment = self.cfg.IMAGE.DO_AUGMENTATION
        self.gray = (self.cfg.MODEL.CNN.INPUT_CHANNELS == 1)
        self.sigma = self.cfg.STEER.AUGMENTATION_SIGMA
        self.crop_height = self.cfg.IMAGE.CROP_HEIGHT

        self.cwd = os.path.dirname(os.path.abspath(__file__))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Target is a array consisting of [throttle, steer]
        target = self.target.iloc[idx].values.tolist()

        image = self._get_image(self.features.iloc[idx])

        if self.augment:
            if random.choice([True, False]):
                image = image[:, ::-1, :]

                target[1] *= -1.

                # Add some noise to the steering.
                target[1] += np.random.normal(loc=0, scale=self.sigma)

                # Clip between -1, 1
                target[1] = np.clip(target[1], -1.0, 1.0)

        # If DO_AUGMENTATION is true, jitter will be applied to images, else just a resize
        # will be applied.
        image = self.transform(Image.fromarray(image))

        if self.debug:
            save_image(image, os.path.join(self.cwd, f'train-images/image+{idx}.png'))

        # Train on both throttle, steer or just steer
        target = target if self.throttle_include else np.array([target[1]])
        target = torch.FloatTensor(target)

        print(image[0])
        return image, target

    def _get_image(self, path):
        if self.gray:
            image = np.array(Image.open(path).convert('L'))
        else:
            image = np.array(Image.open(path))

        image = self._preprocess(image)
        return image

    def _preprocess(self, image):
        # crop image (remove useless information)
        cropped = image[range(*self.crop_height)] if self.gray else image[range(*self.crop_height), :, :]
        return cropped


class GrayScaleDifferenceDataset(ControllerDataset):

    def __init__(self, cfg, split, datasets, transform, n_steps=1, throttle_include=False):
        super().__init__(cfg, split, datasets, transform, throttle_include)

        self.n_steps = n_steps
        self.prev_frame = np.zeros_like(self._get_image(self.features.iloc[0]))
        #self.frames = np.array([self.prev_frame] * n_steps)

    def __getitem__(self, idx):
        target = self.target.iloc[idx].values.tolist()

        image = self._get_image(self.features.iloc[idx])

        # If n_step = 0 just use the gray frame image otherwise take the n_step difference
        if self.n_steps != 0:
            image_diff = np.array(ImageChops.subtract(Image.fromarray(image), Image.fromarray(self.prev_frame)))
            self.prev_frame = image

        if self.augment:
            if random.choice([True, False]):
                image_diff = image_diff[:, ::-1]
        #        im = Image.fromarray(image_diff)
        #        im.save(f'flipped+{idx}.png')

                target[1] *= -1.

                # Add some noise to the steering.
                target[1] += np.random.normal(loc=0, scale=self.sigma)

                # Clip between -1, 1
                target[1] = np.clip(target[1], -1.0, 1.0)

        # If DO_AUGMENTATION is true, jitter will be applied to images, else just a resize
        # will be applied.
        # ToTensor() converts into 0-1 range.
        image_diff = self.transform(Image.fromarray(image_diff))

        if self.debug:
            save_image(image_diff, os.path.join(self.cwd, f'train-images/image+{idx}.png'))

        # Train on both throttle, steer or just steer
        target = target if self.throttle_include else np.array([target[1]])
        target = torch.FloatTensor(target)

        return image_diff, target


def fetch_dataloader(types, data_dir, csv_filename, cfg):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        cfg: parameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    data_csv = pd.read_csv(os.path.join(data_dir, csv_filename))

    if 'test' not in types:
        # NOTE: dont use random sampling, as dealing with sequences. Think about later.
        # Split into train and eval
        n_samples = len(data_csv)
        n_train = int(n_samples * cfg.DATALOADER.TRAIN)
        data_train = data_csv[:n_train]
        data_eval = data_csv[n_train:]

        # Split into train and eval
        datasets = {
            'train': data_train,
            'val': data_eval}
        print("size of data train set: {}".format(len(data_train)))
        print("size of data val set: {}".format(len(data_eval)))
    else:
        datasets = {
            'test': data_csv}
        print("size of data test set: {}".format(len(data_csv)))

    # Set up transformers.
    transforms = basenet_transforms(cfg)
    train_transformer = transforms["train_transformer"]
    eval_transformer = transforms["eval_transformer"]

    if (cfg.MODEL.CNN.INPUT_CHANNELS == 1):
        UseDataset = GrayScaleDifferenceDataset
    else:
        UseDataset = ControllerDataset

    # Create datasets.
    for split in types:
        # use the train_transformer if training data, else use eval_transformer without random flip
        # if using grayscale difference, set shuffle to False
        if split == 'train':
            dl = DataLoader(UseDataset(cfg, split, datasets, train_transformer),
                            batch_size=cfg.INPUT.BATCH_SIZE,
                            shuffle=cfg.DATASETS.SHUFFLE,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            pin_memory=cfg.DATALOADER.PIN_MEMORY)
        else:
            dl = DataLoader(UseDataset(cfg, split, datasets, eval_transformer),
                            batch_size=cfg.INPUT.BATCH_SIZE,
                            shuffle=cfg.DATASETS.SHUFFLE,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            pin_memory=cfg.DATALOADER.PIN_MEMORY)

        dataloaders[split] = dl

    return dataloaders

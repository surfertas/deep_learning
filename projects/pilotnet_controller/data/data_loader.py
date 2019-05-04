import pandas as pd
import numpy as np
import random
import os

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from io import BytesIO
from PIL import Image
from google.cloud import storage

from .transforms import basenet_transforms 



class ControllerDataset(Dataset):
    
    def __init__(self, cfg, bucket_name, split, datasets, transform, throttle_include=False):
        self.cfg = cfg
        self.bucket_name = bucket_name
        self.data_csv = datasets[split]
        self.features = self.data_csv['url']
        self.target = self.data_csv[['throttle', 'steer']]

        self.n_samples = len(self.target)
        self.augment = self.cfg.IMAGE.DO_AUGMENTATION
        self.transform = transform
        self.throttle_include = throttle_include

        self.debug = cfg.MODEL.DEBUG
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
                target[1] += np.random.normal(loc=0, scale=self.cfg.STEER.AUGMENTATION_SIGMA)
            
                # Clip between -1, 1
                target[1] = np.clip(target[1],-1.0,1.0)
       
       
        # If DO_AUGMENTATION is true, jitter will be applied to images, else just a resize
        # will be applied.
        image = self.transform(Image.fromarray(image))

        if self.debug:
            save_image(image, os.path.join(self.cwd, "train-images/image+{}.png".format(idx)))
    
        # Train on both throttle, steer or just steer
        target = target if self.throttle_include else np.array([target[1]])
        target = torch.FloatTensor(target)
    
        return image, target

    def _get_image(self, gs_path):
        image = np.array(Image.open(gs_path))
        image = self._preprocess(image)
        return image

    def _preprocess(self, image):
        # crop image (remove useless information)
        cropped = image[range(*self.cfg.IMAGE.CROP_HEIGHT), :, :]
        return cropped
        

def fetch_dataloader(types, bucket_name, data_dir, csv_filename, cfg):
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
    n_samples = len(data_csv)

    # NOTE: dont use random sampling, as dealing with sequences. Think about later.
    # Split into train and test
    n_train = int(n_samples * cfg.DATALOADER.TRAIN)
    data_train = data_csv[:n_train]
    data_test = data_csv[n_train:]

    # Split into train and eval
    n_train_eval = int(n_train * (1 - cfg.DATALOADER.VAL))
    data_train_new = data_train[:n_train_eval]
    data_eval = data_train[n_train_eval:]
    print("size of data train set: {}".format(len(data_train_new)))
    print("size of data val set: {}".format(len(data_eval)))
    print("size of data test set: {}".format(len(data_test)))

    datasets = {
        'train': data_train_new,
        'val': data_eval,
        'test': data_test}

    transforms = basenet_transforms(cfg)
    train_transformer =  transforms["train_transformer"]
    eval_transformer =  transforms["eval_transformer"]

    for split in ['train', 'val', 'test']:
        if split in types:
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(ControllerDataset(cfg, bucket_name, split, datasets, train_transformer),
                                batch_size=cfg.INPUT.BATCH_SIZE,
                                shuffle=cfg.DATASETS.SHUFFLE,
                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                pin_memory=cfg.DATALOADER.PIN_MEMORY)
            else:
                dl = DataLoader(ControllerDataset(cfg, bucket_name, split, datasets, eval_transformer),
                                batch_size=cfg.INPUT.BATCH_SIZE,
                                shuffle=cfg.DATASETS.SHUFFLE,
                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                pin_memory=cfg.DATALOADER.PIN_MEMORY)

            dataloaders[split] = dl

    return dataloaders

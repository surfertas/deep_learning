import pandas as pd
import random
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from io import BytesIO
from PIL import Image
from google.cloud import storage

from transforms import basenet_transforms 



class ControllerDataset(Dataset):
    
    def __init__(self, cfg, bucket_name, split, datasets, transform):
        self.cfg = cfg
        self.bucket_name = bucket_name
        self.data_csv = datasets[split]
        self.features = self.data_csv['cloud_url']
        self.target = self.data_csv[['throttle', 'steer']]
        self.transform = transform

    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        image = self._get_image(self.features.iloc[idx])
        image = self.transform(image)
        print(image.shape)
        target = torch.FloatTensor(self.target.iloc[idx].values.tolist())
        return image, target

    def _get_image(self, gs_path):
        gs_path = gs_path.split("/")
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob("/".join(gs_path[-2:]))
        image_string = blob.download_as_string()
        image = Image.open(BytesIO(image_string))
        #image = self._preprocess(image)
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
                # NOTE: set shuffle to false as not sure what implications are for time series.
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

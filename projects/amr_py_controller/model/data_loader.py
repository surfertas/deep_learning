import pandas as pd
import random
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from io import BytesIO
from PIL import Image
from google.cloud import storage


# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

class ControllerDataset(Dataset):
    
    def __init__(self, bucket_name, split, datasets, transform):
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
        target = torch.FloatTensor(self.target.iloc[idx].values.tolist())
        return image, target

    def _get_image(self, gs_path):
        gs_path = gs_path.split("/")
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob("/".join(gs_path[-2:]))
        image_string = blob.download_as_string()
        image = Image.open(BytesIO(image_string))
        return image



def fetch_dataloader(types, bucket_name, data_dir, csv_filename, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    data_csv = pd.read_csv(os.path.join(data_dir, csv_filename))
    n_samples = len(data_csv)

    # NOTE: dont use random sampling, as dealing with sequences. Think about later.
    # Split into train and test
    n_train = int(n_samples * params.splits["train"])
    data_train = data_csv[:n_train]
    data_test = data_csv[n_train:]

    # Split into train and eval
    n_train_eval = int(n_train * (1 - params.splits["val"]))
    data_train = data_train[:n_train_eval]
    data_eval = data_train[n_train_eval:]

    datasets = {
        'train': data_train,
        'val': data_eval,
        'test': data_test}

    for split in ['train', 'val', 'test']:
        if split in types:
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                # NOTE: set shuffle to false as not sure what implications are for time series.
                dl = DataLoader(ControllerDataset(bucket_name, split, datasets, train_transformer),
                                batch_size=params.batch_size,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            else:
                dl = DataLoader(ControllerDataset(bucket_name, split, datasets, eval_transformer),
                                batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

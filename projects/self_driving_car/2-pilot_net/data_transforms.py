# @author Tasuku Miura
# @brief Transforms used for data augmentation

import os
import pickle

from skimage import io, transform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils


def imagenet_transforms():
    """ Transforms for imagenet trained models. """
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transforms': train_transforms,
        'eval_transforms': eval_transforms,
    }


def pilotnet_transforms():
    """ Transforms for pilotnet """
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor()
    ])

    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor()
    ])
    return {
        'train_transforms': train_transforms,
        'eval_transforms': eval_transforms,
    }

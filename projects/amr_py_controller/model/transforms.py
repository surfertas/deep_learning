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

def basenet_transforms():
    train_transformer = transforms.Compose([
        transforms.Resize((124,124)),
        transforms.ToTensor()])  # transform it into a torch tensor

    # loader for evaluation, keep separate as transformer for train can be different
    eval_transformer = transforms.Compose([
        transforms.Resize((124,124)),
        transforms.ToTensor()])  # transform it into a torch tensor

    return {
        'train_transformer': train_transformer,
        'eval_transformer': eval_transformer,
    }


def imagenet_transforms():
    """ Transforms for imagenet trained models. """
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformer': train_transformer,
        'eval_transformer': eval_transformer,
    }


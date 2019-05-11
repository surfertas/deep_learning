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

from torchvision import transforms, utils

def basenet_transforms(cfg):

    if cfg.IMAGE.DO_AUGMENTATION:
        train_transformer = transforms.Compose([
            transforms.Resize((cfg.IMAGE.TARGET_HEIGHT,cfg.IMAGE.TARGET_WIDTH)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()])  # transform it into a torch tensor
    else:
        train_transformer = transforms.Compose([
            transforms.Resize((cfg.IMAGE.TARGET_HEIGHT,cfg.IMAGE.TARGET_WIDTH)),
            transforms.ToTensor()])  # transform it into a torch tensor


    # loader for evaluation, keep separate as transformer for train can be different
    eval_transformer = transforms.Compose([
        transforms.Resize((cfg.IMAGE.TARGET_HEIGHT,cfg.IMAGE.TARGET_WIDTH)),
        transforms.ToTensor()])  # transform it into a torch tensor

    return {
        'train_transformer': train_transformer,
        'eval_transformer': eval_transformer,
    }


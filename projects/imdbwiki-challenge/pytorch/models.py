# @author Tasuku Miura

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VggFE(nn.Module):

    """
    Feature Extract Vgg16
    """

    def __init__(self, num_classes):
        super(VggFE, self).__init__()
        self.num_classes = num_classes
        self.vgg16fe = models.vgg16(pretrained=True)
        # Set grad to false to freeze.
        for param in self.vgg16fe.parameters():
            param.requires_grad = False

        # Default sets requires_grad to true,
        # so final fc can be optimized.
        num_ftrs = 512 * 7 * 7
        self.vgg16fe.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.vgg16fe(x)
        return x

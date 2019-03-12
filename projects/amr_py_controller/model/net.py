"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import models


class AlexNetConv4(nn.Module):

    """
    Use conv layers as feature extractor.
    """

    def __init__(self):
        super(AlexNetConv4, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)

        self.features = nn.Sequential(
            # stop at conv4
            *list(self.alexnet.features.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x


class AlexNetTransferFE(nn.Module):

    """
    Fine tuning with AlexNet. Remove classification layers (basically all the fc
    layers) and retrain for regression.
    """

    def __init__(self):
        super(AlexNetTransferFE, self).__init__()
        self.features = AlexNetConv4()
        self.fc1 = nn.Linear(43264, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    """
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        # print(self.num_channels)
        self.fc1 = nn.Linear(15*15*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 2)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #print(s.shape)
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8
        #print(s.shape)
        
        # flatten the output for each image
        s = s.view(-1, 15*15*self.num_channels*4)           # batch_size x 14*14*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        return s


def rmse(outputs, targets):
    residuals = (outputs - targets)
    return np.sqrt(np.mean(residuals**2))
    


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'rmse': rmse
}

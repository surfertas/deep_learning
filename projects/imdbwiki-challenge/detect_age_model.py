#!/usr/bin/env python

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions


class CNN(chainer.Chain):

    def __init__(self, n_out):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, ksize=3, stride=2, pad=1),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(32, 64, ksize=3, stride=2, pad=1),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 128, ksize=3, stride=2, pad=1),
            fc4=L.Linear(2048, 625),
            fc5=L.Linear(625, n_out)
        )

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.dropout(h, ratio=0.3, train=True)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.dropout(h, ratio=0.3, train=True)
        h = F.relu(self.conv3(h))
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.3, train=True)
        return self.fc5(h)

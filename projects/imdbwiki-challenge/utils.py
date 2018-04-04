#!/usr/bin/env python

import os
import numpy as np
import pickle
import chainer
from scipy import linalg


Age = {
   "YOUNG": 0,
   "MIDDLE": 1,
   "OLD": 2,
   "VOLD": 3
}

def classify_age(x):
    """
    Used to classify the age to reduce the number of classifications
    used in the label for training.
    Args:
        x - Age unit.
    """
    if x < 30:
        return Age["YOUNG"]
    elif x >= 30 and x < 45:
        return Age["MIDDLE"]
    elif x >= 45 and x < 60:
        return Age["OLD"]
    else:
        return Age["VOLD"]


def preprocess_train(data):
    """
    Use ZCA Whitening for pre-process.
    Args:
        data - training data to preprocess.
    Returns:
        components - components related to ZCA.
        mean - mean of data set along axis=0.
        whiten - whitened version of data.
    """
    data = np.reshape(data, (-1, 128 * 128 * 1))
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)
    white = np.reshape(whiten, (-1, 1, 128, 128))
    return components, mean, white


def preprocess_test(data, components, mean):
    """
    Use components and mean outputted by ZCA whitening on training data.
    Args:
        data - test data to preprocess.
        compenents - components from ZCA processing on training data.
        mean - mean from training data.
    Returns:
        white - whitened test data.
    """
    mdata = np.reshape(data, (-1, 128 * 128 * 1))
    white = np.dot(mdata - mean, components.T)
    white = np.reshape(white, (-1, 1, 128, 128))
    return white


def get_data(fpath, fname, simple, white, split=0.3):
    """ Gets data set, returns two array of tuples of (input, output).
    Args:
        fpath - path to file.
        fname - name of file.
        simple - Determines if the simple classification is desired.
        split - parameter used to define training and testing subsets.
    Returns:
        train - train data set, subset of original data.
        test - test data set, subset of original data.
        hist - statistics related to age distribution.
    """
    with open(os.path.join(fpath, fname), 'rb') as f:
        data = pickle.load(f)

    if simple:
        Y = [classify_age(y) for y in data['age_labels']]
        bins = [0, 1, 2, 3, 4]
    else:
        Y = data['age_labels']
        bins = range(101)

    Y = np.asarray(Y, dtype=np.int32)
    X = data['image_inputs']
    X = np.asarray(X, dtype=np.float32).reshape(-1, 1, 128, 128)

    # Calculate the age distribution of the data.
    hist = np.histogram(Y, bins=bins, density=True)

    # Preprocess whitening
    spl = int(len(X) * (1 - split))
    trX, trY = X[:spl], Y[:spl]
    teX, teY = X[spl:], Y[spl:]
    if white:
        print("Whitening data...")
        cmpts, mean, trX = preprocess_train(trX)
        teX = preprocess_test(teX, cmpts, mean)

    train = zip(trX, trY)
    test = zip(teX, teY)
    return train, test, hist[0]

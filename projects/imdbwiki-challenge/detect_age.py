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
from chainer.datasets import tuple_dataset
from detect_age_data import get_data
from detect_age_model import CNN


def main():
    parser = argparse.ArgumentParser(description='Chainer-Tutorial: CNN')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of times to train on data set')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID: -1 indicates CPU')
    parser.add_argument('--file_name', '-f', type=str, default='imdb_data_2000.pkl',
                        help='Data set (image, label)')
    parser.add_argument('--white', '-w', type=bool, default=False,
                        help='Preprocess whitening')
    args = parser.parse_args()

    train, test, age_stats = get_data(args.file_name, simple=True, white=args.white, split=0.3)
    print("Age distribution of data is : {}".format(age_stats))

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    model = L.Classifier(CNN(4))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.RMSprop(lr=0.001, alpha=0.9)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))

    report_params = [
        'epoch',
        'main/loss',
        'validation/main/loss',
        'main/accuracy',
        'validation/main/accuracy',
        'elapsed_time'
    ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))

    # Run trainer
    trainer.run()

if __name__ == "__main__":
    main()

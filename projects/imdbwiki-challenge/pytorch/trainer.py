import os
import pickle
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

import mydataloader
import transformer
import models
import utils


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader):
    model.train()
    print("Epoch {} starting.".format(epoch))
    epoch_loss = 0
    for batch in train_loader:
        data, target = batch['faces'].cuda(), batch['ages'].cuda()
        data = Variable(data).type(torch.cuda.FloatTensor)
        target = Variable(target).type(torch.cuda.LongTensor)

        predict = model(data)
        loss = loss_fn(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data[0]

    epoch_loss /= len(train_loader.dataset)
    print("Epoch {:.4f}: Train set: Average loss: {:.6f}\t".format(epoch, epoch_loss))
    # log_value('train_loss', epoch_loss, epoch)


def validate(epoch, model, loss_fn, optimizer, valid_loader):
    model.eval()
    valid_loss = 0
    for batch in valid_loader:
        data, target = batch['faces'].cuda(), batch['ages'].cuda()
        data = Variable(data, volatile=True).type(torch.cuda.FloatTensor)
        target = Variable(target).type(torch.cuda.LongTensor)
        predict = model(data)
        valid_loss += loss_fn(predict, target).data[0]  # sum up batch loss

    valid_loss /= len(valid_loader.dataset)
    print('Valid set: Average loss: {:.6f}\n'.format(valid_loss))
    # log_value('valid_loss', valid_loss, epoch)
    return valid_loss


def main(args):
    # Current implementation requires cuda.
    cuda = torch.cuda.is_available()
    assert(cuda)

    # Set random seed to 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set file paths.
    ckpt_path = os.path.join(args.root_dir, 'output')
    log_path = os.path.join(args.root_dir, 'log')

    utils.create_dir(ckpt_path)
    utils.create_dir(log_path)

    # Configure tensorboard log dir
    # configure(os.path.join(args.root_dir, 'log'))

    train_pickle_file = args.train_data

    # Get transforms
    transforms = transformer.imagenet_transforms()
    train_transforms = transforms['train_transforms']
    pre_process = transforms['eval_transforms']

    # Set Up data
    train_data_aug = mydataloader.IMDBFaces(
        train_pickle_file,
        args.root_dir,
        args.reduce_age_classes,
        train_transforms
    )
    train_data_orig = mydataloader.IMDBFaces(
        train_pickle_file,
        args.root_dir,
        args.reduce_age_classes,
        pre_process
    )

    train_data = ConcatDataset([train_data_orig, train_data_aug])
    print("Train data size: {}".format(len(train_data)))

    # Create train and validation samplers
    indices = list(range(len(train_data)))
    n_train = int((1 - args.train_valid_split) * len(train_data))
    train_sampler = SubsetRandomSampler(indices[:n_train])
    valid_sampler = SubsetRandomSampler(indices[n_train:])

    # Create data loader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4
    )
    valid_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=4
    )
    print("Data loaded...")

    num_classes = 4
    # Initiate model.
    model = models.VggFE(num_classes).cuda()

    if args.resume:
        print("Resuming from checkpoint")
        ckpt = torch.load(os.path.join(ckpt_path, args.ckpt_file_name))
        model.load_state_dict(ckpt['state_dict'])

    # Set up optimizer and define loss function.
    parameters = model.vgg16fe.classifier.parameters()
    optimizer = torch.optim.Adam(parameters)
    loss_fn = nn.CrossEntropyLoss()
    print("Model setup...")

    # Train and validate
    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        train_one_epoch(epoch, model, loss_fn, optimizer, train_loader)
        ave_valid_loss = validate(epoch, model, loss_fn, optimizer, valid_loader)

        if ave_valid_loss < best_valid_loss:
            best_valid_loss = ave_valid_loss
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(ckpt_path, 'checkpoint.pth.tar'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMDB Wiki trainer for age classification.')
    parser.add_argument('--root-dir', type=str, default='.',
                        help='path to root')
    parser.add_argument('--train-data', type=str, default='predictions.pickle',
                        help='filename containing train data')
    parser.add_argument('--ckpt-file-name', type=str, default='checkpoint.pth.tar',
                        help='name of checkpoint file')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training from checkpoint')
    parser.add_argument('--reduce-age-classes', type=bool, default=True,
                        help='reduce the number of age classes')
    parser.add_argument('--train-valid-split', type=float, default='0.2',
                        help='x% valid split')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--valid-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for validation (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    main(args)

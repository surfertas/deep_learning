# @author Tasuku Miura
# @brief PyTorch implementation of PilotNet

import os
import pickle

from skimage import io, transform
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import math


class DriveDataset(Dataset):

    """

    Custom dataset to handle Udacity drive data.

    """

    def __init__(self, csv_file, root_dir, bags, transform=None):
        self._csv_file = csv_file
        self._root_dir = root_dir
        self._bags = bags
        self._transform = transform
        self._frames = self._get_frames()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self._root_dir,
                                self._frames['bag'].iloc[idx],
                                self._frames['filename'].iloc[idx])
        image = io.imread(img_name)
        target = self._frames['angle'].iloc[idx]
        sample = {'image': image, 'image_path': img_name, 'steer': target}

        if self._transform:
            sample['image'] = self._transform(sample['image'])

        return sample

    def _get_frames(self):
        file_paths = [
            (bag, os.path.join(self._root_dir, bag, self._csv_file))
            for bag in self._bags
        ]

        frames = []
        for bag, path in file_paths:
            df = pd.read_csv(path)
            df['bag'] = bag
            frames.append(df[df.frame_id == 'center_camera'])
        return pd.concat(frames, axis=0)


class PilotNet(nn.Module):

    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = (x - 0.5) * 2.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader):
    model.train()
    print("Epoch {} starting.".format(epoch))
    epoch_loss = 0
    for batch in train_loader:
        data, target = batch['image'].cuda(), batch['steer'].cuda()
        data = Variable(data).type(torch.cuda.FloatTensor)
        target = Variable(target).type(torch.cuda.FloatTensor)

        predict = model(data)
        loss = loss_fn(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data[0]

    epoch_loss /= len(train_loader.dataset)
    print("Epoch {:.4f}: Train set: Average loss: {:.6f}\t".format(epoch, epoch_loss))


def validate(epoch, model, loss_fn, optimizer, valid_loader):
    model.eval()
    valid_loss = 0
    for batch in valid_loader:
        data, target = batch['image'].cuda(), batch['steer'].cuda()
        data = Variable(data, volatile=True).type(torch.cuda.FloatTensor)
        target = Variable(target).type(torch.cuda.FloatTensor)
        predict = model(data)
        valid_loss += loss_fn(predict, target).data[0]  # sum up batch loss

    valid_loss /= len(valid_loader.dataset)
    print('Valid set: Average loss: {:.6f}\n'.format(valid_loss))


def test(model, loss_fn, optimizer, test_loader):
    model.eval()
    images = []
    targets = []
    predicts = []
    test_loss = 0
    for batch in test_loader:
        data, target = batch['image'].cuda(), batch['steer'].cuda()
        data = Variable(data, volatile=True).type(torch.cuda.FloatTensor)
        target = Variable(target).type(torch.cuda.FloatTensor)
        output = model(data)
        test_loss += loss_fn(output, target).data[0]  # sum up batch loss

        # Store image path as raw image too large.
        images.append(batch['image_path'])
        targets.append(target.data.cpu().numpy())
        predicts.append(output.data.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

    data_dict = {
        "image": images,
        "steer_target": targets,
        "steer_pred": predicts
    }

    with open("pyt_predictions.pickle", 'wb') as f:
        pickle.dump(data_dict, f)
        print("Predictions pickled...")


def main():
    bags = ['bag1', 'bag2', 'bag4', 'bag5', 'bag6']
    root_dir = r'/home/ubuntu/ws/deep_learning/projects/self_driving_car/1-pilot_net/data'
    train_csv_file = r'train_interpolated.csv'
    valid_csv_file = r'valid_interpolated.csv'

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3),
        transforms.ToTensor()
    ])

    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor()
    ])

    train_data = DriveDataset(train_csv_file, root_dir, bags, train_transforms)
    print("Train data size: {}".format(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

    valid_data = DriveDataset(valid_csv_file, root_dir, bags, valid_transforms)
    print("Valid data size: {}".format(len(valid_data)))

    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)

    print("Data loaded...")
    model = PilotNet().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    print("Model setup...")

    for epoch in range(10):
        train_one_epoch(epoch, model, loss_fn, optimizer, train_loader)
        validate(epoch, model, loss_fn, optimizer, valid_loader)

    test(model, loss_fn, optimizer, valid_loader)

if __name__ == "__main__":
    main()

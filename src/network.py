# coding:utf-8

import librosa
import wave
import numpy as np

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models
from torch.autograd import Variable


def num_flat_features(x):
    # (32L, 50L, 11L, 14L), 32 is batch_size
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class EnvNet(nn.Module):
    def __init__(self):
        super(EnvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(kernel_size=160, stride=160)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = F.relu(self.conv1(x))   # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))   # (batchSize, 40L, 24000L)
        x = self.bn2(x)
        x = F.relu(self.pool2(x))   # (batchSize, 40L, 150L)
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv3(x))   # (batchSize, 50L, 33L, 138L)
        x = self.bn3(x)
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.pool4(x))   # (batchSize, 50L, 11L, 14L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnvNet_v1(nn.Module):
    def __init__(self):
        super(EnvNet_v1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(kernel_size=160, stride=160)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = self.conv1(x)  # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)   # (batchSize, 40L, 24000L)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)   # (batchSize, 40L, 150L)

        x = torch.unsqueeze(x, 1)
        x = self.conv3(x)   # (batchSize, 50L, 33L, 138L)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)   # (batchSize, 50L, 11L, 14L)

        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnvNet44100_srf(nn.Module):
    """
    rf:8
    """
    def __init__(self):
        super(EnvNet44100_srf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        # (50, 11, 14)

        self.fc1 = nn.Linear(8800, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24000L)
        x = self.conv1(x)  # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)   # (batchSize, 40L, 24000L)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)   # (batchSize, 40L, 441L)

        x = torch.unsqueeze(x, 1)
        x = self.conv3(x)   # (batchSize, 50L, 33L, 429L)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x) # (batchSize, 50L, 11L, 85L)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)   # (batchSize, 50L, 11L, 16L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnvNet44100_mrf(nn.Module):
    """
    rf:32
    """
    def __init__(self):
        super(EnvNet44100_mrf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=32, stride=5, padding=16)
        self.bn1 = nn.BatchNorm1d(40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=32, stride=5, padding=15)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        # (50, 11, 14)

        self.fc1 = nn.Linear(8800, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24000L)
        x = self.conv1(x)  # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)   # (batchSize, 40L, 24000L)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)   # (batchSize, 40L, 441L)

        x = torch.unsqueeze(x, 1)
        x = self.conv3(x)   # (batchSize, 50L, 33L, 429L)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x) # (batchSize, 50L, 11L, 85L)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)   # (batchSize, 50L, 11L, 16L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnvNet44100_lrf(nn.Module):
    """
    rf:80
    """
    def __init__(self):
        super(EnvNet44100_lrf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=80, stride=1, padding=40)
        self.bn1 = nn.BatchNorm1d(40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=80, stride=1, padding=39)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        # (50, 11, 14)

        self.fc1 = nn.Linear(8800, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24000L)
        x = self.conv1(x)  # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)   # (batchSize, 40L, 24000L)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)   # (batchSize, 40L, 441L)

        x = torch.unsqueeze(x, 1)
        x = self.conv3(x)   # (batchSize, 50L, 33L, 429L)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x) # (batchSize, 50L, 11L, 85L)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)   # (batchSize, 50L, 11L, 16L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNet0_srf(nn.Module):
    def __init__(self):
        super(EnvNet0_srf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(40)
        self.pool1 = nn.MaxPool1d(kernel_size=160, stride=160)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = self.conv1(x)  # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = torch.unsqueeze(x, 1)   # (batchSize, 1L, 40L, 150L)
        print(x.size())
        exit(0)
        return x

class EnvNet0_mrf(nn.Module):
    def __init__(self):
        super(EnvNet0_mrf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=32, stride=1, padding=16)
        self.bn1 = nn.BatchNorm1d(40)
        self.pool1 = nn.MaxPool1d(kernel_size=160, stride=160)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = self.conv1(x)  # (batchSize, 40L, 24001L)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = torch.unsqueeze(x, 1)   # (batchSize, 1L, 40L, 150L)
        print(x.size())
        exit(0)
        return x

class EnvNet0_lrf(nn.Module):
    def __init__(self):
        super(EnvNet0_lrf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=320, stride=1, padding=159)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=320, stride=1, padding=159)
        self.pool2 = nn.MaxPool1d(kernel_size=160, stride=160)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = F.relu(self.conv1(x))   # (batchSize, 40L, 24001L)
        x = F.relu(self.conv2(x))   # (batchSize, 40L, 24000L)
        x = F.relu(self.pool2(x))   # (batchSize, 40L, 150L)
        x = torch.unsqueeze(x, 1)

        return x


class EnvNet2(nn.Module):
    """
    tConv + fConv + part2
    """
    def __init__(self):
        super(EnvNet2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=8, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=160, stride=160)
        self.conv2 = nn.Conv1d(in_channels=150, out_channels=150, kernel_size=8, stride=1, padding=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = F.relu(self.conv1(x))   # (batchSize, 40L, 24001L)
        x = F.relu(self.pool1(x))   # (batchSize, 40L, 150L)
        x = torch.transpose(x, 1, 2)   # (batchSize, 150L, 80L)
        x = F.relu(self.conv2(x))   # (batchSize, 150L, 81L)
        x = F.relu(self.pool2(x))   # (batchSize, 150L, 40L)

        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv3(x))   # (batchSize, 50L, 33L, 138L)
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool4(x))   # (batchSize, 50L, 11L, 14L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNet3(nn.Module):
    """
    tConv + tConv + fConv + part2
    """
    def __init__(self):
        super(EnvNet3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=8, stride=1, padding=3)
        self.conv2 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=8, stride=1, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=160, stride=160)

        self.conv3_0 = nn.Conv1d(in_channels=150, out_channels=150, kernel_size=8, stride=1, padding=4)
        self.pool3_0 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = F.relu(self.conv1(x))   # (batchSize, 80L, 24001L)
        x = F.relu(self.conv2(x))  # (batchSize, 80L, 24000L)
        x = F.relu(self.pool2(x))   # (batchSize, 80L, 150L)

        x = torch.transpose(x, 1, 2)   # (batchSize, 150L, 80L)
        x = F.relu(self.conv3_0(x))   # (batchSize, 150L, 81L)
        x = F.relu(self.pool3_0(x))   # (batchSize, 150L, 40L)

        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv3(x))   # (batchSize, 50L, 33L, 138L)
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool4(x))   # (batchSize, 50L, 11L, 14L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNet_mrf(nn.Module):
    def __init__(self):
        super(EnvNet_mrf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=80, stride=20, padding=39)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=80, stride=4, padding=39)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = F.relu(self.conv1(x))   # (batchSize, 40L, 24001L)
        x = F.relu(self.conv2(x))   # (batchSize, 40L, 24000L)
        x = F.relu(self.pool2(x))   # (batchSize, 40L, 150L)
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv3(x))   # (batchSize, 50L, 33L, 138L)
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool4(x))   # (batchSize, 50L, 11L, 14L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNet_lrf(nn.Module):
    def __init__(self):
        super(EnvNet_lrf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=320, stride=1, padding=159)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=320, stride=1, padding=159)
        self.pool2 = nn.MaxPool1d(kernel_size=160, stride=160)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x = F.relu(self.conv1(x))   # (batchSize, 40L, 24001L)
        x = F.relu(self.conv2(x))   # (batchSize, 40L, 24000L)
        x = F.relu(self.pool2(x))   # (batchSize, 40L, 150L)
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv3(x))   # (batchSize, 50L, 33L, 138L)
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool4(x))   # (batchSize, 50L, 11L, 14L)
        x = x.view(-1, self.num_flat_features(x))   # (batchSize, 7700L)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # (batchSize, 50L)
        return x
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNet3D(nn.Module):
    def __init__(self):
        super(EnvNet3D, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=80, stride=1, padding=39)
        self.conv2_1 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.conv2_2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=80, stride=1, padding=39)

        self.pool2_1 = nn.MaxPool1d(kernel_size=160, stride=160)
        self.pool2_2 = nn.MaxPool1d(kernel_size=160, stride=160)

        self.conv3 = nn.Conv3d(in_channels=1, out_channels=50, kernel_size=(8, 13, 2), stride=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(3, 3, 1))
        self.conv4 = nn.Conv3d(in_channels=50, out_channels=50, kernel_size=(1, 5, 1), stride=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 3, 1))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x1 = F.relu(self.conv1_1(x))   # (batchSize, 40L, 24001L)
        x2 = F.relu(self.conv1_2(x))  # (batchSize, 40L, 24001L)
        x1 = F.relu(self.conv2_1(x1))   # (batchSize, 40L, 24000L)
        x2 = F.relu(self.conv2_2(x2))  # (batchSize, 40L, 24000L)
        x1 = F.relu(self.pool2_1(x1))   # (batchSize, 40L, 150L)
        x2 = F.relu(self.pool2_2(x2))  # (batchSize, 40L, 150L)

        h = torch.stack((x1, x2))
        h = torch.transpose(h, 0, 1)
        h = torch.transpose(h, 1, 2)
        h = torch.transpose(h, 2, 3) # (batchSize, 40L, 150L, D)

        h = torch.unsqueeze(h, 1) #(batchSize, 1L, 40L, 150L, D)

        h = F.relu(self.conv3(h))   # (batchSize, 50L, 33L, 138L, 1L)
        h = F.relu(self.pool3(h)) # (batchSize, 50L, 11L, 46L, 1L)
        h = F.relu(self.conv4(h))
        h = F.relu(self.pool4(h))   # (batchSize, 50L, 11L, 14L)
        h = h.view(-1, self.num_flat_features(h))   # (batchSize, 7700L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = F.relu(self.fc3(h)) # (batchSize, 50L)

        return h
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNetMultiScale(nn.Module):
    def __init__(self):
        super(EnvNetMultiScale, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=32, stride=5, padding=16)
        self.bn1_1 = nn.BatchNorm1d(40)
        self.bn1_2 = nn.BatchNorm1d(40)

        self.conv2_1 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.conv2_2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=32, stride=5, padding=15)
        self.bn2_1 = nn.BatchNorm1d(40)
        self.bn2_2 = nn.BatchNorm1d(40)
        self.pool2_1 = nn.MaxPool1d(kernel_size=160, stride=160)
        self.pool2_2 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        # (50, 11, 14)

        self.fc1 = nn.Linear(8800, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 24002L)
        x1 = self.conv1_1(x)   # (batchSize, 40L, 24001L)
        x2 = self.conv1_2(x)  # (batchSize, 40L, 24001L)
        x1 = self.bn1_1(x1)
        x2 = self.bn1_2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)

        x1 = self.conv2_1(x1)   # (batchSize, 40L, 24000L)
        x2 = self.conv2_2(x2)  # (batchSize, 40L, 24000L)
        x1 = self.bn2_1(x1)
        x2 = self.bn2_2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x1 = self.pool2_1(x1)   # (batchSize, 40L, 150L)
        x2 = self.pool2_2(x2)  # (batchSize, 40L, 150L)

        h = torch.cat((x1, x2), dim=2) #(batchSize, 40L*N, 150L)
        h = torch.unsqueeze(h, 1) #(batchSize, 1L, 40L*N, 150L)

        h = self.conv3(h)   # (bs, 50L, 33L, 288L)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 50L, 11L, 96L)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (256L, 50L, 11L, 11L)

        h = h.view(-1, self.num_flat_features(h))   # (batchSize, 7700L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = self.fc3(h) # (batchSize, 50L)
        return h


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnvNetMultiScale44100(nn.Module):
    def __init__(self):
        super(EnvNetMultiScale44100, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=4)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=32, stride=5, padding=16)
        self.bn1_1 = nn.BatchNorm1d(40)
        self.bn1_2 = nn.BatchNorm1d(40)

        self.conv2_1 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.conv2_2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=32, stride=5, padding=15)
        self.bn2_1 = nn.BatchNorm1d(40)
        self.bn2_2 = nn.BatchNorm1d(40)
        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 5), stride=(2, 5))
        # (50, 12, 16)

        self.fc1 = nn.Linear(9600, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.conv1_1(x)   # (batchSize, 40L, 66151L)
        x2 = self.conv1_2(x)  # (batchSize, 40L, 66151L)
        x1 = self.bn1_1(x1)
        x2 = self.bn1_2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)

        x1 = self.conv2_1(x1)   # (batchSize, 40L, 24000L)
        x2 = self.conv2_2(x2)  # (batchSize, 40L, 24000L)
        x1 = self.bn2_1(x1)
        x2 = self.bn2_2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x1 = self.pool2_1(x1)   # (batchSize, 40L, 441L)
        x2 = self.pool2_2(x2)  # (batchSize, 40L, 441L)

        h = torch.cat((x1, x2), dim=1) #(batchSize, 80L, 441L)

        h = torch.unsqueeze(h, 1) #(batchSize, 1L, 80L, 441L)

        h = self.conv3(h)   # (bs, 50L, 33L, 288L)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 50L, 11L, 96L)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (256L, 50L, 11L, 11L)

        h = h.view(-1, self.num_flat_features(h))   # (batchSize, 7700L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        # h = F.relu(self.fc2(h))
        # h = self.dropout(h)
        # h = self.fc3(h) # (batchSize, 50L)
        return h


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EnvNetMS(nn.Module):
    def __init__(self):
        super(EnvNetMS, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8, stride=1, padding=4)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=32, stride=5, padding=16)
        self.bn1_1 = nn.BatchNorm1d(40)
        self.bn1_2 = nn.BatchNorm1d(40)

        self.conv2_1 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=1, padding=3)
        self.conv2_2 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=32, stride=5, padding=15)
        self.bn2_1 = nn.BatchNorm1d(40)
        self.bn2_2 = nn.BatchNorm1d(40)
        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv3 = nn.Conv2d(in_channels=2, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        # (50, 12, 16)

        self.fc1 = nn.Linear(8800, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.conv1_1(x)   # (batchSize, 40L, 66151L)
        x2 = self.conv1_2(x)  # (batchSize, 40L, 66151L)
        x1 = self.bn1_1(x1)
        x2 = self.bn1_2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)

        x1 = self.conv2_1(x1)   # (batchSize, 40L, 24000L)
        x2 = self.conv2_2(x2)  # (batchSize, 40L, 24000L)
        x1 = self.bn2_1(x1)
        x2 = self.bn2_2(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x1 = self.pool2_1(x1)   # (batchSize, 40L, 441L)
        x2 = self.pool2_2(x2)  # (batchSize, 40L, 441L)

        x1 = torch.unsqueeze(x1, 1)  # (batchSize, 1L, 40L, 441L)
        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 40L, 441L)

        h = torch.cat((x1, x2), dim=1) #(batchSize, 2L, 40L, 441L)

        h = self.conv3(h)   # (bs, 50L, 33L, 288L)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 50L, 11L, 96L)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (256L, 50L, 11L, 11L)

        h = h.view(-1, self.num_flat_features(h))   # (batchSize, 7700L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        # h = F.relu(self.fc2(h))
        # h = self.dropout(h)
        # h = self.fc3(h) # (batchSize, 50L)
        return h


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class EnvNetLogMel(nn.Module):
    def __init__(self):
        super(EnvNetLogMel, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        # (50, 11, 14)

        self.fc1 = nn.Linear(7700, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # input: (batchSize, 40L, 150L)

        # h = torch.unsqueeze(x, 1) #(batchSize, 1L, 40L, 150L)
        # print '1:', x.size()
        h = F.relu(self.conv3(x))   # (batchSize, 50L, 33L, 138L)
        # print '2:', h.size()
        h = F.relu(self.pool3(h)) # (batchSize, 50L, 11L, 46L)
        # print '3:', h.size()
        h = F.relu(self.conv4(h))
        # print '4:', h.size()
        h = F.relu(self.pool4(h))   # (batchSize, 50L, 11L, 14L)
        # print '5:', h.size()
        h = h.view(-1, self.num_flat_features(h))   # (batchSize, 7700L)
        # print '6:', h.size()
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = F.relu(self.fc3(h)) # (batchSize, 50L)
        # exit(0)
        return h
        #  return F.log_softmax(x)


    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LogmelCNN(nn.Module):
    def __init__(self):
        super(LogmelCNN, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=50, kernel_size=(8, 13), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5))
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 5), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))
        # (50, 12, 16)

        self.fc1 = nn.Linear(8800, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv3(x)   # (bs, 50L, 33L, 288L)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 50L, 11L, 96L)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (256L, 50L, 11L, 11L)

        h = h.view(-1, self.num_flat_features(h))   # (batchSize, 7700L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

    def num_flat_features(self, x):
        # (32L, 50L, 11L, 14L), 32 is batch_size
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class M9Logmel(nn.Module):
    def __init__(self):
        super(M9Logmel, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (bs, 2, 64, 441)
        h = self.conv3(x)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 64L, 16L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (bs, 64L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))   # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9Logmel_norm(nn.Module):
    def __init__(self):
        super(M9Logmel_norm, self).__init__()
        self.bn2 = nn.BatchNorm1d(2)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (bs, 2, 64, 441)
        # h = self.bn2(h)

        h = self.conv3(x)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 64L, 16L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (bs, 64L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))   # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9(nn.Module):
    def __init__(self):
        super(M9, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.bn1_3 = nn.BatchNorm1d(64)

        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.bn2_3 = nn.BatchNorm1d(64)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 64L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 64L, 441L)

        h = torch.cat((x1, x2, x3), dim=1) #(batchSize, 3L, 64L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_srf(nn.Module):
    def __init__(self):
        super(M9_srf, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn1_1 = nn.BatchNorm1d(64)

        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))

        x1 = self.pool2_1(x1)

        x1 = torch.unsqueeze(x1, 1)

        # h = torch.cat((x1, x2, x3), dim=1) #(batchSize, 3L, 64L, 441L)

        h = self.conv3(x1)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_mrf(nn.Module):
    def __init__(self):
        super(M9_mrf, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2_2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x2)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_mrf_1Conv(nn.Module):
    def __init__(self):
        super(M9_mrf_1Conv, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.pool1_2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        # x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool1_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x2)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_mrf_3Conv(nn.Module):
    def __init__(self):
        super(M9_mrf_3Conv, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(64)

        self.conv3_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn3_2 = nn.BatchNorm1d(64)

        self.pool3_2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.relu(self.bn3_2(self.conv3_2(x2)))

        x2 = self.pool3_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x2)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_lrf(nn.Module):
    def __init__(self):
        super(M9_lrf, self).__init__()
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101, stride=1, padding=50)
        self.bn1_3 = nn.BatchNorm1d(64)

        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.pool2_3 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x3 = self.pool2_3(x3)  # (batchSize, 64L, 441L)

        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x3)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_mrf_logmel(nn.Module):
    def __init__(self):
        super(M9_mrf_logmel, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2_2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.bn3_0 = nn.BatchNorm2d(3)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x, logmel):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = self.pool2_2(x2)  # (batchSize, 64L, 441L)
        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        # x2 = torch.cat((x2, logmel), dim=1)  # (batchSize, 3L, 64L, 441L)

        h = torch.cat((x2, logmel), dim=1)

        # h = self.bn3_0(x2)

        # exit(0)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h



class LateFusion(nn.Module):
    def __init__(self):
        super(LateFusion, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2_2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()


        self.conv3_lm = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_lm = nn.BatchNorm2d(64)
        self.pool3_lm = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))
        self.conv4_lm = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4_lm = nn.BatchNorm2d(128)
        self.pool4_lm = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv5_lm = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5_lm = nn.BatchNorm2d(256)
        self.pool5_lm = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1_lm = nn.Linear(5120, 4096)
        self.fc2_lm = nn.Linear(4096, 4096)

        self.fc3 = nn.Linear(8192, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x, logmel):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h1 = self.conv3(x2)
        h1 = self.bn3(h1)
        h1 = self.relu(h1)
        h1 = self.pool3(h1)  # (bs, 64L, 16L, 44L)

        h1 = self.conv4(h1)
        h1 = self.bn4(h1)
        h1 = self.relu(h1)
        h1 = self.pool4(h1)  # (bs, 128L, 8L, 10L)

        h1 = self.conv5(h1)
        h1 = self.bn5(h1)
        h1 = self.relu(h1)
        h1 = self.pool5(h1)  # (bs, 256L, 4L, 5L)

        h1 = h1.view(-1, num_flat_features(h1))  # (batchSize, 6600L)
        h1 = F.relu(self.fc1(h1))
        h1 = self.dropout(h1)
        h1 = self.fc2(h1)


        h2 = self.conv3_lm(logmel)
        h2 = self.bn3(h2)
        h2 = self.relu(h2)
        h2 = self.pool3(h2) # (bs, 64L, 16L, 40L)

        h2 = self.conv4(h2)
        h2 = self.bn4(h2)
        h2 = self.relu(h2)
        h2 = self.pool4(h2)   # (bs, 64L, 8L, 10L)

        h2 = self.conv5(h2)
        h2 = self.bn5(h2)
        h2 = self.relu(h2)
        h2 = self.pool5(h2)  # (bs, 64L, 4L, 5L)

        h2 = h2.view(-1, num_flat_features(h2))   # (batchSize, 6600L)
        h2 = F.relu(self.fc1(h2))
        h2 = self.dropout(h2)
        h2 = self.fc2(h2)

        h = torch.cat((h1, h2), dim=1)
        h = self.fc3(h)
        return h


class DNN3Layer(nn.Module):
    def __init__(self):
        super(DNN3Layer, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 50)
        # self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = x.view(-1, num_flat_features(x))  # (batchSize, 6600L)
        h = self.dropout(F.relu(self.fc1(h)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.fc3(h)
        return h


class ELSTM(nn.Module):
    """                                                                                                                                       
    LSTM                                                                                                                                      
    """
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(ELSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, n_class)
    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 0, 2) # (150L, batch, 40L)
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1), self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1), self.hidden_dim)).cuda()
        # **input** (seq_len, batch, input_size)
        out, _ = self.lstm(x)
        # print('1:', out.size()) # (150, batch, 2048)
        # 用最后一行作为输出
        # ** output ** (seq_len, batch, hidden_size * num_directions)
        out = out[-1, :, :] # (batch, 2048)
        # print out.size()
        # 用所有输出中最大值作为输出
        # out = torch.max(out, dim=0)[0] # (batch, 2048)
        # print out.size()
        out = self.dropout(out)
        out = self.fc(out) # (batch, 50)

        return out


class WaveLSTM(nn.Module):
    """                                                                                                                                       
    LSTM                                                                                                                                      
    """
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(WaveLSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=8, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=160, stride=160)
        self.conv2 = nn.Conv1d(in_channels=150, out_channels=150, kernel_size=8, stride=1, padding=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)


        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, n_class)
    def forward(self, x):
        x = F.relu(self.conv1(x))   # (batchSize, 40L, 24001L)
        x = F.relu(self.pool1(x))   # (batchSize, 40L, 150L)
        x = torch.transpose(x, 1, 2)   # (batchSize, 150L, 80L)
        x = F.relu(self.conv2(x))   # (batchSize, 150L, 81L)
        x = F.relu(self.pool2(x))   # (batchSize, 150L, 40L)
        # x = torch.squeeze(x, dim=1)
        x = torch.transpose(x, 0, 1) # (150L, batch, 40L)
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1), self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1), self.hidden_dim)).cuda()
        # **input** (seq_len, batch, input_size)
        out, _ = self.lstm(x)
        # print('1:', out.size()) # (150, batch, 2048)
        # 用最后一行作为输出
        # ** output ** (seq_len, batch, hidden_size * num_directions)
        # out = out[-1, :, :] # (batch, 2048)
        # print out.size()
        # 用所有输出中最大值作为输出
        out = torch.max(out, dim=0)[0] # (batch, 2048)
        # print out.size()
        out = self.dropout(out)
        out = self.fc(out) # (batch, 50)

        return out


class M9_mrf_22050(nn.Module):
    def __init__(self):
        super(M9_mrf_22050, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2_2 = nn.MaxPool1d(kernel_size=75, stride=75)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x2)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_mrf_22050_avgpool(nn.Module):
    def __init__(self):
        super(M9_mrf_22050_avgpool, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2_2 = nn.MaxPool1d(kernel_size=75, stride=75)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 11), stride=(4, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.avgpool = nn.AvgPool2d((4, 5), stride=1)

        self.fc1 = nn.Linear(256, 1000)
        self.fc2 = nn.Linear(1000, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x2)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 4L, 5L)

        h = self.avgpool(h)
        h = h.view(h.size(0), -1)  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_v2(nn.Module):
    def __init__(self):
        super(M9_v2, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 11), stride=(2, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 64L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 64L, 441L)

        h = torch.cat((x1, x2, x3), dim=1) #(batchSize, 3L, 64L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_srf_v2(nn.Module):
    def __init__(self):
        super(M9_srf_v2, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.bn1_1 = nn.BatchNorm1d(96)

        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.bn2_1 = nn.BatchNorm1d(96)
        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(6, 11), stride=(6, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))

        x1 = self.pool2_1(x1)

        x1 = torch.unsqueeze(x1, 1)

        # h = torch.cat((x1, x2, x3), dim=1) #(batchSize, 3L, 64L, 441L)

        h = self.conv3(x1)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_mrf_v2(nn.Module):
    def __init__(self):
        super(M9_mrf_v2, self).__init__()
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=51, stride=1, padding=25)
        self.bn1_2 = nn.BatchNorm1d(96)

        self.conv2_2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.bn2_2 = nn.BatchNorm1d(96)
        self.pool2_2 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(6, 11), stride=(6, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x2)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_lrf_v2(nn.Module):
    def __init__(self):
        super(M9_lrf_v2, self).__init__()
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=101, stride=1, padding=50)
        self.bn1_3 = nn.BatchNorm1d(96)

        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.bn2_3 = nn.BatchNorm1d(96)
        self.pool2_3 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(6, 11), stride=(6, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x3 = self.pool2_3(x3)  # (batchSize, 64L, 441L)

        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 64L, 441L)

        h = self.conv3(x3)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 16L, 44L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 8L, 11L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_v3(nn.Module):
    def __init__(self):
        super(M9_v3, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        h = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9Logmel_v3(nn.Module):
    def __init__(self):
        super(M9Logmel_v3, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: #(batchSize, 1L, 96L, 441L)

        h = self.conv3(x)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 128L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 256L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  # (bs, 256L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class M9_mrf_fixed_logmel(nn.Module):
    def __init__(self, phase):
        super(M9_mrf_fixed_logmel, self).__init__()
        self.phase = phase
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=51, stride=5, padding=25)

        self.bn1_2 = nn.BatchNorm1d(96)

        self.conv2_2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_2 = nn.BatchNorm1d(96)

        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)

        self.conv3 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 10)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def changePhase(self, newphase):
        self.phase = newphase

    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))

        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))

        x2 = self.pool2_2(x2)  # (batchSize, 96L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 96L, 441L)

        if self.phase == 1:
            h = torch.cat((x2, x2), dim=1)  # (batchSize, 2L, 96L, 441L)
        elif self.phase == 2:
            h = torch.cat((x2, feats), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 60L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_srf_fixed_logmel(nn.Module):
    def __init__(self, phase):
        super(M9_srf_fixed_logmel, self).__init__()
        self.phase = phase
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn1_1 = nn.BatchNorm1d(96)

        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(96)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)

        self.conv3 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 10)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def changePhase(self, newphase):
        self.phase = newphase

    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))

        x1 = self.pool2_1(x1)

        x1 = torch.unsqueeze(x1, 1)

        if self.phase == 1:
            h = torch.cat((x1, x1), dim=1)  # (batchSize, 2L, 96L, 441L)
        elif self.phase == 2:
            h = torch.cat((x1, feats), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 60L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_lrf_fixed_logmel(nn.Module):
    def __init__(self, phase):
        super(M9_lrf_fixed_logmel, self).__init__()
        self.phase = phase
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=101, stride=10, padding=50)

        self.bn1_3 = nn.BatchNorm1d(96)

        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_3 = nn.BatchNorm1d(96)

        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 10)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def changePhase(self, newphase):
        self.phase = newphase

    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        if self.phase == 1:
            h = torch.cat((x3, x3), dim=1)  # (batchSize, 2L, 96L, 441L)
        elif self.phase == 2:
            h = torch.cat((x3, feats), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 60L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_fixed_logmel(nn.Module):
    def __init__(self, phase):
        super(M9_fixed_logmel, self).__init__()
        self.phase = phase
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 10)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def changePhase(self, newphase):
        self.phase = newphase

    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        if self.phase == 1:
            h = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)
            h = torch.cat((h, h), dim=1)  # (batchSize, 2L, 96L, 441L)
        elif self.phase == 2:
            h = torch.cat((x1, x2, x3), dim=2)  # (batchSize, 1L, 96L, 441L)
            h = torch.cat((h, feats), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  # (bs, 64L, 32L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  # (bs, 60L, 16L, 20L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 60L, 8L, 10L)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))  # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class M9_mfcc(nn.Module):
    def __init__(self):
        super(M9_mfcc, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 11), stride=(2, 11))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(5120, 4096)
        self.fc2 = nn.Linear(4096, 50)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input: (bs, 2, 64, 441)
        h = self.conv3(x)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h) # (bs, 64L, 16L, 40L)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)   # (bs, 64L, 8L, 10L)

        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)  # (bs, 64L, 4L, 5L)

        h = h.view(-1, num_flat_features(h))   # (batchSize, 6600L)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet18_mrf(nn.Module):

    def __init__(self, block, layers, phase, num_classes=50):
        self.inplanes = 64
        super(ResNet18_mrf, self).__init__()

        self.phase = phase
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(96)
        self.bn1_2 = nn.BatchNorm1d(96)
        self.bn1_3 = nn.BatchNorm1d(96)

        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(96)
        self.bn2_2 = nn.BatchNorm1d(96)
        self.bn2_3 = nn.BatchNorm1d(96)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)



        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d((3, 14))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def changePhase(self, newphase):
        self.phase = newphase


    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        # x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        # x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        # x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        # x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        # x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        # x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        # x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        # x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        # if self.phase == 1:
        #     h = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, h), dim=1)  # (batchSize, 2L, 96L, 441L)
        # elif self.phase == 2:
        #     h = torch.cat((x1, x2, x3), dim=2)  # (batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, feats), dim=1)  # (batchSize, 2L, 96L, 441L)
        h = torch.cat((x2, x2), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)  #[32, 2048, 3, 14]
        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)

        return h
class ResNet18_lrf(nn.Module):

    def __init__(self, block, layers, phase, num_classes=50):
        self.inplanes = 64
        super(ResNet18_lrf, self).__init__()

        self.phase = phase
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(96)
        self.bn1_2 = nn.BatchNorm1d(96)
        self.bn1_3 = nn.BatchNorm1d(96)

        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(96)
        self.bn2_2 = nn.BatchNorm1d(96)
        self.bn2_3 = nn.BatchNorm1d(96)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)



        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d((3, 14))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def changePhase(self, newphase):
        self.phase = newphase


    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        # x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        # x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        # x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        # x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        # x1 = self.pool2_1(x1)
        # x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        # x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        # if self.phase == 1:
        #     h = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, h), dim=1)  # (batchSize, 2L, 96L, 441L)
        # elif self.phase == 2:
        #     h = torch.cat((x1, x2, x3), dim=2)  # (batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, feats), dim=1)  # (batchSize, 2L, 96L, 441L)
        h = torch.cat((x3, x3), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)  #[32, 2048, 3, 14]
        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)

        return h

class VGG(nn.Module):

    def __init__(self, features, phase, num_classes=50):
        super(VGG, self).__init__()
        self.phase = phase
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(96)
        self.bn1_2 = nn.BatchNorm1d(96)
        self.bn1_3 = nn.BatchNorm1d(96)

        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(96)
        self.bn2_2 = nn.BatchNorm1d(96)
        self.bn2_3 = nn.BatchNorm1d(96)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.relu = nn.ReLU(inplace=True)

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 13, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        # x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        # x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        # x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        # x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        # x2 = self.pool2_2(x2)
        # x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        # x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        # if self.phase == 1:
        #     h = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, h), dim=1)  # (batchSize, 2L, 96L, 441L)
        # elif self.phase == 2:
        #     h = torch.cat((x1, x2, x3), dim=2)  # (batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, feats), dim=1)  # (batchSize, 2L, 96L, 441L)
        h = torch.cat((x1, x1), dim=1)  # (batchSize, 2L, 96L, 441L)

        h = self.features(h) # [bs, 512, 3, 13]
        h = h.view(h.size(0), -1)
        h = self.classifier(h)
        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 2
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_bn_srf(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model

class AlexNet_srf(nn.Module):

    def __init__(self, phase, num_classes=50):
        super(AlexNet_srf, self).__init__()
        self.phase = phase
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(96)
        self.bn1_2 = nn.BatchNorm1d(96)
        self.bn1_3 = nn.BatchNorm1d(96)

        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(96)
        self.bn2_2 = nn.BatchNorm1d(96)
        self.bn2_3 = nn.BatchNorm1d(96)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.relu = nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 12, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, feats=None):
        # input: (batchSize, 1L, 66150L)
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        # x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        # x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        # x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        # x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        # x2 = self.pool2_2(x2)
        # x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        # x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        # if self.phase == 1:
        #     h = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, h), dim=1)  # (batchSize, 2L, 96L, 441L)
        # elif self.phase == 2:
        #     h = torch.cat((x1, x2, x3), dim=2)  # (batchSize, 1L, 96L, 441L)
        #     h = torch.cat((h, feats), dim=1)  # (batchSize, 2L, 96L, 441L)
        h = torch.cat((x1, x1), dim=1)

        h = self.features(h)
        h = h.view(h.size(0), 256 * 2 * 12)
        h = self.classifier(h)
        return h

if __name__ == "__main__":
    print('network.py')
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



class WaveMsNet(nn.Module):
    def __init__(self):
        super(WaveMsNet, self).__init__()
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


class WaveMsNet_Logmel(nn.Module):
    """
    Backend of the Network. It will be trained by Log-Mel feature.
    """
    def __init__(self):
        super(WaveMsNet_Logmel, self).__init__()

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



class WaveMsNet_srf_fixed_logmel(nn.Module):
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


class WaveMsNet_mrf_fixed_logmel(nn.Module):
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

class WaveMsNet_lrf_fixed_logmel(nn.Module):
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

class WaveMsNet_fixed_logmel(nn.Module):
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

if __name__ == "__main__":
    print('network.py')
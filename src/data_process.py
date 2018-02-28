# -*- coding: utf-8 -*-
"""
optimize I/O with packet squeence input.
__Author__ = Zhu.bq

"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
from util import *
import random


class WaveformDataset(Dataset):
# }}}
    def __init__(self, pkl_file, window_size=66150, fs=44100, train_slices=1, add_logmel=False, transform=None):
        """
        :param pkl_file: 
        :param window_size: 
        :param num_slices: slices number of one record divide into.
        :param transform: 
        """

        self.transform = transform
        self.sampleSet = load_data(pkl_file)
        self.window_size = window_size
        self.fs = fs
        self.train_slices = train_slices
        self.add_logmel = add_logmel

    def __len__(self):
        return len(self.sampleSet)*self.train_slices

    def __getitem__(self, index):

        # key = self.sampleSet[index//self.num_slices]['key']

        data = self.sampleSet[index // self.train_slices]['data']
        feat = self.random_selection(data)
        label = self.sampleSet[index // self.train_slices]['label']

        if self.add_logmel == False:
            feat = feat[np.newaxis, :]
            sample = {'feat': feat, 'label': label}
        else:
            melspec = librosa.feature.melspectrogram(feat, self.fs, n_fft=2048, hop_length=150//(self.fs//44100), n_mels=96)  # (40, 442)
            logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)

            # delta = librosa.feature.delta(logmel, width=3) # (40, 441)

            # feat = np.stack((logmel, delta))

            # sample = {'feat': feat, 'label': label}
            feat = logmel[np.newaxis, :, :]
            sample = {'feat': feat, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def random_selection(self, wave):
        wl = len(wave) - self.window_size
        maxamp = 0.

        # win_start = random.randint(0, wl)
        # win_data = wave[win_start: win_start + self.window_size]

        while maxamp < 0.005:
            win_start = random.randint(0, wl)
            win_data = wave[win_start: win_start + self.window_size]
            maxamp = np.max(np.abs(win_data))
        return win_data

#}}}


class LogMelDataset(Dataset):
# }}}
    def __init__(self, pkl_file, transform=None):
        """
        :param pkl_file: 
        :param window_size: 
        :param num_slices: slices number of one record divide into.
        :param transform: 
        """

        self.transform = transform
        self.sampleSet = load_data(pkl_file)

    def __len__(self):
        return len(self.sampleSet)

    def __getitem__(self, index):

        # key = self.sampleSet[index//self.num_slices]['key']

        feat = self.sampleSet[index]['logmel']
        feat = feat[np.newaxis, :]
        # print feat.shape

        label = self.sampleSet[index]['label']
        sample = {'feat': feat, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

#}}}

class FusionDataset(Dataset):
# }}}
    def __init__(self, pkl_file, window_size=66150, train_slices=1, transform=None):
        """
        :param pkl_file: 
        :param window_size: 
        :param num_slices: slices number of one record divide into.
        :param transform: 
        """

        self.transform = transform
        self.sampleSet = load_data(pkl_file)
        self.window_size = window_size
        self.train_slices = train_slices

    def __len__(self):
        return len(self.sampleSet)*self.train_slices

    def __getitem__(self, index):

        # key = self.sampleSet[index//self.num_slices]['key']

        data = self.sampleSet[index // self.train_slices]['data']
        wave = self.random_selection(data)
        label = self.sampleSet[index // self.train_slices]['label']

        melspec = librosa.feature.melspectrogram(wave, 44100, n_fft=2048, hop_length=150, n_mels=96)  # (40, 442)
        logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)
        # mfcc = librosa.feature.mfcc(wave, n_fft=2048, hop_length=150, sr=44100, n_mfcc=32)
        # mfcc = mfcc[:, :441]
        # delta = librosa.feature.delta(logmel) # (40, 441)
        # feat = np.stack((logmel, mfcc, delta))
        feat = logmel[np.newaxis, :, :]

        wave = wave[np.newaxis, :]
        sample = {'wave': wave, 'feat': feat, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def random_selection(self, wave):
        wl = len(wave) - self.window_size
        maxamp = 0.

        # win_start = random.randint(0, wl)
        # win_data = wave[win_start: win_start + self.window_size]

        while maxamp < 0.005:
            win_start = random.randint(0, wl)
            win_data = wave[win_start: win_start + self.window_size]
            maxamp = np.max(np.abs(win_data))
        return win_data


class MFCCDataset(Dataset):
# }}}
    def __init__(self, pkl_file, window_size=66150, fs=44100, train_slices=1, add_logmel=False, transform=None):
        """
        :param pkl_file: 
        :param window_size: 
        :param num_slices: slices number of one record divide into.
        :param transform: 
        """
        self.transform = transform
        self.sampleSet = load_data(pkl_file)
        self.window_size = window_size
        self.fs = fs
        self.train_slices = train_slices
        self.add_logmel = add_logmel

    def __len__(self):
        return len(self.sampleSet)*self.train_slices

    def __getitem__(self, index):

        # key = self.sampleSet[index//self.num_slices]['key']

        data = self.sampleSet[index // self.train_slices]['data']
        feat = self.random_selection(data)
        label = self.sampleSet[index // self.train_slices]['label']


        mfcc = librosa.feature.mfcc(y=feat, n_fft=2048, hop_length=150, sr=44100, n_mfcc=32)
        mfcc = mfcc[:, :441]
            # melspec = librosa.feature.melspectrogram(feat, self.fs, n_fft=2048, hop_length=150/(self.fs//44100), n_mels=64)  # (40, 442)
            # logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)

            # delta = librosa.feature.delta(logmel, width=3) # (40, 441)

            # feat = np.stack((logmel, delta))

            # sample = {'feat': feat, 'label': label}
        feat = mfcc[np.newaxis, :, :]
        sample = {'feat': feat, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def random_selection(self, wave):
        wl = len(wave) - self.window_size
        maxamp = 0.

        # win_start = random.randint(0, wl)
        # win_data = wave[win_start: win_start + self.window_size]

        while maxamp < 0.005:
            win_start = random.randint(0, wl)
            win_data = wave[win_start: win_start + self.window_size]
            maxamp = np.max(np.abs(win_data))
        return win_data


class ToTensor(object):
    """#{{{
    convert ndarrays in sample to Tensors.
„
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)

    """

    def __call__(self, sample):
        feat, label = sample['feat'], sample['label']

        feat = torch.from_numpy(feat).type(torch.FloatTensor)
        label = torch.LongTensor([label])
        #  feat = torch.rand(1, 33, 34).type(torch.FloatTensor)
        #  label = torch.LongTensor([1])
        return feat, label#}}}


class ToTensor2(object):
    """#{{{
    convert ndarrays in sample to Tensors.

    """
    def __call__(self, sample):
        wave, feat, label = sample['wave'], sample['feat'], sample['label']

        wave = torch.from_numpy(wave).type(torch.FloatTensor)
        feat = torch.from_numpy(feat).type(torch.FloatTensor)
        label = torch.LongTensor([label])
        #  feat = torch.rand(1, 33, 34).type(torch.FloatTensor)
        #  label = torch.LongTensor([1])
        return wave, feat, label#}}}


if __name__ == "__main__":

    waveformDataset = WaveformDataset('../data_wave/fold0_valid.cPickle', add_logmel=True, transform=ToTensor())
    dataloader = DataLoader(waveformDataset, batch_size=5, shuffle=False, num_workers=1)

    # metaDataset = MetaDataset('probilities.0.txt', transform=ToTensor())
    # dataloader = DataLoader(metaDataset, batch_size=5, shuffle=False, num_workers=1)

    print(len(dataloader))
    print(len(dataloader.dataset))
    # for idx, (feat, label) in enumerate(dataloader):
    #     # print(idx)
    #     print(feat)  # (bs, 40L, 150L)
    #     print(label)
    #     # print(type(sample_batched[0].size()))
    #     if idx == 2:
    #         break

    # sampleSet = load_data('../data_wave_44100/fold0_valid.cPickle')
    # data = sampleSet[0]['data']
    # feat = data[:66150]
    # melspec = librosa.feature.melspectrogram(feat, 44100, n_fft=2048, hop_length=150, n_mels=40)  # (40, 441)
    # print melspec.shape
    # logmel = librosa.logamplitude(melspec)  # (40, 151)
    # print logmel.shape
    # print logmel
    # label = sampleSet[0]['label']

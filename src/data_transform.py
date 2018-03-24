
from util import *
import os
import random
import numpy as np


def get_fold_wavelist(fold_wavelist):
    f = open(fold_wavelist, 'r')
    waveList = []
    for line in f.readlines():
        filePath = os.path.join('../ESC-10', line.split('\t')[0])
        waveList.append(filePath)
    return waveList


def get_pkl(fs):
    """
    :return: [{'key', 'data', 'label'}, {}, {}... {}]
    """

    wav_len = fs * 5

    for fold_num in range(5):

        print('get pkl on fold ', str(fold_num))
        trainWaveName = '../cross_folds/evaluate-setup-ESC10/fold' + str(fold_num) + '_train.txt'
        # validWaveName = '../evaluate-setup/fold' + str(fold_num) + '_valid.txt'
        testWaveName = '../cross_folds/evaluate-setup-ESC10/fold' + str(fold_num) + '_test.txt'
        trainWaveList = get_fold_wavelist(trainWaveName)
        # validWaveList = get_fold_wavelist(validWaveName)
        testWaveList = get_fold_wavelist(testWaveName)

        # waveLists = [trainWaveList, validWaveList, testWaveList]
        waveLists = [trainWaveList, testWaveList]

        data = []
        item = {}

        for idx, wavelist in enumerate(waveLists):
            for f in wavelist:
                cls_id = f.split('/')[2].split(' ')[0]

                # cls_id = num_to_id_ESC50(int(cls_id))
                cls_id = num_to_id_ESC10(cls_id)

                audio_data, _ = librosa.load(f, fs)

                # make each audio exactly 5s.
                audio_data = audio_data[: wav_len]

                # audio_data = audio_data * 1.0 / np.max(abs(audio_data))

                if len(audio_data) < wav_len:
                    audio_data = np.r_[audio_data, np.zeros(wav_len - len(audio_data))]

                item['label'] = int(cls_id)
                item['key'] = f.split('/')[-1].split('.')[0]
                item['data'] = audio_data

                data.append(item)
                item = {}

            if idx == 0:
                random.shuffle(data)
                save_data('../data_wave_ESC10_'+ str(fs) +'/fold'+str(fold_num)+'_train.cPickle', data)
            # elif idx == 1:
            #     save_data('../data_wave_'+ str(fs) +'/fold' + str(fold_num) + '_valid.cPickle', data)
            elif idx == 1:
                save_data('../data_wave_ESC10_'+ str(fs) +'/fold' + str(fold_num) + '_test.cPickle', data)

            data=[]


def get_spec(pkl):
    """
    :return: [{'key', 'data', 'label'}, {}, {}... {}]
    """

    win_size = 66150
    stride = int(44100 * 0.2)

    for fold_num in range(5):
        trainPkl = '../data_wave_ESC10_44100/fold' + str(fold_num) + '_train.cPickle'
        validPkl = '../data_wave_ESC10_44100/fold' + str(fold_num) + '_valid.cPickle'
    sampleSet = load_data(pkl)

    segs = []

    for item in sampleSet:
        print(item['label'], item['key'])
        record_data = item['data']

        for j in range(0, len(record_data) - win_size + 1, stride):

            seg = {}
            win_data = record_data[j: j+win_size]
            # Continue if cropped region is silent

            maxamp = np.max(np.abs(win_data))
            if maxamp < 0.005:
                continue
            melspec = librosa.feature.melspectrogram(win_data, 44100, n_fft=2048, hop_length=150, n_mels=64)  # (40, 442)
            logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)
            delta = librosa.feature.delta(logmel)

            feat = np.stack((logmel, delta))

            seg['label'] = item['label']
            seg['key'] = item['key']
            seg['data'] = feat

            segs.append(seg)

    # save_data('../segments_logmel/fold' + str(fold_num) + '_train.cPickle', segs)
    save_data('../segments_logmel/fold0_valid.cPickle', segs)
    print(segs)
    print(len(segs))


if __name__ == '__main__':


    get_pkl(fs=44100)
    data = load_data('../data_wave_ESC10_44100/fold0_train.cPickle')
    print("data num: ", len(data))
    print(data[0])

    # get_spec('../data_wave_44100/fold0_valid.cPickle')

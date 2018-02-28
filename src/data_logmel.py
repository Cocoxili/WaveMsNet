# coding=utf-8

from util import *
import os
import random
from data_process import *
import sys

def get_logmel_feature(root_path):
    """
    提取数据集的logmel特征

    """
    data = []
    item = {}

    sr = 44100
    hl = int(sr*0.010)

    count = 0

    for root, dirs, files in os.walk(root_path):

        for f in files:
            count += 1
            print('{}/{}\r'.format(count, 2000)),
            sys.stdout.flush()

            file_base, file_extension = os.path.splitext(f)
            if file_extension == '.ogg':
                cls_id = root.split('/')[-1].split(' ')[0]
                # print cls_id
                cls_id = num_to_id_ESC50(int(cls_id))
                # print cls_id
                filepath = os.path.join(root, f)
                y, sr = librosa.load(filepath, sr=sr)

                melspec = librosa.feature.melspectrogram(y, 44100, n_fft=2048, hop_length=hl, n_mels=40)
                logmel = librosa.logamplitude(melspec)
                logmel = logmel[:, :500]
                if logmel.shape[1] < 500:
                    print file_base
                    p = 500 - logmel.shape[1]
                    padding = np.zeros((40, p))
                    logmel = np.concatenate((logmel, padding), axis=1)

                if logmel.shape != (40, 500):
                    print('logmel.shape error')
                    exit(0)

                item['label'] = int(cls_id)
                item['key'] = file_base
                item['logmel'] = logmel

                data.append(item)
                item = {}

    random.shuffle(data)
    save_data('../logmel/logmel.cPickle', data)
    print('Feature extraction finished.')


def seg_feature(logmel_path, seg_train_path, seg_test_path):
    """
    将每个record的logmel特征分割为5分

    """
    data = load_data(logmel_path)
    cut_point = int(len(data)*0.8)
    print 'cut_point/len: ', cut_point,'/',len(data)

    train_data = data[:cut_point]
    test_data = data[cut_point:]

    seg_train = []
    seg_test = []
    seg_item = {}

    for item in train_data:
        for i in range(5):
            seg_item['label'] = item['label']
            seg_item['logmel'] = item['logmel'][:, i*87:i*87+150]
            seg_train.append(seg_item)
            seg_item = {}

    print('\nlen of seg_train: ', len(seg_train))
    save_data(seg_train_path, seg_train)

    for item in test_data:
        for i in range(5):
            seg_item['label'] = item['label']
            seg_item['logmel'] = item['logmel'][:, i*87:i*87+150]
            seg_test.append(seg_item)
            seg_item = {}

    print('\nlen of seg_train: ', len(seg_test))
    save_data(seg_test_path, seg_test)

    print('Feature segmentation finished.')


if __name__ == '__main__':
    dataset = 'ESC-50'

    # get_logmel_feature('/home/zbq/work/envNet/' + dataset)
    seg_feature('../logmel/logmel.cPickle', '../segments_logmel/seg.train.cPickle', '../segments_logmel/seg.test.cPickle')
    seg = load_data('../segments_logmel/seg.test.cPickle')
    print len(seg)
    print seg[0]['logmel'].shape
    print seg[0]['label']
""" usage:
    python main.py
    python main.py --network=dnn --mode=test --model='../model/dnn_mix.pkl'

"""
import argparse

from network import *
from data_process import *
import os

# Training settings
parser = argparse.ArgumentParser(description='pytorch model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=18, metavar='N',
                            help='input batch size for testing (default: 5)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                            help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
parser.add_argument('--gpu', type=list, default=[4,5,6,7],
                            help='gpu device number')
parser.add_argument('--seed', type=int, default=777, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                            help='how many batches to wait before logging training status')
parser.add_argument('--model_save_interval', type=int, default=50, metavar='N',
                            help='how many epochs to wait before saving the model.')
parser.add_argument('--network', type=str, default='M9_mrf_logmel',
                            help='EnvNet or EnvNet_mrf or EnvNet_lrf or EnvNet3D or EnvNetMultiScale')
parser.add_argument('--mode', type=str, default='train',
                            help='train or test')
parser.add_argument('--model', type=str, default='../model/EnvNet_v1_fold0_v2_epoch120.pkl',
                            help='trained model path')
parser.add_argument('--train_slices', type=int, default=1,
                            help='slices number of one record divide into.')
parser.add_argument('--test_slices_interval', type=int, default=0.2,
                            help='slices number of one record divide into.')
parser.add_argument('--fs', type=int, default=44100)


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#  torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #  torch.cuda.set_device(2)


def train(model, optimizer, train_loader, epoch):
#{{{
    model.train()
    #  start = time.time()
    for idx, (data, label) in enumerate(train_loader):

        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        optimizer.zero_grad()

        # print data.size() # (bs, 256L, 4L, 5L)

        output = model(data) # (batch, 50L)
        # print(label)
        # print output.size()
        loss = F.cross_entropy(output, label)
        #  loss = F.nll_loss(output, label)

        loss.backward()

        optimizer.step()


        if idx % args.log_interval == 0:

            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct = pred.eq(label.data.view_as(pred)).sum()
            acc = 100.0 * correct / args.batch_size
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'lr:{} Loss: {:.6f} '
                  'TrainAcc: {:.2f}'.format(
                    epoch, idx * len(data), len(train_loader.dataset),
                    100. * idx / len(train_loader),
                    optimizer.param_groups[0]['lr'], loss.data[0], acc))
            #  elapse = time.time() - start
            #  start = time.time()
            #  print('time/%dbatch: %fs' %(args.log_interval, elapse))

    # save model
    #  if args.network == 'densenet':
        #  torch.save(model, 'model/densenet.pkl')
        #  print('model has been saved as: model/densenet.pkl')
    #  elif args.network == 'dnn':
        #  torch.save(model, 'model/dnn.pkl')
        #  print('model has been saved as: model/dnn.pkl')
#}}}


def test(modelWave, modelLogmel, modelDNN, test_pkl):
#{{{
    modelWave.eval()
    modelLogmel.eval()
    modelDNN.eval()

    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    win_size = 66150
    stride = int(44100 * args.test_slices_interval)
    sampleSet = load_data(test_pkl)

    for item in sampleSet:
        label = item['label']
        record_data = item['data']
        wins_data = []
        feats = []
        for j in range(0, len(record_data) - win_size + 1, stride):

            win_data = record_data[j: j+win_size]
            # Continue if cropped region is silent

            maxamp = np.max(np.abs(win_data))
            if maxamp < 0.005:
                continue
            melspec = librosa.feature.melspectrogram(win_data, 44100, n_fft=2048, hop_length=150, n_mels=64)  # (40, 442)
            logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)
            delta = librosa.feature.delta(logmel)

            feat = np.stack((logmel, delta))

            wins_data.append(win_data)
            feats.append(feat)

        if len(wins_data) == 0:
            print item['key']

        wins_data = np.array(wins_data)
        wins_data = wins_data[:, np.newaxis, :]
        feats = np.array(feats)

        # print wins_data.shape

        wave = torch.from_numpy(wins_data).type(torch.FloatTensor) # (N, 1L, 24002L)
        logmel_delta = torch.from_numpy(feats).type(torch.FloatTensor)
        label = torch.LongTensor([label])

        if args.cuda:
            wave, logmel_delta, label = wave.cuda(), logmel_delta.cuda(), label.cuda()
        wave, logmel_delta, label = Variable(wave, volatile=True), Variable(logmel_delta, volatile=True), Variable(label)


        x2 = modelWave.relu(modelWave.bn1_2(modelWave.conv1_2(wave)))
        x2 = modelWave.relu(modelWave.bn2_2(modelWave.conv2_2(x2)))
        x2 = modelWave.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = modelWave.conv3(x2)
        h = modelWave.bn3(h)
        h = modelWave.relu(h)
        h = modelWave.pool3(h)  # (bs, 64L, 16L, 44L)

        h = modelWave.conv4(h)
        h = modelWave.bn4(h)
        h = modelWave.relu(h)
        h = modelWave.pool4(h)  # (bs, 128L, 8L, 10L)

        h = modelWave.conv5(h)
        h = modelWave.bn5(h)
        h = modelWave.relu(h)
        pool5_wave = modelWave.pool5(h)  # (18, 256L, 4L, 5L)

        h = modelLogmel.conv3(logmel_delta)
        h = modelLogmel.bn3(h)
        h = modelLogmel.relu(h)
        h = modelLogmel.pool3(h) # (bs, 64L, 16L, 40L)

        h = modelLogmel.conv4(h)
        h = modelLogmel.bn4(h)
        h = modelLogmel.relu(h)
        h = modelLogmel.pool4(h)   # (bs, 128L, 8L, 10L)

        h = modelLogmel.conv5(h)
        h = modelLogmel.bn5(h)
        h = modelLogmel.relu(h)
        pool5_logmel = modelLogmel.pool5(h)  # (18, 256L, 4L, 5L)

        pool5 = torch.cat((pool5_wave, pool5_logmel), dim=0)  # (36, 256L, 4L, 5L)

        output = modelDNN(pool5) # (36, 50)

        output = torch.sum(output, dim=0, keepdim=True)
        # print output

        # test_loss += F.cross_entropy(output, label).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # print pred, label.data
        correct += pred.eq(label.data.view_as(pred)).sum()

    # test_loss /= len(sampleSet)
    test_acc = 100. * correct / len(sampleSet)

    print('\nTest set: TestACC: {}/{} {:.2f}%\n'.format(
        correct, len(sampleSet), test_acc))



def main_on_fold(foldNum, model_wave, model_logmel, trainPkl, validPkl):

    modelWave = torch.load(model_wave)
    modelLogmel = torch.load(model_logmel)
    modelDNN = DNN3Layer()

    if args.cuda:
        modelWave.cuda()
        modelLogmel.cuda()
        modelDNN.cuda()

    optimizer = optim.SGD(modelDNN.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    trainDataset = Pool5Dataset(trainPkl, transform=ToTensor())

    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for epoch in range(1, args.epochs + 1):

        exp_lr_scheduler.step()

        train(modelDNN, optimizer, train_loader, epoch)

        if epoch % 10 == 0:
            test(modelWave, modelLogmel, modelDNN, validPkl)

        #  save model
        # if epoch % args.model_save_interval == 0:
        #     model_name = '../model/'+args.network+'_fold'+str(foldNum)+'_v1_epoch'+str(epoch)+'.pkl'
        #     torch.save(model, model_name)
        #     print('model has been saved as: ' + model_name)
#}}}


def main():
    print args.network
    for fold_num in range(1, 5):
        model_wave = '../model/M9_mrf_fold' + str(fold_num) + '_v1_epoch160.pkl'
        model_logmel = '../model/M9Logmel_fold' + str(fold_num) + '_v1_epoch100.pkl'

        trainPkl = '../pool5/M9_mrf_logmel_fold' + str(fold_num) + '_train.cPickle'
        validPkl = '../data_wave_44100/fold' + str(fold_num) + '_valid.cPickle'

        # testPkl = '../data_wave_44100/fold' + str(fold_num) + '_test.cPickle'
        start = time.time()
        main_on_fold(fold_num, model_wave, model_logmel, trainPkl, validPkl)
        print('time/epoch: %fs' % (time.time() - start))


def get_pool5(model_wave, model_logmel, trainPkl):
    modelWave = torch.load(model_wave)
    modelLogmel = torch.load(model_logmel)

    trainDataset = WaveformDataset(trainPkl, window_size=66150, train_slices=args.train_slices, transform=ToTensor())
    train_loader = DataLoader(trainDataset, batch_size=1, shuffle=False, num_workers=8)

    modelWave.eval()
    modelWave.eval()

    pool5_wave = []
    pool5_logmel = []

    for idx, (data, label) in enumerate(train_loader):
        print idx

        item = {}
        item2 = {}
        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])


        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        x2 = modelWave.relu(modelWave.bn1_2(modelWave.conv1_2(data)))
        x2 = modelWave.relu(modelWave.bn2_2(modelWave.conv2_2(x2)))
        x2 = modelWave.pool2_2(x2)  # (batchSize, 64L, 441L)

        x2 = torch.unsqueeze(x2, 1)  # (batchSize, 1L, 64L, 441L)

        h = modelWave.conv3(x2)
        h = modelWave.bn3(h)
        h = modelWave.relu(h)
        h = modelWave.pool3(h)  # (bs, 64L, 16L, 44L)

        h = modelWave.conv4(h)
        h = modelWave.bn4(h)
        h = modelWave.relu(h)
        h = modelWave.pool4(h)  # (bs, 128L, 8L, 10L)

        h = modelWave.conv5(h)
        h = modelWave.bn5(h)
        h = modelWave.relu(h)
        h = modelWave.pool5(h)  # (bs, 256L, 4L, 5L)


        item['pool5'] = h.data.cpu().numpy()
        item['label'] = label.cpu().numpy()[0]

        pool5_wave.append(item)


        feats = []

        for i in range(data.data.shape[0]):
            melspec = librosa.feature.melspectrogram(data.data.cpu().numpy()[i][0], 44100, n_fft=2048, hop_length=150, n_mels=64)  # (40, 442)
            logmel = librosa.logamplitude(melspec)[:, :441]  # (40, 441)
            delta = librosa.feature.delta(logmel)
            feat = np.stack((logmel, delta))

            feats.append(feat)
        feats = np.array(feats)

        feats = torch.from_numpy(feats).type(torch.FloatTensor)

        if args.cuda:
            feats = feats.cuda()
        feats = Variable(feats)

        # feats:(1L, 2L, 64L, 441L)
        h = modelLogmel.conv3(feats)
        h = modelLogmel.bn3(h)
        h = modelLogmel.relu(h)
        h = modelLogmel.pool3(h) # (bs, 64L, 16L, 40L)

        h = modelLogmel.conv4(h)
        h = modelLogmel.bn4(h)
        h = modelLogmel.relu(h)
        h = modelLogmel.pool4(h)   # (bs, 128L, 8L, 10L)

        h = modelLogmel.conv5(h)
        h = modelLogmel.bn5(h)
        h = modelLogmel.relu(h)
        h = modelLogmel.pool5(h)  # (bs, 256L, 4L, 5L)

        item2['pool5'] = h.data.cpu().numpy()
        item2['label'] = label.cpu().numpy()[0]

        pool5_logmel.append(item2)

    print len(pool5_wave)
    print len(pool5_logmel)

    pool5 = pool5_wave + pool5_logmel


    random.shuffle(pool5)
    print pool5[0]['pool5'].shape
    return pool5


def save_pool5_on_fold():

    for fold_num in range(1,5):
        model_wave = '../model/M9_mrf_fold' + str(fold_num) + '_v1_epoch160.pkl'
        model_logmel = '../model/M9Logmel_fold' + str(fold_num) + '_v1_epoch100.pkl'

        trainPkl = '../data_wave_44100/fold' + str(fold_num) + '_train.cPickle'
        pool5 = get_pool5(model_wave, model_logmel, trainPkl)
        save_data('../pool5/M9_mrf_logmel_fold' + str(fold_num) + '_train.cPickle', pool5)

        validPkl = '../data_wave_44100/fold' + str(fold_num) + '_valid.cPickle'
        pool5 = get_pool5(model_wave, model_logmel, validPkl)
        save_data('../pool5/M9_mrf_logmel_fold' + str(fold_num) + '_valid.cPickle', pool5)


if __name__ == "__main__":
    # save_pool5_on_fold()

    if args.mode == 'train':
        main()
    # if args.mode == 'test':
    #     model_name = args.model
    #     model = torch.load(model_name)
    #     testPkl = '../data_wave/fold0_test.cPickle'
    #     test(model, testPkl)

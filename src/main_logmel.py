""" usage:
    python main.py
    python main.py --network=dnn --mode=test --model='../model/dnn_mix.pkl'

"""
import argparse
import time
from network import *
from data_process import *
import os

# Training settings
parser = argparse.ArgumentParser(description='pytorch model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=18, metavar='N',
                            help='input batch size for testing (default: 5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
parser.add_argument('--model_save_interval', type=int, default=100, metavar='N',
                            help='how many epochs to wait before saving the model.')
parser.add_argument('--network', type=str, default='M9Logmel_v3',
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


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#  torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #  torch.cuda.set_device(2)

def train(model, optimizer, train_loader, epoch):
#{{{
    model.train()
    start = time.time()

    running_loss = 0
    running_correct = 0

    for idx, (data, label) in enumerate(train_loader):

        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        optimizer.zero_grad()

        # print data.size()
        output = model(data) # (batch, 50L)
        # print(label)
        # print output.size()
        # exit(0)
        loss = F.cross_entropy(output, label)

        loss.backward()

        optimizer.step()

        _, pred = torch.max(output.data, 1)  # get the index of the max log-probability

        # statistics
        running_loss += loss.data[0]
        running_correct += torch.sum(pred == label.data.view_as(pred))

        # if idx % args.log_interval == 0:
        #
        #     pred = output.data.max(1)[1] # get the index of the max log-probability
        #
        #     correct = pred.eq(label.data.view_as(pred)).sum()
        #     acc = 100.0 * correct / args.batch_size

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * running_correct / len(train_loader.dataset)

    elapse = time.time() - start

    print('Epoch:{} ({:.1f}s) lr:{}  '
          'samples:{}  Loss:{:.3f}  TrainAcc:{:.2f}%'.format(
        epoch, elapse, optimizer.param_groups[0]['lr'],
        len(train_loader.dataset), epoch_loss, epoch_acc))

#}}}


def test(model, test_pkl):
#{{{
    model.eval()

    start = time.time()

    running_loss = 0
    running_correct = 0
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
            melspec = librosa.feature.melspectrogram(win_data, 44100, n_fft=2048, hop_length=150, n_mels=96)  # (40, 442)
            logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)
            # delta = librosa.feature.delta(logmel)
            # feat = np.stack((logmel, delta))

            feat = logmel[np.newaxis, :, :]
            feats.append(feat)
            # wins_data.append(win_data)

        if len(feats) == 0:
            print(item['key'])

        # wins_data = np.array(wins_data)
        #
        # wins_data = wins_data[:, np.newaxis, :]

        feats = np.array(feats)

        data = torch.from_numpy(feats).type(torch.FloatTensor)
        label = torch.LongTensor([label])

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)

        # print data.size()
        output = model(data)
        output = torch.sum(output, dim=0, keepdim=True)
        # print output

        running_loss += F.cross_entropy(output, label).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        running_correct += pred.eq(label.data.view_as(pred)).sum()

    test_loss = running_loss / len(sampleSet)
    test_acc = 100. * running_correct / len(sampleSet)

    elapse = time.time() - start

    print('\nTest set: Average loss: {:.3f} ({:.1f}s), TestACC: {}/{} {:.2f}%\n'.format(
        test_loss, elapse, running_correct, len(sampleSet), test_acc))

    return test_acc
#}}}


def main_on_fold(foldNum, trainPkl, validPkl):

    if args.network == 'M9Logmel':
        model = M9Logmel()
    elif args.network == 'M9Logmel_norm':
        model = M9Logmel_norm()
    elif args.network == 'M9Logmel_v3':
        model = M9Logmel_v3()

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 90], gamma=0.1)

    trainDataset = WaveformDataset(trainPkl, window_size=66150, train_slices=args.train_slices,
                                   add_logmel=True, transform=ToTensor())

    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        exp_lr_scheduler.step()

        train(model, optimizer, train_loader, epoch)

        #  test and save the best model
        if epoch % 25 == 0:
            test_acc = test(model, validPkl)
            if test_acc > best_acc:
                best_acc = test_acc
                # best_model_wts = model.state_dict()

                model_name = '../model/' + args.network + '_fold' + str(foldNum) + '_more.pkl'
                torch.save(model, model_name)
                print('model has been saved as: ' + model_name)


def main():
    print(args.network)
    for fold_num in range(5):
        trainPkl = '../data_wave_more_44100/fold' + str(fold_num) + '_train.cPickle'
        validPkl = '../data_wave_more_44100/fold' + str(fold_num) + '_test.cPickle'
        start = time.time()
        main_on_fold(fold_num, trainPkl, validPkl)
        print('time/epoch: %fs' % (time.time() - start))

if __name__ == "__main__":
        main()


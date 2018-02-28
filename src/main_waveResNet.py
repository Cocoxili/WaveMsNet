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
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=18, metavar='N',
                            help='input batch size for testing (default: 5)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
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
parser.add_argument('--network', type=str, default='waveResNet34',
                            help='EnvNet or EnvNet_mrf or EnvNet_lrf or EnvNet3D or EnvNetMultiScale')
parser.add_argument('--mode', type=str, default='train',
                            help='train or test')
parser.add_argument('--model', type=str, default='../model/EnvNet_v1_fold0_v2_epoch120.pkl',
                            help='trained model path')
parser.add_argument('--train_slices', type=int, default=1,
                            help='slices number of one record divide into.')
parser.add_argument('--test_slices_interval', type=int, default=0.2,
                            help='slices number of one record divide into.')
parser.add_argument('--fs', type=int, default=16000)


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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

        # print data.size()
        output = model(data) # (batch, 50L)
        # print(label)
        # print output.size()
        # exit(0)
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


def test(model, test_pkl):
#{{{
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    win_size = 24000
    stride = int(16000 * args.test_slices_interval)
    sampleSet = load_data(test_pkl)

    for item in sampleSet:
        label = item['label']
        record_data = item['data']
        wins_data = []
        for j in range(0, len(record_data) - win_size + 1, stride):

            win_data = record_data[j: j+win_size]
            # Continue if cropped region is silent

            maxamp = np.max(np.abs(win_data))
            if maxamp < 0.005:
                continue
            wins_data.append(win_data)

        if len(wins_data) == 0:
            print item['key']

        wins_data = np.array(wins_data)

        wins_data = wins_data[:, np.newaxis, :]
        # print wins_data.shape

        data = torch.from_numpy(wins_data).type(torch.FloatTensor) # (N, 1L, 24002L)
        label = torch.LongTensor([label])

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)

        # print data.size()
        output = model(data)
        output = torch.sum(output, dim=0, keepdim=True)
        # print output

        test_loss += F.cross_entropy(output, label).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).sum()

    test_loss /= len(sampleSet)
    test_acc = 100. * correct / len(sampleSet)

    print('\nTest set: Average loss: {:.4f}, TestACC: {}/{} {:.2f}%\n'.format(
        test_loss, correct, len(sampleSet), test_acc))


def adjust_learning_rate(optimizer, epoch):
    if epoch == 1:
        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 80:
        lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 120:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 140:
        lr = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main_on_fold(foldNum, trainPkl, validPkl):

    model = models.resnet34(num_classes=50)
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    trainDataset = WaveformDataset(trainPkl, window_size=24000, train_slices=args.train_slices, transform=ToTensor())

    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for epoch in range(1, args.epochs + 1):

        adjust_learning_rate(optimizer, epoch)

        train(model, optimizer, train_loader, epoch)

        if epoch % 30 == 0:
            test(model, validPkl)

        #  save model
        if epoch % args.model_save_interval == 0:
            model_name = '../model/'+args.network+'_fold'+str(foldNum)+'_v1_epoch'+str(epoch)+'.pkl'
            torch.save(model, model_name)
            print('model has been saved as: ' + model_name)
#}}}


def main():
    for fold_num in range(5):
        trainPkl = '../data_wave/fold' + str(fold_num) + '_train.cPickle'
        validPkl = '../data_wave/fold' + str(fold_num) + '_valid.cPickle'
        start = time.time()
        main_on_fold(fold_num, trainPkl, validPkl)
        print('time/epoch: %fs' % (time.time() - start))

if __name__ == "__main__":
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        model_name = args.model
        model = torch.load(model_name)
        testPkl = '../data_wave/fold0_test.cPickle'
        test(model, testPkl)

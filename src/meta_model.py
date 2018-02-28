
from util import *
from network import *
from data_process import *
from sklearn import linear_model

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def meta_train(model, optimizer, train_loader, epoch):
    log_interval = 100
    batch_size = 128
    model.train()
    #  start = time.time()
    for idx, (data, label) in enumerate(train_loader):

        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])

        data, label = Variable(data.cuda()), Variable(label.cuda())

        optimizer.zero_grad()

        # print data.size()
        output = model(data)  # (batch, 50L)
        # print(label)
        # print output.size()
        # exit(0)
        loss = F.cross_entropy(output, label)
        #  loss = F.nll_loss(output, label)

        loss.backward()

        optimizer.step()

        if idx % log_interval == 0:
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct = pred.eq(label.data.view_as(pred)).sum()
            acc = 100.0 * correct / batch_size
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
            # }}}


def meta_test(model, optimizer, test_loader):
    log_interval = 20
    batch_size = 128
    model.eval()
    #  start = time.time()
    correct = 0
    for idx, (data, label) in enumerate(test_loader):

        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])

        data, label = Variable(data.cuda()), Variable(label.cuda())

        # print data.size()
        output = model(data)  # (batch, 50L)
        # print(label)
        # print output.size()
        # exit(0)
        # loss = F.cross_entropy(output, label)
        #  loss = F.nll_loss(output, label)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).sum()


    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTestACC: {}/{} {:.2f}%\n'.format(
        correct, len(test_loader.dataset), test_acc))



def main_on_fold(foldNum, trainPkl, validPkl):

    lr = 0.01
    momentum = 0.9
    model = DNN3Layer()
    weight_decay = 5e-5
    epochs = 100
    batch_size = 128
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 40, 60], gamma=0.1)

    trainDataset = MetaDataset(trainPkl, transform=ToTensor())
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False, num_workers=2)

    testDataset = MetaDataset(validPkl, transform=ToTensor())
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for epoch in range(1, epochs + 1):

        exp_lr_scheduler.step()

        meta_train(model, optimizer, train_loader, epoch)

        if epoch % 1 == 0:
            meta_test(model, validPkl, test_loader)

        #  save model
        # if epoch % args.model_save_interval == 0:
        #     model_name = '../model/'+args.network+'_fold'+str(foldNum)+'_v1_epoch'+str(epoch)+'.pkl'
        #     torch.save(model, model_name)
        #     print('model has been saved as: ' + model_name)
#}}}


def train_lr(trainPkl, testPkl):

    sampleSet_train = np.loadtxt(trainPkl, delimiter=',')
    sampleSet_test = np.loadtxt(testPkl, delimiter=',')
    X_train = sampleSet_train[:,:100]
    Y_train = sampleSet_train[:, 100]

    X_test = sampleSet_test[:, :100]
    Y_test = sampleSet_test[:, 100]

    correct = 0

    logreg = linear_model.LogisticRegression(C=1e5)


    logreg.fit(X_train, Y_train)

    Z = logreg.predict(X_test)

    print Z.shape
    print Y_test.shape

    correct += np.equal(Z, Y_test).sum()

    test_acc = 100. * correct / Y_test.shape[0]

    print('\nTestACC: {}/{} {:.2f}%\n'.format(
        correct, Y_test.shape[0], test_acc))

def main():
    for fold_num in range(5):
        # trainPkl = 'probabilities.' + str(fold_num) + '.train.txt'
        trainPkl = 'p.' + str(fold_num) + '.txt' # valid
        validPkl = 'probilities.' + str(fold_num) + '.test.txt'
        start = time.time()
        main_on_fold(fold_num, trainPkl, validPkl)
        # train_lr(trainPkl, validPkl)
        print('time/epoch: %fs' % (time.time() - start))

main()

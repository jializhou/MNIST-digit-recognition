from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sub import subMNIST       # testing the subclass of MNIST dataset

#loading dataset
trainset_imoprt = pickle.load(open("train_labeled.p", "rb"))
validset_import = pickle.load(open("validation.p", "rb"))
unlabel_import = pickle.load(open("train_unlabeled.p", "rb"))
testset = pickle.load(open("test.p", "rb" ))
unlabel_import.train_labels = torch.LongTensor(len(unlabel_import)).zero_()
train_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)
unlabel_loader = torch.utils.data.DataLoader(unlabel_import, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

import random
import scipy.misc

def data_transformer(set_import):
    result = []
    for data, target in set_import:
        data_2D = data.numpy().reshape(28, 28)
        p = (data, target)
        result.append(p)
        for i in range(1):
            b = scipy.misc.imrotate(data_2D, random.randint(-15, 15))
            max_val = np.max(b)
            min_val = np.min(b)
            b = (b-min_val) / (max_val-min_val)
            Y = torch.from_numpy(b.reshape(1, 28, 28))
            Y = torch.FloatTensor(1, 28, 28).copy_(Y)
            pair = (Y, target)
            result.append(pair)
    return result

trainset_rotate = data_transformer(trainset_imoprt)
rotate_loader = torch.utils.data.DataLoader(trainset_rotate, batch_size=64, shuffle=True)

#model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.ConvBNRelu0 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, eps = 1e-3),
            nn.ReLU()
            )

        self.ConvBNRelu1 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128, eps = 1e-3),
            nn.ReLU()
            )

        self.ConvBNRelu2 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128, eps = 1e-3),
            nn.ReLU()
            )
        self.ConvBNRelu3 = nn.Sequential(
            nn.Conv2d(128, 256, 2),
            nn.BatchNorm2d(256, eps = 1e-3),
            nn.ReLU()
            ) 

        self.conv_drop = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512, eps = 1e-03)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.ConvBNRelu0(x)
        x = self.conv_drop(x)
        x = F.max_pool2d(x, 2)

        x = self.ConvBNRelu1(x)
        x = self.conv_drop(x)
        x = F.max_pool2d(x, 2)

        x = self.ConvBNRelu2(x)
        x = self.conv_drop(x)

        x = self.ConvBNRelu3(x)
        x = self.conv_drop(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 256)
        x = F.dropout(x, training = self.training)
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.fc3(x)
        x = F.relu(x)

        
        return F.log_softmax(x)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# CPU only training
def train(epoch, lis=[train_loader]):
    model.train()
    loss = 0
    for train_loader in lis:
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            # print (output.size())
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))
    return (loss.data[0])

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return (100. * correct / len(valid_loader.dataset))

#add pseudo label
def semi(valid_loader = unlabel_loader):
    model.eval() 
    result = []
    for batch_idx, (Data, target) in enumerate(valid_loader):
        size = len(Data)
        data, target = Variable(Data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        Y = torch.from_numpy(pred.numpy().reshape(-1))
        Y = torch.LongTensor(size).copy_(Y)
        pair = (Data, Y)
        result.append(pair)
    return result

#training
losslist = []
acclist  = []
for epoch in range(1, 300):
    losslist.append(train(epoch, [rotate_loader]))
    acclist.append(test(epoch))

#semi-supervised learning
losstest = []
acctest  = []
new_loader = semi(unlabel_loader)
for epoch in range(1, 200):
    losstest.append(train(epoch, [rotate_loader, new_loader]))
    losstest.append(test(epoch))

pickle.dump(model, open("model.p", "wb" ))
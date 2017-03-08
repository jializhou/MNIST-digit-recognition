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
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.fc3(x)
        x = F.relu(x)

        
        return F.log_softmax(x)

testset = pickle.load(open("test.p", "rb" ))
test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)
model = pickle.load(open('model.p', 'rb'))
#prediction
label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))

import pandas as pd
predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)
predict_label.to_csv('sample_submission.csv', index=False)
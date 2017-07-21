from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.mlp import mlp
from keras.datasets import mnist
import numpy as np

model = mlp()
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

(train_x,train_y),(test_x,test_y) = mnist.load_data()
train_x = train_x /  255.0
test_x =  test_x / 255.0
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_y = train_y.astype('int')
test_y = test_y.astype('int')

# train_x,test_x = [ np.transpose(x,[1,2,0]) for x in [train_x,test_x] ]


def train(epoch):
    model.train()

    # train_sample = np.random.randint(0,60000,[60000])
    train_sample = np.array(range(0,200))

    data,target = torch.from_numpy(train_x[train_sample]),torch.from_numpy(train_y[train_sample])

    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss =   criterion(output , target )
    loss.backward()
    optimizer.step()


    if epoch % 100 ==0:
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct = pred.eq(target.data).sum()
        correct = correct / target.size()[0]

        print('train loss ' , loss.data.numpy()[0],correct)



def test():
    model.eval()
    test_loss = 0
    correct = 0
    data,target = torch.from_numpy(test_x),torch.from_numpy(test_y)

    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    test_loss +=  criterion(output , target )
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct += pred.eq(target.data).sum()
    correct = correct / len(test_y)

    print('test loss / acc' , test_loss.data.numpy()[0], correct )







for epoch in range(1, 1000000 + 1):
    train(epoch)
    if epoch % 400 == 0:
        test()
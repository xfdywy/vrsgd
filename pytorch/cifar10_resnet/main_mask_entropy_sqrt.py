'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
import pickle
from collections import OrderedDict



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--variance', default=0, type=float, help='variance')
parser.add_argument('--entropy', default=0, type=float, help='entropy')
parser.add_argument('--epoch', default=160, type=int, help='epoch')
parser.add_argument('--net', default='resnet20', type=str, help='net name')


parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=160, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=10)

trainset_test = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
trainloader_test = torch.utils.data.DataLoader(trainset_test, batch_size=1000, shuffle=False, num_workers=10)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





train_acc = []
train_meanloss =[]
train_vrloss =[]

test_acc = []
test_meanloss =[]
test_vrloss =[]

train_var =[]
test_var= []

train_entropy=[]
test_entropy=[]

all_res = {'train_acc' : train_acc ,
'train_meanloss' : train_meanloss,
# 'test_vrloss'  : test_vrloss,
'train_var'  : train_var ,
'test_acc' : test_acc,
'train_entropy' : train_entropy,
'test_meanloss' : test_meanloss,
# 'test_vrloss'  : test_vrloss,
'test_var'  : test_var   ,
'test_entropy' : test_entropy

}

info =OrderedDict()
info['net'] = args.net
info['dataset'] ='cifar10'
info['variance'] = str(args.variance).replace('.','')
info['entropy'] = str(args.entropy).replace('.','')
info['epoch'] = str(args.epoch)

print(info)
file_index = '_maskentropy_sqrt'

file_name = '_'.join( info.values())+file_index

printoutfile = open(file_name + '_printout.txt','w')

print('#########' , args.entropy, args.lr, args.variance)




# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+file_name+'_ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = eval(args.net)()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    #net = MobileNet()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

def adjust_learning_rate(optimizer,epoch,lr0) :
    lr = lr0 *(0.1 ** int(epoch >= 70) * (0.1 ** int(epoch >= 100) ) *(0.5 ** int(epoch >= 150) ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# Training














def train(epoch,all_res,filename):
    print('\nEpoch: %d' % epoch)


    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        one_hot_t = torch.FloatTensor(targets.size()[0], 10)
        one_hot_t = one_hot_t.zero_()
        targets_col = targets.view([-1, 1])

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

            one_hot_t = one_hot_t.cuda()
            targets_col = targets_col.cuda()

        # loss = criterion(outputs, targets)


        one_hot_t = one_hot_t.scatter_(1,targets_col,1)

        one_hot_t = Variable(one_hot_t)
        inputs, targets = Variable(inputs), Variable(targets)
        targets_col = Variable(targets_col)

        outputs = net(inputs)

        log_softmax = torch.nn.LogSoftmax()(outputs)
        softmax = torch.nn.Softmax()(outputs)
        loss = -1 * torch.sum(log_softmax *one_hot_t,1)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        mask = predicted.eq(targets.data)
        mask = Variable(mask.float())


        entropy = -1* torch.mean( mask * torch.sum(softmax * log_softmax,1))

        # print(mask, entropy)

        variance =  torch.var(loss)
        variance = torch.sqrt(variance)

        meanloss = torch.mean(loss) + args.entropy * entropy + args.variance * variance
        # meanloss = criterion(outputs , targets )
        optimizer.zero_grad()
        meanloss.backward()
        optimizer.step()

        train_loss += meanloss.data[0]

        # print(batch_idx , train_loss, total , correct)
        progress_bar(batch_idx, len(trainloader), '(%d/%d) Loss: %.3f,Acc:%.3f%%,Lr %.4f,Var:%.4f,Entr:%.4f'
            % (correct, total ,train_loss/(batch_idx+1), 100.*correct/total,  optimizer.param_groups[0]['lr'],variance.data[0],entropy.data[0]))
        # return ()


 



def test(epoch,all_res,filename):
    global best_acc
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    entropy = []
    var = []
    allloss = []
 

    for batch_idx, (inputs, targets) in enumerate(testloader):

        one_hot_t = torch.FloatTensor(targets.size()[0], 10)
        one_hot_t = one_hot_t.zero_()
        targets_col = targets.view([-1, 1])

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

            one_hot_t = one_hot_t.cuda()
            targets_col = targets_col.cuda()

        # loss = criterion(outputs, targets)


        one_hot_t = one_hot_t.scatter_(1,targets_col,1)

        one_hot_t = Variable(one_hot_t)
        inputs, targets = Variable(inputs,volatile=True), Variable(targets)
        # targets_col = Variable(targets_col)

        outputs = net(inputs)

        log_softmax = torch.nn.LogSoftmax()(outputs)
        softmax = torch.nn.Softmax()(outputs)
        loss = -1 * torch.sum(log_softmax *one_hot_t,1)
        allloss.append( loss )
        entropy.append( -1* torch.mean(torch.sum(softmax * log_softmax,1)))

        variance =  torch.var(loss)

        test_loss += loss.data[0][0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # print((test_loss.cpu().numpy()/(batch_idx+1) , 100.*correct/total, correct, total))
        progress_bar(batch_idx, len(testloader), '(%d/%d) Loss: %.3f,Acc:%.3f%%,Lr %.4f,Var:%.4f,Entr:%.4f'
            % (correct, total ,test_loss/(batch_idx+1), 100.*correct/total,  optimizer.param_groups[0]['lr'],variance.data[0],entropy[-1].data[0]))


    allloss = torch.cat(allloss)
    allentropy = torch.cat(entropy)
    v_meanloss = torch.mean(allloss)
    variance = torch.var(allloss)
    entropy = torch.mean(allentropy)

    all_res['test_meanloss'].append(v_meanloss.data[0])
    all_res['test_var'].append(variance.data[0])
    all_res['test_entropy'].append(entropy.data[0])
    all_res['test_acc'].append(1.0*correct / total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+filename+'_ckpt.t7')
        best_acc = acc

def test_train(epoch,all_res,filename):
 
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    entropy = []
    var = []
    allloss = []
    for batch_idx, (inputs, targets) in enumerate(trainloader_test):

        one_hot_t = torch.FloatTensor(targets.size()[0], 10)
        one_hot_t = one_hot_t.zero_()
        targets_col = targets.view([-1, 1])

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

            one_hot_t = one_hot_t.cuda()
            targets_col = targets_col.cuda()

        # loss = criterion(outputs, targets)


        one_hot_t = one_hot_t.scatter_(1,targets_col,1)

        one_hot_t = Variable(one_hot_t)
        inputs, targets = Variable(inputs,volatile=True), Variable(targets)
        # targets_col = Variable(targets_col)

        outputs = net(inputs)

        log_softmax = torch.nn.LogSoftmax()(outputs)
        softmax = torch.nn.Softmax()(outputs)
        loss = -1 * torch.sum(log_softmax *one_hot_t,1)
        allloss.append( loss )
        entropy.append( -1* torch.mean(torch.sum(softmax * log_softmax,1)))

        variance =  torch.var(loss)

        test_loss += loss.data[0][0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader_test), '(%d/%d) Loss: %.3f,Acc:%.3f%%,Lr %.4f,Var:%.4f,Entr:%.4f'
            % (correct, total ,test_loss/(batch_idx+1), 100.*correct/total,   optimizer.param_groups[0]['lr'],variance.data[0],entropy[-1].data[0]))


    allloss = torch.cat(allloss)
    allentropy = torch.cat(entropy)
    v_meanloss = torch.mean(allloss)
    variance = torch.var(allloss)
    entropy = torch.mean(allentropy)


    all_res['train_meanloss'].append(v_meanloss.data[0])
    all_res['train_var'].append(variance.data[0])
    all_res['train_entropy'].append(entropy.data[0])
    all_res['train_acc'].append(1.0*correct / total)



 


for epoch in range(start_epoch, start_epoch+args.epoch):

    adjust_learning_rate(optimizer,epoch,args.lr)
    # print(type( optimizer.param_groups[0]['lr']))
    train(epoch ,all_res,file_name )
    test(epoch,all_res ,file_name)
    test_train(epoch, all_res, file_name)
    print(optimizer.param_groups[0]['lr'])

    print("##epoch:%d## meanloss : %f/%f ,acc : %f/%f , variance : %f/%f , entropy : %f/%f , lr: %f"
          % (epoch, all_res['train_meanloss'][-1] ,all_res['test_meanloss'][-1] ,
            all_res['train_acc'][-1],all_res['test_acc'][-1],
            all_res['train_var'][-1],all_res['test_var'][-1],
            all_res['train_entropy'][-1],all_res['test_entropy'][-1],optimizer.param_groups[0]['lr']
            )  )
    print("##epoch:%d## meanloss : %f/%f ,acc : %f/%f , variance : %f/%f , entropy : %f/%f, lr:%f"
              % (epoch, all_res['train_meanloss'][-1] ,all_res['test_meanloss'][-1] ,
                all_res['train_acc'][-1],all_res['test_acc'][-1],
                all_res['train_var'][-1],all_res['test_var'][-1],
                all_res['train_entropy'][-1],all_res['test_entropy'][-1],

                 optimizer.param_groups[0]['lr']) ,file = printoutfile)


with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(all_res, f)

printoutfile.close()


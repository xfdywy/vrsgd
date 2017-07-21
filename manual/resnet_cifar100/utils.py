'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
from scipy import misc

import skimage
import skimage.io
import skimage.transform
import random
import numpy as np
from PIL import Image,ImageOps
# import torch.nn as nn
# import torch.nn.init as init


# def get_mean_and_std(dataset):
#     '''Compute the mean and std value of dataset.'''
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     print('==> Computing mean and std..')
#     for inputs, targets in dataloader:
#         for i in range(3):
#             mean[i] += inputs[:,i,:,:].mean()
#             std[i] += inputs[:,i,:,:].std()
#     mean.div_(len(dataset))
#     std.div_(len(dataset))
#     return mean, std

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)-1
# print term_width
TOTAL_BAR_LENGTH = 40.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    # L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def random_flip(img):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else :
        return(img)



def random_crop(img,size=32,padding=4):
    if padding > 0:
        img = ImageOps.expand(img, border=padding, fill=0)

    w, h = img.size
    th, tw = [size]*2
    if w == tw and h == th:
        return img

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return img.crop((x1, y1, x1 + tw, y1 + th))

def normalalize(img,mean,std):
    mean_mat = np.zeros_like(img)
    std_mat = np.zeros_like(img)

    for ii in range(3):
        std_mat[:,:,:,ii] = std[ii]
        mean_mat[:,:,:,ii] = mean[ii]
    img = (img - mean_mat ) / std_mat
    return(img)



def transform_train(image_file ,padding,size):
 
    img = np.uint8(image_file)
    
    img = [Image.fromarray(x) for x in img]

    img = [random_crop(x,size,padding) for x in img]
    img = [random_flip(x) for x in img]
    img = np.array([np.asarray(x) for x in img])
    img = img/255.0
    mean,std = [],[]
    for ii in range(3):
        mean.append( np.mean(img[:,:,:,ii] ))
        std.append( np.std(img[:,:,:,ii] ))

    img = normalalize(img, mean, std)

    return(img)

def transform_test(image_file  ):
    img = np.uint8(image_file)

    img = img/255.0
    mean,std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
#    for ii in range(3):
#        mean.append( np.mean(img[:,:,:,ii] ))
#        std.append( np.std(img[:,:,:,ii] ))

    img = normalalize(img, mean, std)

    return(img)



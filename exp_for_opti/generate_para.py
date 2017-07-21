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



weight = []

for  i in range(10):
    model = mlp()
    weight.append([x.data.numpy() for x in list(model.parameters())])

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 14:34:32 2017

@author: v-yuewng
"""

import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
slim = tf.contrib.slim
#import tflearn

class mnistdnnnetdef():
    def __init__(self,imagesize):
        self.images = tf.placeholder('float32',[None,imagesize,imagesize ,1])
        self.dropout_keep_prob = tf.placeholder('float32',[])

    def buildnet(self):
       
        para_fc1 = tf.get_variable('para_fc1',[28*28,512])
        para_fc1_bias = tf.get_variable('para_fc1_bias',[ 512]) 
          
        para_fc5 = tf.get_variable('para_fc5',[512,10])
        para_fc5_bias = tf.get_variable('para_fc5_bias',[ 10])

        
        net = tf.contrib.slim.flatten(self.images  )
        net = tf.nn.relu(tf.matmul(net,para_fc1) + para_fc1_bias)
        net = tf.nn.dropout(x = net, keep_prob =  self.dropout_keep_prob , name='dropout1') 
        
        
        self.logits = tf.matmul(net,para_fc5) + para_fc5_bias





 

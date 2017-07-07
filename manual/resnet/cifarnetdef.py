#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:29:23 2017

@author: yuewang
"""
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import tflearn

class cifarnetdef():
    def __init__(self,imagesize,n,num_class=10):
        self.images = tf.placeholder('float32',[None,imagesize,imagesize ,3])
        self.dropout_keep_prob = tf.placeholder('float32',[])
        self.n = (n-2)/6
        assert((n-2) % 6 == 0 )
        self.num_class = num_class
    def buildnet(self):
        n=int(self.n)
        
        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True)

        # Real-time data augmentation
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_crop([32, 32], padding=4)

        # Building Residual Network
        net = tflearn.input_data(shape=[None, 32, 32, 3],
                                 placeholder = self.images,
                                 data_preprocessing=img_prep,
                                 data_augmentation=img_aug)
        net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
        net = tflearn.residual_block(net, n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.residual_block(net, n-1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.residual_block(net, n-1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)
        # Regression
        self.logits = tflearn.fully_connected(net, self.num_class)





 

    

class cifarnetdef_simple():
    def __init__(self,imagesize):
        self.images = tf.placeholder('float32',[None,imagesize,imagesize ,3])
 
        self.dropout_keep_prob = tf.placeholder('float32',[])

    def buildnet(self):
        net = Conv2D(32, (3, 3), padding='same',activation='relu' )(self.images)
        net = Conv2D(32, (3, 3), padding='same',activation='relu' )(net)
        net = MaxPooling2D((2, 2) , padding='same')(net)
        net = tf.nn.dropout(net,self.dropout_keep_prob)

        net = Conv2D(64, (3, 3), padding='same',activation='relu' )(net)
        net = Conv2D(64, (3, 3), padding='same',activation='relu' )(net)
        net = MaxPooling2D((2, 2) , padding='same')(net)
        net = tf.nn.dropout(net,self.dropout_keep_prob)
 
        net = Flatten()(net)
        net = Dense(512,activation='relu')(net)
        net = tf.nn.dropout(net,self.dropout_keep_prob)
        
        self.logits = Dense(10)(net)
        self.prob = tf.nn.softmax(self.logits)
        self.net = net
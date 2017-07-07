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

class cifarnetdef():
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
        

    

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:26:04 2017

@author: v-yuewng
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import cifar10

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


import tensorflow as tf

slim = tf.contrib.slim
from collections import OrderedDict


trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)
import numpy as np
import pickle
class cifar10cnnnet:
    def __init__(self,num_classes=10,minibatchsize=1,imagesize=32,dropout_keep_prob=1 ,scope='cifarnet' ,learningrate = 0.001,momentum = 0.5,tradeoff = 0,decay=0):
       self.num_classes=num_classes  
       self.batch_size=minibatchsize
       self.imagesize = imagesize
#       self.dropout_keep_prob=dropout_keep_prob
       self.scope=scope 
       self.prediction_fn=slim.softmax
       self.is_training = True
       
       self.info = OrderedDict()
       
       self.lr = self.lr0 = learningrate
       self.dp = dropout_keep_prob
       self.mt = momentum
       
       self.epoch = 0
       self.tradeoff = tradeoff
       self.info['tradeoff'] = str(self.tradeoff).replace('.','')
       self.decay = decay
       

       
       
       
       
#       self.learningrate = learningrate


    def loaddata(self,data = None):
        if data is None : 
            print(1111)

            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        else:
            self.x_train,self.y_train,self.x_test,self.y_test = data
            
        self.x_train  =  self.x_train / 255.0
        self.y_train  = self.y_train[:,0]
        self.x_test  = self.x_test / 255.0
        self.y_test  = self.y_test[:,0]
            
        self.train_data_num = len(self.x_train)
        self.test_data_num = len(self.x_test)
        
        self.one_epoch_iter_num = self.train_data_num   //  self.batch_size
        
        self.qujian()
        self.shuffledata()
        self.info['dataset'] = 'cifar10'
  

    def buildnet(self):
#        self.end_points = {}
        tf.reset_default_graph()
        self.learningrate = tf.placeholder('float32',[ ])
        self.images = tf.placeholder('float32',[None,self.imagesize,self.imagesize ,3])
        self.label = tf.placeholder('int32',[None,])
        self.dropout_keep_prob = tf.placeholder('float32',[])
        self.momentum = tf.placeholder('float32',[])
        
#        28*28 --- 64 ， 64 --- 32， 32 --- 16 ， 16 ----10
        
        
        
        with tf.variable_scope(self.scope, 'CifarNet', [self.images, self.num_classes]):
        

            # model = Sequential()

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
 

             


            
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,labels = self.label)
            self.meanloss = tf.reduce_mean(self.loss)
            
            self.var = tf.reduce_mean(tf.pow(self.loss - tf.reduce_mean(self.loss),2)) 
            self.vrloss = self.meanloss + self.tradeoff * self.var

            self.parameters = tf.trainable_variables()
            
            self.grad_op = tf.gradients(self.vrloss, self.parameters)
            self.hess_op = None

 
            

            
            
            self.train_sgd = tf.train.GradientDescentOptimizer(self.learningrate).minimize(self.vrloss)
            self.train_momentum   = tf.train.MomentumOptimizer(self.learningrate,self.momentum).minimize(self.vrloss)
            self.train_adam = tf.train.AdamOptimizer(self.learningrate).minimize(self.vrloss)
            
            self.init_allvars = tf.global_variables_initializer()
            
            self.saver = tf.train.Saver()
            self.info['nettype'] = 'cnn'
            
     
    def init_net(self ):
        self.sess = tf.Session()
        self.sess.run(self.init_allvars)
        self.global_step = 0
        self.data_point = 0
        self.epoch_final = False
        
    def data_mode(self,mode_data) :
        
        self.mode_data = mode_data
        
        if mode_data == 1:
            self.info['sample_method'] = 'random_sample'

        elif mode_data ==2:
            self.info['sample_method'] = 'order_batch'
            
            
    def train_mode(self,mode_train):
        self.mode_train = mode_train
#        self.decay = 0
        
        if mode_train == 1:           
            self.info['opti_method'] = 'sgd'
#            self.decay = 1e-8
 
            
        elif mode_train ==2 :
            self.lr *= (1.0 / (1.0 + self.decay * self.global_step))
            self.info['opti_method'] = 'momentum'
 
            
        elif mode_train ==3:
            self.info['opti_method'] = 'adam'
 
            
        elif mode_train == 4:
            pass
     
            
#    def fill_train_data(self):
#        self.datax = self.x_train[:20000]
#        self.datay = self.y_train[:20000]

    def evaluate_train(self):
#        vrlosstemp = []
#        meanlosstemp = []
        acctemp = []
#        vartemp = []
        losstemp = []
        
        
        for ii in range(5):
            ind = [ii*10000 , (ii+1)*10000]
            self.datax = self.x_train[ind[0] : ind[1]]
            self.datay = self.y_train[ind[0] : ind[1]]
 
            self.calacc()
            acctemp.append(self.v_acc) 
            losstemp.append(self.calallloss())
            
        self.v_acc = np.mean(acctemp)
        self.v_meanloss = np.mean(losstemp)
        self.v_var = np.var(losstemp)
        self.v_vrloss = self.v_meanloss + self.v_var * self.tradeoff
            
            
            
            
            
    
    def fill_test_data(self):
        self.datax = self.x_test
        self.datay = self.y_test
 
        
    def next_batch(self):
        
        if self.data_point >= self.one_epoch_iter_num -1:
            self.epoch_final = True
        
        if self.epoch_final == True:
            self.data_point =0
            self.shuffledata()
            self.epoch = self.epoch+1
            self.epoch_final = False        
        
        if self.mode_data == 1: 
            

            
            sample = np.random.randint(0,self.train_data_num,[self.batch_size])
            self.datax = self.x_train[sample  ]
            self.datay = self.y_train[sample]
            self.data_point += 1
            
        elif self.mode_data ==2:             
 
          
            sample =  self.data_index[self.batch_index[self.data_point][0] : self.batch_index[self.data_point][1] ]
      
            self.datax = self.x_train[sample]
            self.datay = self.y_train[sample]
            
            self.data_point = self.data_point + 1
        
    
    def train_net(self):
        mode_train = self.mode_train
        self.global_step += 1
        
        self.feed_dict = {self.images : self.datax, self.label : self.datay ,
                     self.learningrate : self.lr , self.dropout_keep_prob : self.dp,
                     self.momentum : self.mt}
        
        if mode_train == 1:      
            self.lr *= (1.0 / (1.0 + self.decay * self.global_step))
#            self.info['opti_method'] = 'sgd'
            self.sess.run(self.train_sgd,self.feed_dict)
            self.lr *= (1.0 / (1.0 + self.decay * self.global_step))
            
        elif mode_train ==2 :
            self.lr *= (1.0 / (1.0 + self.decay * self.global_step))
#            self.info['opti_method'] = 'momentum'
            self.sess.run(self.train_momentum,self.feed_dict)
            
        elif mode_train ==3:
#            self.info['opti_method'] = 'adam'
            self.sess.run(self.train_adam,self.feed_dict)
            
        elif mode_train == 4:
            pass
     
        
        
   
        

        
    def shuffledata(self):               
        all_data_index = list(range(self.train_data_num))
        np.random.shuffle(all_data_index)
        all_data_index = np.array(all_data_index)        
        self.data_index = all_data_index    
        
        
    def qujian(self):
        self.batch_index =[]
        for ii in range(self.one_epoch_iter_num):
            self.batch_index.append( [ii * self.batch_size,(ii+1) * self.batch_size ] )
            
               
    def calloss(self):
        self.v_meanloss = self.sess.run(self.meanloss,feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp})
        self.v_vrloss = self.sess.run(self.vrloss,feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp})
        self.v_var = self.sess.run(self.var,feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp})
 
            
    def calacc(self):
        predict = self.sess.run(self.logits,feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp})
        predict = np.argmax(predict,1)
        self.v_acc = (np.sum(predict == self.datay)*1.0 / len(self.datay))
#        return(self.acc)
        
    def eval_grad(self ):
        v_grad = self.sess.run(self.grad_op,feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp})
        self.v_grad = v_grad
        self.cal_norm()
#        self.cal_norm_max()
#        self.cal_norml1()
    def calallloss(self):
        return(self.sess.run(self.loss,feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp}))
        
        
#    def eval_hess(self):
#        if self.hess_op == None:
#            self.hess_op = tf.hessians(self.meanloss,self.parameters)
#        self.v_hess = self.sess.run(self.hess_op, feed_dict = {self.images : self.datax , self.label : self.datay ,self.dropout_keep_prob : self.dp})
#        
#        
        
    def eval_weight(self):
        self.v_weight  = self.sess.run(self.parameters) 
        
 
        
#    def save_model(self ,path, name):
#        tfmodel_name = name + '_' + '_'.join(self.info.values())
#        self.saver.save(self.sess,path+tfmodel_name)
#    def save_weight(self,name):    
#        tfmodel_name = name + '_' + '_'.join(self.info.values())
#        with open('./save/dnn/'+tfmodel_name+'.pkl','wb') as f:
#            pickle.dump(self.v_weight , f)
            
            
    def cal_norm(self):
       self.v_grad_norm_l2 =  np.array([np.linalg.norm(np.ravel(x))  for x in self.v_grad])
       self.v_grad_norm_max =  np.array([np.linalg.norm(np.ravel(x),np.inf)  for x in self.v_grad])
       self.v_grad_norm_l1 =  np.array([np.linalg.norm(np.ravel(x),1)  for x in self.v_grad])
            
        
        
        
        
        
        
        
        
 

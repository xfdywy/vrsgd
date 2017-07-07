from __future__ import print_function
from cifar10cnn import cifar10cnnnet
import numpy as np
import matplotlib.pyplot as plt
from utils import progress_bar
 
import pickle

tradeoff = 1  
tradeoff2 =0.01

model   = cifar10cnnnet(minibatchsize=128, learningrate = 0.01,tradeoff = tradeoff , tradeoff2=tradeoff2,momentum=0.9,decay = 0)


model.buildnet()

model.loaddata()

model.init_net()

model.data_mode(1)
model.train_mode(2)

epoch=0


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





weight = []

grad_norm = []

dis =[]
#model.lr = 0.1

file_index = '_entropy'

file_name = '_'.join(model.info.values())+file_index

printoutfile = open(file_name + '_printout.txt','w')

print(file_name)
    # model.eval_weight()
    # weight.append(model.v_weight)
    # model.lr *= 0.9
temp_step =0

temploss = []

model.global_step = 0
model.next_batch()   
model.train_net()
model.calacc()
model.calloss()
 

for ii in range(65000): 

    # if model.epoch >30:
    #     break

    if ii == 32000:
        model.lr = model.lr/10.0
    if ii == 64000:
        model.lr = model.lr /10.0


    model.global_step = 0
    model.next_batch()   
    model.train_net()
    model.calacc()
    model.calmeanloss()
    progress_bar(model.data_point, model.one_epoch_iter_num,'epoch:%d, loss:%.5f, acc:%.5f, var:%.5f, entr:%.5f, lr:%.5f' %(model.epoch,model.v_meanloss,model.v_acc,model.v_var,model.v_entropy,model.lr))
 
    
    
    
    if model.epoch_final == True:
#        if model.lr > 1e-5 and model.epoch % 2 == 0:
#            model.lr = model.lr / 2.0
#            print('learning rate decrease to ', model.lr )
#            print('learning rate decrease to ', model.lr,file = printoutfile)

        model.eval_weight()
        weight.append(model.v_weight)
#            model.save_model('exp1')
    

    if (model.data_point+1) % (model.one_epoch_iter_num // 3) == 0 :
        model.evaluate_train()
        train_vrloss.append(model.v_vrloss)
        train_meanloss.append(model.v_meanloss)    
        train_var.append(model.v_var)
        train_entropy.append(model.v_entropy) 
        train_acc.append(model.v_acc)
             
        model.fill_test_data()
        model.calloss()
        test_vrloss.append(model.v_vrloss)
        test_meanloss.append(model.v_meanloss)          
        test_var.append(model.v_var)
        test_entropy.append(model.v_entropy)
        model.calacc()
        test_acc.append(model.v_acc)
        
        
        print("##epoch:%d## meanloss : %f/%f , vrloss : %f/%f , acc : %f/%f , variance : %f/%f , entropy : %f/%f, lr : %f" 
              % (model.epoch,train_meanloss[-1] , test_meanloss[-1] , train_vrloss[-1],test_vrloss[-1],train_acc[-1],test_acc[-1], train_var[-1],test_var[-1],train_entropy[-1],test_entropy[-1],model.lr))
        print("##epoch:%d## meanloss : %f/%f , vrloss : %f/%f , acc : %f/%f , variance : %f/%f , entropy : %f/%f, lr : %f" 
              % (model.epoch,train_meanloss[-1] , test_meanloss[-1] , train_vrloss[-1],test_vrloss[-1],train_acc[-1],test_acc[-1], train_var[-1],test_var[-1],train_entropy[-1],test_entropy[-1],model.lr) ,file = printoutfile)


            
#            print('epoch',model.epoch,'meanloss', model.v_meanloss,'vrloss', model.v_vrloss , 'variance', model.v_var,'acc',model.v_acc,'lr',model.lr)
           
            
            
            
            
all_res = {'train_acc' : train_acc ,
'train_meanloss' : train_meanloss,
'test_vrloss'  : test_vrloss,
'train_var'  : train_var , 
'test_acc' : test_acc,
'train_entropy' : train_entropy,
'test_meanloss' : test_meanloss,
'test_vrloss'  : test_vrloss,
'test_var'  : test_var   ,
'test_entropy' : test_entropy 

}
            
with open(file_name + '.pkl','wb') as f:
    pickle.dump(all_res,f)      
#    model.fill_train_data()
#    model.eval_grad()
#    print(model.v_grad_max)
#    print(model.v_grad_min)
#    print(model.v_grad_norm)
#    grad_norm.append(model.v_grad_norm)
##    
#    model.eval_weight()
#    weight.append(model.v_weight)
#    
#    dis_1 = np.linalg.norm(weight[-1]-weight[-2])
#    dis.append(dis_1)   
#    print(dis_1 )
printoutfile.close()
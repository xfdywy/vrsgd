from mnistdnn import mnistnet
import numpy as np
import matplotlib.pyplot as plt
 
import pickle

model   = mnistnet(minibatchsize=100, learningrate = 0.01,tradeoff = 2)


model.buildnet()

model.loaddata()

model.init_net()

model.data_mode(1)
model.train_mode(1)

epoch=0


train_acc = []
train_meanloss =[]
train_vrloss =[]

test_acc = []
test_meanloss =[]
test_vrloss =[]

train_var =[]
test_var= []





weight = []

grad_norm = []

dis =[]
model.lr = 0.05

for jj in range(10):
    model.eval_weight()
    weight.append(model.v_weight)
    model.lr *= 0.9
    for ii in range(10000): 

        model.global_step = 0
        model.next_batch()
    
    
        model.train_net( )
        
        if epoch < model.epoch:
            epoch = model.epoch
            print('epoch %d done!'%(epoch))
            break
#            model.save_model('exp1')
        
    
        if (ii+1) % 10 == 0 :
            model.fill_train_data()
            model.calloss()
            train_vrloss.append(model.v_vrloss)
            train_meanloss.append(model.v_meanloss)    
            train_var.append(model.v_var)
            model.calacc()
            train_acc.append(model.v_acc)
             
            model.fill_test_data()
            model.calloss()
            test_vrloss.append(model.v_vrloss)
            test_meanloss.append(model.v_meanloss)          
            test_var.append(model.v_var)
            model.calacc()
            test_acc.append(model.v_acc)
            
            
            print("##epoch:%d## meanloss : %f/%f , vrloss : %f/%f , acc : %f/%f , variance : %f/%f , lr : %f" 
                  % (model.epoch,train_meanloss[-1] , test_meanloss[-1] , train_vrloss[-1],test_vrloss[-1],train_acc[-1],test_acc[-1], train_var[-1],test_var[-1],model.lr))
 
            
#            print('epoch',model.epoch,'meanloss', model.v_meanloss,'vrloss', model.v_vrloss , 'variance', model.v_var,'acc',model.v_acc,'lr',model.lr)
           
            
            
            
            
all_res = {'train_acc' : train_acc ,
'train_meanloss' : train_meanloss,
'train_var'  : train_var , 
'test_acc' : test_acc,
'test_meanloss' : test_meanloss,
'test_vrloss'  : test_vrloss,
'test_var'  : test_var    

}
            
with open('varloss_2.pkl','wb') as f:
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

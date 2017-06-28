from mnistdnn import mnistnet
import numpy as np
import matplotlib.pyplot as plt
 
import pickle
tradeoff =0
model   = mnistnet(minibatchsize=128, learningrate = 0.1,tradeoff = tradeoff,momentum=0.9,decay=1e-7)


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





weight = []

grad_norm = []

dis =[]
#model.lr = 0.1

file_index = '_test'

file_name = '_'.join(model.info.values())+file_index

printoutfile = open(file_name + '_printout.txt','w')

print(file_name)


temp_step =0
#temp_loss = []
#
#temp_acc  = []



for ii in range(1000000): 

    if model.epoch >50:
        break


#    model.global_step = 0
    model.next_batch()   
    model.train_net( )
#    model.calloss()
#    model.calacc()
#    temp_loss.append(model.v_vrloss)
#    temp_acc.append(model.v_acc)
    
    temp_step += 1
#    model.lr *= (1.0 / (1.0 + model.decay*model.global_step))
        
#
#    if  temp_step >200 and  np.mean(temp_loss[-200:-100]) -  np.mean(temp_loss[-100:]) < -0.001 and model.lr > 1e-6:
#            model.lr = model.lr / 2.0
#            temp_step = 0
#            print('learning rate decrease to ', model.lr, np.mean(temp_loss[-200:-100]) -  np.mean(temp_loss[-100:]))
#            print('learning rate decrease to ', model.lr,file = printoutfile)
#    
    
    
    if model.epoch_final == True:
        model.eval_weight()
        weight.append(model.v_weight)
#            model.save_model('exp1')
    

    if model.data_point % (model.one_epoch_iter_num // 5 ) == 0 :
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
        print("##epoch:%d## meanloss : %f/%f , vrloss : %f/%f , acc : %f/%f , variance : %f/%f , lr : %f" 
              % (model.epoch,train_meanloss[-1] , test_meanloss[-1] , train_vrloss[-1],test_vrloss[-1],train_acc[-1],test_acc[-1], train_var[-1],test_var[-1],model.lr) ,file = printoutfile)



            
#            print('epoch',model.epoch,'meanloss', model.v_meanloss,'vrloss', model.v_vrloss , 'variance', model.v_var,'acc',model.v_acc,'lr',model.lr)
           
            
            
            
            
all_res = {'train_acc' : train_acc ,
'train_meanloss' : train_meanloss,
'test_vrloss'  : test_vrloss,
'train_var'  : train_var , 
'test_acc' : test_acc,
'test_meanloss' : test_meanloss,
'test_vrloss'  : test_vrloss,
'test_var'  : test_var    

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
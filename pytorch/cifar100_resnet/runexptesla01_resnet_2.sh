gpu=7

CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.1 --entropy 1 --epoch 180 --net resnet20
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.1 --entropy 0.1 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.1 --entropy 0.01 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.01 --entropy 1 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.01 --entropy 0.1 --epoch 180 --net resnet20  
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.01 --entropy 0.01 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.001 --entropy 1 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.001 --entropy 0.1 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=${gpu} python main_sqrt.py --lr 0.1 --variance 0.001 --entropy 0.01 --epoch 180 --net resnet20 



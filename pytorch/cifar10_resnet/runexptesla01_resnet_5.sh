CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0 --entropy 0 --epoch 180 --net resnet20
CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0.1 --entropy 0 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0.05 --entropy 0 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0.01 --entropy 0 --epoch 180 --net resnet20  
CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0.005 --entropy 0 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0.001 --entropy 0 --epoch 180 --net resnet20 
CUDA_VISIBLE_DEVICES=3 python main_mask_entropy_sqrt.py --lr 0.1 --variance 0.5 --entropy 0 --epoch 180 --net resnet20


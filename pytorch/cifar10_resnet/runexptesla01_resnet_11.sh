cuda=4

for index in 1 2 3 4 5 6
do
CUDA_VISIBLE_DEVICES=${cuda} python main_sqrt_other_traindata_nowd.py --lr 0.1 --variance 0 --entropy 0 --epoch 180 --net resnet20 --index ${index}
done
 

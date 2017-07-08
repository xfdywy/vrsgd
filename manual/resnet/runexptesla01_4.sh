for  i in 11 12 
do 
    CUDA_VISIBLE_DEVICES=4 python experiment${i}.py
done

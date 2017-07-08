for  i in 1 2 3 
do 
    CUDA_VISIBLE_DEVICES=7 python experiment${i}.py
done

for  i in 9 10 
do 
    CUDA_VISIBLE_DEVICES=5 python experiment${i}.py
done

for  i in 7 8 
do 
    CUDA_VISIBLE_DEVICES=6 python experiment${i}.py
done

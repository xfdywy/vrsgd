for  i in 5 6 
do 
    CUDA_VISIBLE_DEVICES=7 python experiment${i}.py
done

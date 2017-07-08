for  i in 5 6 7 8
do 
    CUDA_VISIBLE_DEVICES=2 python experiment${i}.py
done

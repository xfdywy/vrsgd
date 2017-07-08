for  i in  2 3 4
do 
    CUDA_VISIBLE_DEVICES=1 python experiment${i}.py
done

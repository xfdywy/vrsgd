for  i in 9 10 11 12 
do 
    CUDA_VISIBLE_DEVICES=3 python experiment${i}.py
done


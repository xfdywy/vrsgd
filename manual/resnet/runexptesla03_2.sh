for i in  4 13 14 
do 
	CUDA_VISIBLE_DEVICES=4 python experiment${i}.py
done

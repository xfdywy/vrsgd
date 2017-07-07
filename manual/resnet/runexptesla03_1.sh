for i in 2 3
do 
	CUDA_VISIBLE_DEVICES=6 python experiment${i}.py
done

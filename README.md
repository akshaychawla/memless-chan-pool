# memless-chan-pool
Memory less channel pooling for TN

## To Run: 

nvcc max3d.cu utils.cu

 
./a.out

## CIFAR
export CUDA_VISIBLE_DEVICES="7"
python train.py --init_lr 0.1 --momentum 0.9 --epochs 300 \
--batch_size 64 --display_iter 50 --save_iter 10 --model denseprc \
--weight_decay 1e-5 --dataset cifar --CMP 4 --G 12 \
--logs ./runs/fast_fix/exp2 --lr_schedule 150 225

#!/bin/bash
# Training script with DDP (DistributedDataParallel) support
# Usage: 
#   Single GPU: bash arun_train.sh
#   Multi-GPU DDP: bash arun_train.sh (uses torchrun automatically)

name="none"

# Number of GPUs to use
NGPUS=4

if [ "$name" = "none" ]; then
    # DDP training with torchrun
    torchrun --nproc_per_node=$NGPUS --master_port=29500 \
        train.py --dataroot ../datasets/sperm_instance_800 --name $name --display_env $name \
                    --model usseg --dataset_mode usseg --lambda_A 1 --lambda_B 1 \
                    --n_epochs 100 --n_epochs_decay 300 --lr 0.0001 --lr_policy step --lr_decay_iters 100 \
                    --batch_size 1 --num_threads 4 \
                    --no_flip --preprocess none --load_size 800 --crop_size 800 --max_instances 4 \
                    --gpu_ids 4,5,6,7  > train_$name.log 2>&1
elif [ "$name" = "crop" ]; then
    torchrun --nproc_per_node=$NGPUS --master_port=29500 \
        train.py --dataroot ../datasets/sperm --name $name --display_env $name \
                    --model usseg --dataset_mode usseg --lambda_A 10 --lambda_B 10 \
                    --n_epochs 100 --n_epochs_decay 100 --lr_policy step --lr_decay_iters 50 \
                    --batch_size 1 --num_threads 2 \
                    --no_flip --preprocess crop --crop_size 512 \
                    --gpu_ids 0,1,2,3 \
                    --display_id -1 --no_html > train_$name.log 2>&1
fi

# Note: 
# - batch_size is now per-GPU (effective batch = batch_size * NGPUS)
# - num_threads is per-GPU
# - --display_id -1 --no_html disables visdom (recommended for DDP)
# - To use the old DataParallel mode, use: python train.py ... (without torchrun)

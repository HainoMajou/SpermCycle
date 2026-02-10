name=pregen
NGPUS=4

# DDP training with torchrun
torchrun --nproc_per_node=$NGPUS --master_port=29600 \
    train.py --dataroot ../datasets/sperm_instance_800 --name $name \
                --model generative --dataset_mode usseg \
                --n_epochs 200 --n_epochs_decay 600 --lr 0.01 --lr_policy step --lr_decay_iters 200 \
                --batch_size 4 --num_threads 8 \
                --no_flip --preprocess none --load_size 800 --crop_size 800 --max_instances 7 \
                --gpu_ids 4,5,6,7 --use_amp  > log/train_$name.log 2>&1

name=mask2former_pregen
NGPUS=8
gpu_ids=0,1,2,3,4,5,6,7

# DDP training with torchrun
torchrun --nproc_per_node=$NGPUS --master_port=29600 \
    train.py --dataroot checkpoints/inference --name $name \
                --model instance_seg --dataset_mode instance_seg \
                --n_epochs 200 --n_epochs_decay 400 --lr 0.0001 --lr_policy step --lr_decay_iters 200 \
                --batch_size 4 --num_threads 4 \
                --no_flip --preprocess none --load_size 800 --crop_size 800 --max_instances 10 \
                --gpu_ids $gpu_ids --use_amp \
                --preseg hf > log/train_$name.log 2>&1
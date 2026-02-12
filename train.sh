name="800size_10ins"
NGPUS=4
gpu_ids=4,5,6,0

if [ "$name" = "800size_10ins" ]; then
    torchrun --nproc_per_node=$NGPUS --master_port=29700 \
        train.py --dataroot ../datasets/sperm_instance_800_opt --name $name \
                    --model usseg --dataset_mode usseg \
                    --lambda_A 1 --lambda_B 1 --lambda_GB 20 --lambda_DB 1 --lambda_GA 20 --lambda_DA 1 \
                    --n_epochs 150 --n_epochs_decay 450 --lr 0.0001 --lr_policy step --lr_decay_iters 150 \
                    --batch_size 2 --num_threads 4 \
                    --no_flip --preprocess none --load_size 800 --crop_size 800 --max_instances 10 \
                    --gpu_ids $gpu_ids --use_amp  > log/train_$name.log 2>&1
elif [ "$name" = "800size_10ins_preseg_pregen" ]; then
    torchrun --nproc_per_node=$NGPUS --master_port=29500 \
        train.py --dataroot ../datasets/sperm_instance_800_opt --name $name \
                    --model usseg --dataset_mode usseg \
                    --lambda_A 1 --lambda_B 1 --lambda_GB 20 --lambda_DB 1 --lambda_GA 20 --lambda_DA 1 \
                    --n_epochs 150 --n_epochs_decay 450 --lr 0.0001 --lr_policy step --lr_decay_iters 150 \
                    --batch_size 2 --num_threads 4 \
                    --no_flip --preprocess none --load_size 800 --crop_size 800 --max_instances 10 \
                    --preseg checkpoints/800size_10ins_preseg/latest \
                    --pregen checkpoints/800size_10ins_preseg/latest \
                    --gpu_ids $gpu_ids --use_amp > log/train_$name.log 2>&1
elif [ "$name" = "800size_10ins_preseg" ]; then
    torchrun --nproc_per_node=$NGPUS --master_port=29600 \
        train.py --dataroot ../datasets/sperm_instance_800_opt --name $name \
                    --model usseg --dataset_mode usseg \
                    --lambda_A 1 --lambda_B 1 --lambda_GB 20 --lambda_DB 1 --lambda_GA 20 --lambda_DA 1 \
                    --n_epochs 150 --n_epochs_decay 450 --lr 0.0001 --lr_policy step --lr_decay_iters 150 \
                    --batch_size 2 --num_threads 4 \
                    --no_flip --preprocess crop --load_size 800 --crop_size 800 --max_instances 10 \
                    --gpu_ids $gpu_ids --use_amp \
                    --preseg hf > log/train_$name.log 2>&1
elif [ "$name" = "800size_10ins_pregen" ]; then
    torchrun --nproc_per_node=$NGPUS --master_port=29600 \
        train.py --dataroot ../datasets/sperm_instance_800_opt --name $name \
                    --model usseg --dataset_mode usseg \
                    --lambda_A 1 --lambda_B 1 --lambda_GB 20 --lambda_DB 1 --lambda_GA 20 --lambda_DA 1 \
                    --n_epochs 150 --n_epochs_decay 450 --lr 0.00004 --lr_policy step --lr_decay_iters 150 \
                    --batch_size 2 --num_threads 4 \
                    --no_flip --preprocess crop --load_size 800 --crop_size 800 --max_instances 10 \
                    --gpu_ids $gpu_ids --use_amp \
                    --pregen checkpoints/800size_10ins_preseg/500_net_G_B.pth > log/train_$name.log 2>&1
else
    echo "Invalid name: $name"
    exit 1
fi


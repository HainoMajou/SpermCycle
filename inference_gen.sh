#!/bin/bash
# Inference script for image generation from masks

# Configuration
MODEL_PATH="checkpoints/800size_10ins/600_net_G_B.pth"
INPUT_NC=1
OUTPUT_NC=1
NGF=64
NET_G="resnet_9blocks"
NORM="instance"
GPU_IDS="0,1,2,3,4,5,6,7"
IMAGE_SIZE=800

# Parse command line arguments
INPUT_PATH=${1:-"../datasets/sperm_instance_800_opt/trainA_instance"}  # default mask path
OUTPUT_PATH=${2:-"checkpoints/inference"}  # default output path
BACKGROUND_PATH=${3:-"../datasets/sperm_instance_800_opt/trainC"}  # optional background

# Inference on a directory (batch) with multi-GPU parallel processing
echo "Running multi-GPU inference on directory: $INPUT_PATH"
python inference_gen.py \
    --model_path $MODEL_PATH \
    --mask_dir $INPUT_PATH \
    --output_dir $OUTPUT_PATH \
    --input_nc $INPUT_NC \
    --output_nc $OUTPUT_NC \
    --ngf $NGF \
    --netG $NET_G \
    --norm $NORM \
    --gpu_ids $GPU_IDS \
    --image_size $IMAGE_SIZE \
    --background_dir $BACKGROUND_PATH

echo "Inference complete!"

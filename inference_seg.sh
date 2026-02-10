#!/bin/bash
# Inference script for Mask2Former instance segmentation model
#
# Usage examples:
#   1. Inference on a single image:
#      bash inference.sh single path/to/image.png
#
#   2. Inference on a directory:
#      bash inference.sh dir path/to/images/
#
#   3. Custom model and parameters:
#      bash inference.sh custom

# Configuration
# MODEL_PATH="checkpoints/train_mask2former_pregen/latest_net_G.pth"
MODEL_PATH="checkpoints/800size_10ins/600_net_G_A.pth"
GPU_IDS="0,1,2,3,4,5,6,7"
MAX_INSTANCES=10
SCORE_THRESH=0.5
MASK_THRESH=0.5
IMAGE_SIZE=800

# Parse command line arguments
# INPUT_PATH=${1:-"../datasets/sperm_instance_800_opt/trainB"}  
INPUT_PATH=${1:-"checkpoints/inference/images"}
LABEL_PATH=${2:-"checkpoints/inference/masks"}

# OUTPUT_DIR=${2:-"checkpoints/inference"}
OUTPUT_DIR=${3:-"checkpoints/inference_real_notpregen"}

echo "Running inference on directory: $INPUT_PATH"
python inference_seg.py \
    --model_path $MODEL_PATH \
    --input_dir $INPUT_PATH \
    --output_dir $OUTPUT_DIR \
    # --label_dir $LABEL_PATH \
    --gpu_ids $GPU_IDS \
    --max_instances $MAX_INSTANCES \
    --score_thresh $SCORE_THRESH \
    --mask_thresh $MASK_THRESH \
    --image_size $IMAGE_SIZE \
    --save_masks \
    # --hf_pretrained

echo "Inference complete!"

"""
Inference script for Mask2Former instance segmentation model.

This script loads a trained Mask2Former model and performs multi-GPU parallel inference on test images.
It can process single images or entire directories, and outputs visualized instance masks.
All inference is done using multi-GPU parallelism for optimal performance.

"""

import os
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.networks import generator
from tqdm import tqdm
from util.load_inference import load_and_preprocess_image, load_instance_masks

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mask2Former Inference')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model checkpoint (e.g., checkpoints/pretrained_seg/latest_net_G.pth)')
    parser.add_argument('--hf_pretrained', action='store_true',
                       help='Use Hugging Face pretrained weights (facebook/mask2former-swin-tiny-coco-instance)')
    parser.add_argument('--max_instances', type=int, default=7,
                       help='Maximum number of instances to detect (should match training)')
    parser.add_argument('--score_thresh', type=float, default=0.5,
                       help='Score threshold for filtering detections')
    parser.add_argument('--mask_thresh', type=float, default=0.5,
                       help='Threshold for binarizing instance masks')

    # Input/Output parameters
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Path to directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./results/inference',
                       help='Directory to save inference results')
    parser.add_argument('--label_dir', type=str, default=None,
                       help='Path to directory containing label images')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save individual instance masks as separate files')
    # Processing parameters
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Comma-separated list of GPU IDs for multi-GPU inference (e.g., "0,1,2,3")')
    parser.add_argument('--image_size', type=int, default=800,
                       help='Input image size (images will be resized to this)')

    args = parser.parse_args()
    
    # Parse GPU IDs - always use multi-GPU mode
    if args.gpu_ids is not None:
        args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        # Default: use all available GPUs
        if torch.cuda.is_available():
            args.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            raise RuntimeError("No CUDA devices available. Multi-GPU inference requires at least one GPU.")
    
    if len(args.gpu_ids) == 0:
        raise RuntimeError("At least one GPU is required for multi-GPU inference.")

    # Validate input
    if not args.hf_pretrained and args.model_path is None:
        parser.error("Either --model_path or --hf_pretrained must be specified")

    return args


def load_model(model_path, max_instances, device, hf_pretrained):
    """Load the trained Mask2Former model.

    Args:
        model_path: Path to model checkpoint
        max_instances: Maximum number of instances
        score_thresh: Score threshold
        mask_thresh: Mask threshold
        device: torch device

    Returns:
        Loaded model in eval mode
    """
    if hf_pretrained:
        print("Loading Hugging Face pretrained model...")
        model = generator.Mask2FormerWrapper(
            preseg='hf',
            num_queries=max_instances,
        ).to(device)
    else:
        print(f"Loading model from {model_path}...")
        model = generator.Mask2FormerWrapper(
            preseg=model_path,
            num_queries=max_instances,
        ).to(device)

    model.eval()
    print("Model loaded successfully!")
    return model


@torch.no_grad()
def inference(model, image_tensor, original_size, device, args, label_masks=None):
    """Run inference on a single image.

    Args:
        model: Mask2Former model
        image_tensor: Preprocessed image tensor (1, 1, H, W)
        original_size: (H, W) original image size
        device: torch device

    Returns:
        instance_masks: (N, H, W) numpy array of instance masks
        scores: (N,) confidence scores
    """
    # Move to device
    image_tensor = image_tensor.to(device)

    # Forward pass
    loss = None
    if label_masks is not None:
        mask_list, class_list = label_masks
        mask_list = [m.to(device) for m in mask_list]
        class_list = [c.to(device) for c in class_list]
        outputs = model(image_tensor, mask_list, class_list)
        loss = outputs['loss']
    else:
        outputs = model(image_tensor)

    # Get predictions
    masks_logits = outputs['mask_logits']  # (1, num_queries, H', W')
    class_logits = outputs['class_logits']  # (1, num_queries, num_classes+1)

    # Process outputs
    B, N, H_low, W_low = masks_logits.shape
    H_orig, W_orig = original_size

    # Upsample masks to original size
    masks_logits_upsampled = F.interpolate(
        masks_logits, size=(H_orig, W_orig),
        mode='bilinear', align_corners=False
    )  # (1, num_queries, H_orig, W_orig)

    # Apply sigmoid to get probabilities
    masks_probs = torch.sigmoid(masks_logits_upsampled).squeeze(0)  # (N, H, W)

    # Get class predictions (foreground vs no-object)
    # For single-class instance seg, class 0 = foreground, class 1 = no-object
    class_probs = torch.softmax(class_logits, dim=-1).squeeze(0)  # (N, num_classes+1)
    foreground_scores = class_probs[:, 0]  # (N,) - probability of foreground

    # Filter by score threshold
    valid_mask = foreground_scores > args.score_thresh

    if not valid_mask.any():
        # No valid detections
        return np.zeros((0, H_orig, W_orig)), np.zeros(0)

    # Get valid masks and scores
    valid_masks = masks_probs[valid_mask]  # (M, H, W)
    valid_scores = foreground_scores[valid_mask]  # (M,)

    # Binarize masks
    binary_masks = (valid_masks > args.mask_thresh).float()

    # Filter out masks with very small area (< 0.1% of image)
    min_area = 0.001 * H_orig * W_orig
    mask_areas = binary_masks.sum(dim=(1, 2))
    area_valid = mask_areas > min_area

    if not area_valid.any():
        return np.zeros((0, H_orig, W_orig)), np.zeros(0)

    final_masks = binary_masks[area_valid]  # (K, H, W)
    final_scores = valid_scores[area_valid]  # (K,)

    # Sort by score (descending)
    sorted_indices = torch.argsort(final_scores, descending=True)
    final_masks = final_masks[sorted_indices]
    final_scores = final_scores[sorted_indices]

    # Convert to numpy
    instance_masks = final_masks.cpu().numpy()
    scores = final_scores.cpu().numpy()

    return instance_masks, scores, loss


def visualize_results(original_image, instance_masks, scores, output_path):
    """Visualize and save inference results.

    Args:
        original_image: PIL Image (grayscale)
        instance_masks: (N, H, W) numpy array
        scores: (N,) confidence scores
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Convert grayscale to RGB for better visualization
    img_array = np.array(original_image)
    img_rgb = np.stack([img_array] * 3, axis=-1)

    # Plot 1: Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot 2: Instance masks overlay
    axes[1].imshow(img_rgb)

    # Generate random colors for each instance
    np.random.seed(42)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(instance_masks)))

    # Overlay each instance with a different color
    overlay = np.zeros_like(img_rgb).astype(float)
    for i, (mask, score) in enumerate(zip(instance_masks, scores)):
        color = colors[i, :3]
        for c in range(3):
            overlay[:, :, c] += mask * color[c]

    # Normalize overlay
    if overlay.max() > 0:
        overlay = overlay / overlay.max()

    # Blend with original image
    blended = 0.6 * (img_rgb / 255.0) + 0.4 * overlay
    axes[1].imshow(blended)
    axes[1].set_title(f'Instance Segmentation ({len(instance_masks)} instances)')
    axes[1].axis('off')

    # Plot 3: Individual instances with bounding boxes
    axes[2].imshow(img_array, cmap='gray')

    for i, (mask, score) in enumerate(zip(instance_masks, scores)):
        color = colors[i, :3]

        # Find bounding box
        rows, cols = np.where(mask > 0)
        if len(rows) > 0:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()

            # Draw bounding box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[2].add_patch(rect)

            # Add label
            axes[2].text(
                x_min, y_min - 5, f'#{i+1}: {score:.2f}',
                color=color, fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    axes[2].set_title('Bounding Boxes & Scores')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_instance_masks(instance_masks, scores, output_dir, image_name):
    """Save individual instance masks.

    Args:
        instance_masks: (N, H, W) numpy array
        scores: (N,) confidence scores
        output_dir: Directory to save masks
        image_name: Base name for saved files
    """
    mask_dir = os.path.join(output_dir, 'predicted_masks', image_name)
    os.makedirs(mask_dir, exist_ok=True)

    for i, (mask, score) in enumerate(zip(instance_masks, scores)):
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Save as PNG
        mask_img = Image.fromarray(mask_uint8, mode='L')
        mask_path = os.path.join(mask_dir, f'instance_{i+1:02d}_score{score:.3f}.png')
        mask_img.save(mask_path)


def process_images_on_gpu(gpu_id, image_paths, args, progress_queue=None, loss_list=None):
    """Process a subset of images on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        image_paths: List of image paths to process
        args: Command line arguments
        progress_queue: Optional queue for progress reporting
    
    Returns:
        List of inference results for processed images
    """
    # Set device
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load model on this GPU
    model = load_model(
        args.model_path,
        args.max_instances,
        device,
        args.hf_pretrained
    )
    
    # Process images
    results = []
    for image_path in image_paths:
        try:
            # Load and preprocess
            image_tensor, original_image, original_size = load_and_preprocess_image(
                image_path, args.image_size
            )

            label_masks = None
            if args.label_dir is not None and os.path.isdir(args.label_dir):
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                label_folder = os.path.join(args.label_dir, image_name)
                label_masks = load_instance_masks(
                    label_folder, args.image_size, args.max_instances
                )

            # Run inference
            instance_masks, scores, loss = inference(
                model, image_tensor, original_size, device, args, label_masks=label_masks
            )
            if loss_list is not None and loss is not None:
                loss_list.append(float(loss.item()))
            
            # Save results to list
            image_result = {
                "image_name": os.path.basename(image_path),
                "num_instances": len(instance_masks),
                "scores": scores.tolist() if len(scores) > 0 else []
            }
            if loss is not None:
                image_result["loss"] = float(loss.item())
            results.append(image_result)
            
            # Save results
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Visualize: save to a sibling folder of "masks"
            vis_dir = os.path.join(args.output_dir, 'segmented_visualization')
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f'{image_name}.png')
            visualize_results(original_image, instance_masks, scores, vis_path)
            
            # Save individual masks if requested
            if args.save_masks and len(instance_masks) > 0:
                save_instance_masks(instance_masks, scores, args.output_dir, image_name)
            
            # Report progress
            if progress_queue is not None:
                progress_queue.put(1)
                
        except Exception as e:
            print(f"Error processing {image_path} on GPU {gpu_id}: {str(e)}")
            if progress_queue is not None:
                progress_queue.put(1)
    
    return results



def main():
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of images to process
    image_paths = sorted(
        glob.glob(os.path.join(args.input_dir, '*.png')) +
        glob.glob(os.path.join(args.input_dir, '*.jpg')) +
        glob.glob(os.path.join(args.input_dir, '*.jpeg'))
    )
    
    print(f"Found {len(image_paths)} images to process")
    print(f"Using {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    
    # Distribute images across GPUs
    num_gpus = len(args.gpu_ids)
    images_per_gpu = [[] for _ in range(num_gpus)]
    
    for idx, img_path in enumerate(image_paths):
        gpu_idx = idx % num_gpus
        images_per_gpu[gpu_idx].append(img_path)
    
    # Print distribution
    for gpu_idx, imgs in enumerate(images_per_gpu):
        print(f"GPU {args.gpu_ids[gpu_idx]}: {len(imgs)} images")
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create progress queue for tracking
    manager = mp.Manager()
    progress_queue = manager.Queue()
    loss_list = manager.list() if args.label_dir is not None and os.path.isdir(args.label_dir) else None
    
    # Create processes for each GPU
    processes = []
    for gpu_idx, gpu_id in enumerate(args.gpu_ids):
        if len(images_per_gpu[gpu_idx]) == 0:
            continue
            
        p = mp.Process(
            target=process_images_on_gpu,
            args=(gpu_id, images_per_gpu[gpu_idx], args, progress_queue, loss_list)
        )
        p.start()
        processes.append(p)
    
    # Monitor progress
    total_images = len(image_paths)
    completed = 0
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
        while completed < total_images:
            try:
                progress_queue.get(timeout=1)
                completed += 1
                pbar.update(1)
            except:
                pass
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"\n✓ Multi-GPU inference complete! Results saved to {args.output_dir}")
    print(f"✓ Processed {total_images} images using {len(args.gpu_ids)} GPUs")
    if loss_list is not None and len(loss_list) > 0:
        avg_loss = float(sum(loss_list) / len(loss_list))
        print(f"✓ Average loss over {len(loss_list)} images: {avg_loss:.6f}")


if __name__ == '__main__':
    main()

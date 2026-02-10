"""
使用训练好的生成器(G_B)进行图像生成的推理脚本

该脚本加载预训练的 ResnetGenerator 模型，根据输入的掩码图像和背景图像生成新的图像。
生成过程参考了 USSEGModel 中的 instances_to_image 方法。
支持多GPU并行推理以提高处理速度。

掩码输入格式:
    - 掩码文件夹: --mask_dir path/to/masks/
      其中mask_dir包含多个子文件夹，每个子文件夹包含多个黑白掩码图片
      这些掩码会被合并成(1, N, H, W)的tensor，通过取max生成最终掩码
"""

import os
import argparse
from turtle import back
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.networks import generator
from tqdm import tqdm
from util.load_inference import load_and_preprocess_image, load_and_preprocess_mask_folder

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generator (G_B) Inference for Image Generation')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='预训练生成器模型路径 (e.g., checkpoints/usseg/latest_net_G_B.pth)')
    parser.add_argument('--input_nc', type=int, default=1,
                       help='生成器输入通道数（掩码通道数）')
    parser.add_argument('--output_nc', type=int, default=1,
                       help='生成器输出通道数（图像通道数）')
    parser.add_argument('--ngf', type=int, default=64,
                       help='生成器最后一层卷积的滤波器数量')
    parser.add_argument('--netG', type=str, default='resnet_9blocks',
                       help='生成器架构 [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default='instance',
                       help='归一化层类型 [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal',
                       help='网络初始化方法 [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                       help='初始化缩放因子')

    # 输入/输出参数
    parser.add_argument('--mask_dir', type=str, default=None,
                       help='掩码根目录（包含多个子文件夹，每个子文件夹包含多个黑白掩码图片）')
    parser.add_argument('--background_dir', type=str, default=None,
                       help='背景图像目录（可选，如果不提供则使用纯黑背景）')
    parser.add_argument('--output_dir', type=str, default='./results/generated',
                       help='批量生成时的输出目录')

    # 处理参数
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='逗号分隔的GPU ID列表，用于多GPU并行推理 (例如: "0,1,2,3")')
    parser.add_argument('--image_size', type=int, default=800,
                       help='图像尺寸（图像将被调整为此尺寸）')
    parser.add_argument('--add_noise', action='store_true',
                       help='是否在生成的图像上添加轻微噪声')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='噪声标准差')

    args = parser.parse_args()
    
    # 解析 GPU IDs - 始终使用多GPU模式
    if args.gpu_ids is not None:
        args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else: # 默认：使用所有可用的GPU
        if torch.cuda.is_available():
            args.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            raise RuntimeError("没有可用的CUDA设备。多GPU推理需要至少一个GPU。")
    
    if len(args.gpu_ids) == 0:
        raise RuntimeError("多GPU推理至少需要一个GPU。")

    return args


@torch.no_grad()
def generate_image(net_G_B, mask_tensor, background_tensor, device, args):
    """使用生成器从掩码和背景生成图像
    
    Args:
        net_G_B: 生成器模型
        mask_tensor: 掩码张量 (1, N, H, W) 或 (1, 1, H, W)，范围 [0, 1]
        background_tensor: 背景张量 (1, C, H, W)，范围 [-1, 1]
        device: torch 设备
        args: 命令行参数
        
    Returns:
        generated_image: 生成的图像张量 (1, C, H, W)，范围 [-1, 1]
    """
    # 移动到设备
    mask_tensor = mask_tensor.to(device)
    background_tensor = background_tensor.to(device)
    
    # 如果背景是多通道，转换为单通道（取均值）
    if background_tensor.shape[1] > 1:
        background_tensor = background_tensor.mean(dim=1, keepdim=True)
    
    # 处理掩码
    if mask_tensor.dim() == 4 and mask_tensor.shape[1] > 1:
        mask_merged, _ = torch.max(mask_tensor, dim=1, keepdim=True)
    else:
        mask_merged = mask_tensor
    
    # 将掩码从 [0, 1] 归一化到 [-1, 1]（生成器输入）
    masks_input = mask_merged * 2.0 - 1.0
    
    # 通过生成器生成实例图像
    inst_image = net_G_B(masks_input)
    
    # 可选：添加轻微噪声
    if args.add_noise:
        inst_image = inst_image + torch.randn_like(inst_image) * args.noise_std
    
    # 将生成的实例与背景混合
    mask_expanded = mask_merged.expand_as(background_tensor)
    generated_image = inst_image * mask_expanded + background_tensor * (1 - mask_expanded)
    
    return inst_image, generated_image


def visualize_results(mask_image, background_image, inst_image, generated_image, output_path):
    """可视化并保存生成结果
    
    Args:
        mask_image: PIL 掩码图像
        background_image: PIL 背景图像（可以为 None）
        generated_image: 生成的图像张量 (1, 1, H, W)，范围 [-1, 1]
        output_path: 输出基础路径（用于生成文件名）
    """
    
    # 转换生成的图像到 numpy
    gen_np = generated_image.squeeze().cpu().numpy()
    inst_np = inst_image.squeeze().cpu().numpy()
    # 从 [-1, 1] 转换到 [0, 255]
    gen_np = ((gen_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    inst_np = ((inst_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    # 创建可视化
    num_plots = 3 if background_image is None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # 绘制掩码
    axes[0].imshow(mask_image, cmap='gray')
    axes[0].set_title('input mask')
    axes[0].axis('off')
    
    # 绘制背景
    axes[1].imshow(background_image, cmap='gray')
    axes[1].set_title('background')
    axes[1].axis('off')
    
    axes[2].imshow(inst_np, cmap='gray')
    axes[2].set_title('generated all')
    axes[2].axis('off')
    
    # 绘制生成的图像
    axes[3].imshow(gen_np, cmap='gray')
    axes[3].set_title('Generated Image')
    axes[3].axis('off')
    
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    viz_dir = os.path.join(base_dir, "generated_visualization")
    gen_dir = os.path.join(base_dir, "images")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    viz_path = os.path.join(viz_dir, f"{base_name}.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 同时保存单独的生成图像
    generated_only_path = os.path.join(gen_dir, f"{base_name}.png")
    gen_pil = Image.fromarray(gen_np, mode='L')
    gen_pil.save(generated_only_path)


def process_images_on_gpu(gpu_id, image_pairs, args, progress_queue=None):
    """在指定GPU上处理一组图像
    
    Args:
        gpu_id: GPU设备ID
        image_pairs: 要处理的图像对列表 [(mask_path, background_path, output_path), ...]
        args: 命令行参数
        progress_queue: 可选的进度报告队列
    
    Returns:
        处理结果列表
    """
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}')
    
    # 在此GPU上加载模型
    net_G_B = generator.define_G(args.model_path,
        args.input_nc, args.output_nc, args.ngf, args.netG, args.norm,
        use_dropout=False, init_type=args.init_type, init_gain=args.init_gain,
        gpu_ids=gpu_id, thres=True, distributed=False
    ).to(device)
    net_G_B.eval()
    
    # 处理图像
    results = []
    for mask_path, background_path, output_path in image_pairs:
        try:
            # 掩码文件夹：加载所有掩码并合并
            mask_tensor, mask_image = load_and_preprocess_mask_folder(
                mask_path, args.image_size
            )
            # 加载并预处理背景
            background_tensor, background_image, _ = load_and_preprocess_image(
                background_path, args.image_size
            )
            # 生成图像
            inst_image, generated_image = generate_image(
                net_G_B, mask_tensor, background_tensor, device, args
            )
            
            # 可视化并保存结果
            visualize_results(mask_image, background_image, inst_image, generated_image, output_path)
            
            results.append({
                "image_name": os.path.basename(mask_path),
                "success": True
            })
            
            # 报告进度
            if progress_queue is not None:
                progress_queue.put(1)
                
        except Exception as e:
            import traceback
            error_msg = f"GPU {gpu_id} 处理 {mask_path} 时出错: {str(e)}"
            print(error_msg)
            print(f"详细错误信息:\n{traceback.format_exc()}")
            results.append({
                "image_name": os.path.basename(mask_path),
                "success": False,
                "error": str(e)
            })
            if progress_queue is not None:
                progress_queue.put(1)
    
    return results


def main():
    args = get_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 批量处理：获取mask_dir下的所有子文件夹
    mask_paths = []
    for item in sorted(os.listdir(args.mask_dir)):
        item_path = os.path.join(args.mask_dir, item)
        if os.path.isdir(item_path):
            mask_paths.append(item_path)
    
    background_paths = []
    for item in sorted(os.listdir(args.background_dir)):
        item_path = os.path.join(args.background_dir, item)
        background_paths.append(item_path)
    
    print(f"找到 {len(mask_paths)} 个掩码文件夹")
    print(f"找到 {len(background_paths)} 个背景图片")
    print(f"使用 {len(args.gpu_ids)} 个GPU: {args.gpu_ids}")
    
    # 准备图像对列表 (mask_path, background_path, output_path)
    image_pairs = []
    for mask_path in mask_paths:
        background_path = background_paths[np.random.randint(0, len(background_paths))]
        output_path = os.path.join(args.output_dir, os.path.basename(mask_path) + ".png")
        image_pairs.append((mask_path, background_path, output_path))
    # 将图像分配到不同的GPU
    num_gpus = len(args.gpu_ids)
    images_per_gpu = [[] for _ in range(num_gpus)]
    
    for idx, img_pair in enumerate(image_pairs):
        gpu_idx = idx % num_gpus
        images_per_gpu[gpu_idx].append(img_pair)
    
    # 打印分配情况
    for gpu_idx, imgs in enumerate(images_per_gpu):
        print(f"GPU {args.gpu_ids[gpu_idx]}: {len(imgs)} 张图像")
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 创建进度队列用于追踪
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    # 为每个GPU创建进程
    processes = []
    for gpu_idx, gpu_id in enumerate(args.gpu_ids):
        if len(images_per_gpu[gpu_idx]) == 0:
            continue
            
        p = mp.Process(
            target=process_images_on_gpu,
            args=(gpu_id, images_per_gpu[gpu_idx], args, progress_queue)
        )
        p.start()
        processes.append(p)
    
    # 监控进度
    total_images = len(image_pairs)
    completed = 0
    
    with tqdm(total=total_images, desc="生成图像") as pbar:
        while completed < total_images:
            try:
                progress_queue.get(timeout=1)
                completed += 1
                pbar.update(1)
            except:
                pass
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print(f"\n✓ 多GPU图像生成完成! 结果已保存到 {args.output_dir}")
    print(f"✓ 使用 {len(args.gpu_ids)} 个GPU处理了 {total_images} 张图像")


if __name__ == '__main__':
    main()

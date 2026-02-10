import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def load_and_preprocess_image(image_path, image_size):
    """加载并预处理图像

    Args:
        image_path: 图像路径
        image_size: 目标尺寸

    Returns:
        image_tensor: 预处理后的图像张量 (1, 1, H, W)
        original_image: 原始 PIL 图像
        original_size: (H, W) 原始尺寸
    """
    original_image = Image.open(image_path).convert('L')
    original_size = original_image.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(original_image).unsqueeze(0)
    return image_tensor, original_image, (original_size[1], original_size[0])


def load_and_preprocess_mask_folder(mask_folder, image_size):
    """加载文件夹中的所有掩码图像并合并成多通道tensor

    Args:
        mask_folder: 掩码文件夹路径（包含多个黑白掩码图片）
        image_size: 目标尺寸

    Returns:
        mask_tensor: 合并后的掩码张量 (1, N, H, W)
        preview_image: 预览的合并掩码图像（取max）
    """
    mask_files = sorted(
        glob.glob(os.path.join(mask_folder, '*.png')) +
        glob.glob(os.path.join(mask_folder, '*.jpg')) +
        glob.glob(os.path.join(mask_folder, '*.jpeg'))
    )

    if len(mask_files) == 0:
        raise ValueError(f"文件夹 {mask_folder} 中没有找到掩码图像")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    mask_list = []
    for mask_file in mask_files:
        mask_img = Image.open(mask_file).convert('L')
        mask_tensor = transform(mask_img)
        mask_tensor = (mask_tensor >= 0.5).float()
        mask_list.append(mask_tensor)

    masks_stacked = torch.stack(mask_list, dim=0)  # (N, 1, H, W)
    masks_stacked = masks_stacked.squeeze(1)  # (N, H, W)
    mask_tensor = masks_stacked.unsqueeze(0)  # (1, N, H, W)

    max_mask = torch.max(masks_stacked, dim=0)[0]
    preview_np = (max_mask.numpy() * 255).astype(np.uint8)
    preview_image = Image.fromarray(preview_np, mode='L')

    return mask_tensor, preview_image


def load_instance_masks(mask_folder, image_size, max_instances):
    """加载实例掩码并返回 mask_list, class_list

    Args:
        mask_folder: 包含实例掩码的文件夹
        image_size: 目标尺寸
        max_instances: 最大实例数量

    Returns:
        mask_list: list[Tensor]，长度为1，每个元素是 (N_i, H, W)
        class_list: list[Tensor]，长度为1，每个元素是 (N_i,)
    """
    if not os.path.exists(mask_folder):
        return [torch.zeros(0, image_size, image_size)], [torch.zeros(0, dtype=torch.int64)]

    mask_files = sorted(
        glob.glob(os.path.join(mask_folder, '*.png')) +
        glob.glob(os.path.join(mask_folder, '*.jpg')) +
        glob.glob(os.path.join(mask_folder, '*.jpeg'))
    )

    if len(mask_files) == 0:
        return [torch.zeros(0, image_size, image_size)], [torch.zeros(0, dtype=torch.int64)]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    masks = []
    for mask_file in mask_files[:max_instances]:
        mask_img = Image.open(mask_file).convert('L')
        mask_tensor = transform(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()
        masks.append(mask_tensor.squeeze(0))

    masks_tensor = torch.stack(masks, dim=0) if len(masks) > 0 else torch.zeros(0, image_size, image_size)
    num_valid = masks_tensor.shape[0]
    class_labels = torch.zeros(num_valid, dtype=torch.int64)
    return [masks_tensor], [class_labels]

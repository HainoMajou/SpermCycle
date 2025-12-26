import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import glob
import numpy as np
import torch
import torchvision.transforms as transforms


class USSEGDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A_instance')  # 存储实例掩码文件夹的路径
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # 真实图像路径
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')  # 背景图像路径
        
        # A_paths 现在是文件夹列表，每个文件夹包含多个实例掩码
        self.A_paths = sorted([d for d in glob.glob(os.path.join(self.dir_A, '*')) if os.path.isdir(d)])
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        
        self.max_instances = getattr(opt, 'max_instances', 10)
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        
        # 图像变换
        self.transform_A = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_C = get_transform(self.opt, grayscale=(output_nc == 1))

    def _load_instance_masks(self, folder_path):
        """从文件夹加载所有实例掩码
        Returns:
            Tensor (N, H, W), N = max_instances，不足的用 0 填充
        """
        # 获取文件夹中所有图像文件
        mask_files = sorted(glob.glob(os.path.join(folder_path, '*.png')) + 
                           glob.glob(os.path.join(folder_path, '*.jpg')) +
                           glob.glob(os.path.join(folder_path, '*.jpeg')))
        
        masks = []
        for mask_file in mask_files[:self.max_instances]:  # 最多读取 max_instances 个
            mask_img = Image.open(mask_file).convert('L')  # 转为灰度图
            mask_tensor = self.transform_A(mask_img)  # (1, H, W)
            # 二值化
            mask_tensor = (mask_tensor > 0.5).float()
            masks.append(mask_tensor.squeeze(0))  # (H, W)
        
        if len(masks) == 0:
            # 如果没有掩码，返回全零张量
            H = W = self.opt.load_size
            return torch.zeros(self.max_instances, H, W)
        
        # 堆叠所有掩码
        masks_tensor = torch.stack(masks, dim=0)  # (N_actual, H, W)
        N_actual, H, W = masks_tensor.shape
        
        # 填充到 max_instances
        if N_actual < self.max_instances:
            padding = torch.zeros(self.max_instances - N_actual, H, W)
            masks_tensor = torch.cat([masks_tensor, padding], dim=0)
        
        return masks_tensor  # (max_instances, H, W)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        # A: 实例掩码文件夹
        A_path = self.A_paths[index % self.A_size]
        C_path = self.C_paths[index % self.C_size]
        
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # 加载图像
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        
        # 加载实例掩码（从文件夹中）
        A = self._load_instance_masks(A_path)  # (N, H, W)
        
        # 应用图像变换
        B = self.transform_B(B_img)
        C = self.transform_C(C_img)
        
        return {
            'A': A,           # (N, H, W), 实例掩码
            'B': B,           # (C, H, W), 真实图像
            'C': C,           # (C, H, W), 背景图像
            'A_paths': A_path,
            'B_paths': B_path,
            'C_paths': C_path
        }

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)

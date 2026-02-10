import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import glob
import torch


class InstanceSegDataset(BaseDataset):
    """实例分割数据集

    数据集结构:
        dataroot/
            images/              # 灰度图像文件夹
                img001.png
                img002.png
                ...
            masks/               # 实例掩码文件夹
                img001/          # 与 images 中的文件名对应
                    instance_01.png   # 实例1的二值掩码
                    instance_02.png   # 实例2的二值掩码
                    ...
                img002/
                    instance_01.png
                    ...
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)

        # 图像和掩码文件夹路径
        self.dir_images = os.path.join(opt.dataroot, 'images')
        self.dir_masks = os.path.join(opt.dataroot, 'masks')

        # 获取所有图像路径
        self.image_paths = sorted(make_dataset(self.dir_images, opt.max_dataset_size))

        self.max_instances = getattr(opt, 'max_instances', 10)

        # 图像变换（灰度图）
        self.transform = get_transform(self.opt, grayscale=True)

    def _get_mask_folder_path(self, image_path):
        """根据图像路径获取对应的掩码文件夹路径

        Args:
            image_path: 图像文件路径，如 dataroot/images/img001.png

        Returns:
            mask_folder: 掩码文件夹路径，如 dataroot/masks/img001/
        """
        # 获取图像文件名（不含扩展名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # 构建掩码文件夹路径
        mask_folder = os.path.join(self.dir_masks, image_name)
        return mask_folder

    def _load_instance_masks(self, mask_folder):
        """从文件夹加载所有实例掩码

        Args:
            mask_folder: 包含实例掩码的文件夹路径

        Returns:
            Tensor (N, H, W), N = max_instances，不足的用 0 填充
        """
        # 检查文件夹是否存在
        if not os.path.exists(mask_folder):
            # 如果文件夹不存在，返回全零张量
            H = W = self.opt.load_size
            return torch.zeros(self.max_instances, H, W)

        # 获取文件夹中所有图像文件（支持 png, jpg, jpeg）
        mask_files = sorted(
            glob.glob(os.path.join(mask_folder, '*.png')) +
            glob.glob(os.path.join(mask_folder, '*.jpg')) +
            glob.glob(os.path.join(mask_folder, '*.jpeg'))
        )

        masks = []
        for mask_file in mask_files[:self.max_instances]:  # 最多读取 max_instances 个
            mask_img = Image.open(mask_file).convert('L')  # 转为灰度图
            mask_tensor = self.transform(mask_img)  # (1, H, W)
            # 二值化（阈值 0.5）
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
            dict: {
                'A': Tensor (C, H, W) - 输入灰度图像
                'B': Tensor (N, H, W) - 目标实例掩码（N = max_instances）
                'A_paths': str - 图像路径
                'B_paths': str - 掩码文件夹路径
            }
        """
        # 获取图像路径
        image_path = self.image_paths[index % len(self.image_paths)]

        # 获取对应的掩码文件夹路径
        mask_folder = self._get_mask_folder_path(image_path)

        # 加载图像（灰度）
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image)  # (1, H, W)

        # 加载实例掩码
        masks_tensor = self._load_instance_masks(mask_folder)  # (N, H, W)

        return {
            'B': image_tensor,      # (1, H, W), 输入灰度图像
            'A': masks_tensor,      # (N, H, W), 目标实例掩码
            'B_paths': image_path,
            'A_paths': mask_folder
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

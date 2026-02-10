import numpy as np
import os
import time
import torch
from . import util


def is_instance_mask_tensor(data):
    """判断数据是否为实例掩码张量 (B, N, H, W)

    区分实例掩码 (B, N, H, W) 和普通图像 (B, C, H, W):
    - 实例掩码：N > 3 或 (N > 1 且值在[0,1]范围)
    - 普通图像：C = 1 或 3
    """
    if not isinstance(data, torch.Tensor):
        return False
    if data.dim() != 4:
        return False

    B, N, H, W = data.shape

    # 如果通道数 > 3，肯定是实例掩码（不是RGB或灰度图）
    if N > 3:
        return True

    # 如果通道数为1或3，很可能是普通图像
    if N == 1 or N == 3:
        return False

    return False


class Visualizer():
    """This class includes functions to save images and print/save logging information."""

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.name = opt.name
        self.saved = False

        # create image directory for saving images
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        print('create image directory %s...' % self.img_dir)
        util.mkdirs([self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Save current results to disk.

        Parameters:
            visuals (OrderedDict) -- dictionary of images to save
            epoch (int) -- the current epoch
            save_result (bool) -- if save the current results to disk
        """
        # Save images to disk
        if save_result or not self.saved:
            self.saved = True

            # 创建当前epoch的文件夹
            epoch_dir = os.path.join(self.img_dir, 'epoch%.3d' % epoch)
            util.mkdirs([epoch_dir])

            # save images to the disk
            for label, image_data in visuals.items():
                # 通过张量维度和形状判断是否为实例掩码 (B, N, H, W)
                # 不依赖变量名，只看数据的实际形状
                is_instance_mask = is_instance_mask_tensor(image_data)

                if is_instance_mask:
                    # 为实例掩码创建子文件夹（在epoch文件夹下）
                    mask_folder = os.path.join(epoch_dir, label)
                    util.mkdirs([mask_folder])

                    # 保存N张灰度图
                    instances = image_data[0]  # (N, H, W)
                    N = instances.shape[0]
                    for i in range(N):
                        mask = instances[i].detach().cpu().float().numpy()

                        # 跳过空掩码（全零或接近全零）
                        if mask.max() < 0.01 and mask.min() > -0.01:
                            continue

                        # 归一化到 [0, 1]
                        if mask.min() < 0:
                            # 如果是 [-1, 1] 范围，归一化到 [0, 1]
                            mask = (mask + 1) / 2
                        
                        # 确保在 [0, 1] 范围内
                        mask = np.clip(mask, 0, 1)

                        # 转换为 [0, 255] 的灰度图
                        mask_uint8 = (mask * 255).astype(np.uint8)

                        # 保存灰度图（单通道）
                        img_path = os.path.join(mask_folder, 'inst%d.png' % i)
                        from PIL import Image
                        img_pil = Image.fromarray(mask_uint8, mode='L')
                        img_pil.save(img_path)
                else:
                    # 普通图像（real_B, fake_B, rec_B, real_C 等），保存在epoch文件夹下
                    if isinstance(image_data, np.ndarray):
                        image_numpy = image_data
                    else:
                        # 检查是否为单通道图像 (B, 1, H, W)
                        if isinstance(image_data, torch.Tensor) and image_data.dim() == 4 and image_data.shape[1] == 1:
                            # 单通道图像（B, 1, H, W），保存为灰度图
                            image_numpy = util.tensor2im(image_data, convert_grayscale_to_rgb=False)
                        else:
                            # 多通道图像，正常转换为RGB
                            image_numpy = util.tensor2im(image_data, convert_grayscale_to_rgb=True)
                    img_path = os.path.join(epoch_dir, '%s.png' % label)
                    util.save_image(image_numpy, img_path)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

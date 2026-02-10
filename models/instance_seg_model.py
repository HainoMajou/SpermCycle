import torch
import torch.nn.functional as F
from .base_model import BaseModel
from .networks import generator
from torch.nn.parallel import DistributedDataParallel as DDP

def tensor2list(masks):
    """
    将批量掩码张量转换为列表，过滤掉padding的全零掩码
    Args:
        masks: shape为(B, N, H, W)的张量
    Returns:
        长度为B的列表，每个元素是shape为(N_i, H, W)的张量
        其中N_i是该图片中非全零掩码的数量
    """
    B, N, H, W = masks.shape
    mask_list = []
    class_list = []

    for i in range(B):
        single_image_masks = masks[i]  # shape: (N, H, W)
        non_zero_mask = single_image_masks.sum(dim=(1, 2)) > 0  # shape: (N,)
        valid_masks = single_image_masks[non_zero_mask]  # shape: (N_i, H, W)
        mask_list.append(valid_masks)

        # 为每个有效掩码创建类别标签（全部为0，表示前景）
        num_valid = valid_masks.shape[0]
        class_labels = torch.zeros(num_valid, dtype=torch.int64, device=masks.device)
        class_list.append(class_labels)

    return mask_list, class_list

class InstanceSegModel(BaseModel):
    """
    纯实例分割训练模型 (使用 Mask2Former + InstanceCycleLoss)

    网络定义:
        - netG: Mask2FormerWrapper - 图像 → 实例掩码 (B, N, H, W)

    损失:
        - InstanceCycleLoss: 匹配预测和目标实例，计算 Dice + BCE + Class 损失
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='template')
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_seg', type=float, default=1.0, help='weight for segmentation loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['seg']
        self.visual_names = ['real_A', 'real_B', 'pred_A']

        if self.isTrain:
            self.model_names = ['G']

        distributed = getattr(opt, 'distributed', False)
        self.max_instances = getattr(opt, 'max_instances', 10)
        self.preseg = getattr(opt, 'preseg', None)
        # 直接实例化 Mask2FormerWrapper
        self.netG = generator.Mask2FormerWrapper(
            preseg=self.preseg,
            num_queries=self.max_instances
        )

        # 移动到 GPU 并包装为 DDP
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.to(self.gpu_ids[0])
            if distributed and torch.distributed.is_initialized():
                self.netG = DDP(self.netG, device_ids=[self.gpu_ids[0]], find_unused_parameters=True)
            elif len(self.gpu_ids) > 1:
                self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if self.isTrain:
            # Mask2Former: AdamW + weight decay
            self.optimizer_G = torch.optim.AdamW(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=0.01
            )

            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """设置输入
        input['A']: (B, N, H, W) - 目标实例掩码（N个实例，不足的用0填充）
        input['B']: (B, C, H, W) - 输入图像
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """前向传播"""
        B, N, H, W = self.real_A.shape
        mask_list, class_list = tensor2list(self.real_A)        
        outputs = self.netG(self.real_B, mask_list, class_list)
        self.loss_seg = outputs['loss']
        masks_logits = outputs['mask_logits']

        B, N, H_low, W_low = masks_logits.shape
        if (H_low, W_low) != (H, W):
            pred_A_logits = F.interpolate(
                masks_logits, size=(H, W), mode='bilinear', align_corners=False
            ) 
        else:
            pred_A_logits = masks_logits
        self.pred_A = torch.sigmoid(pred_A_logits)

    def optimize_parameters(self):
        """优化参数"""
        self.optimizer_G.zero_grad()

        if self.use_amp: # 使用混合精度训练
            with torch.amp.autocast("cuda"):
                self.forward()

            self.scaler.scale(self.loss_seg).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer_G)
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)

            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            # FP32 训练
            self.forward()
            self.loss_seg.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)

            self.optimizer_G.step()

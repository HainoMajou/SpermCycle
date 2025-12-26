import torch
import torch.nn.functional as F
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class USSEGModel(BaseModel):
    """
    实例分割 CycleGAN 模型 (使用 Mask2Former)
    
    张量定义:
        实例掩码张量 (A域):
            - real_A: Tensor (B, N, H, W), N 为 batch 中最大实例数，不足的用 0 填充
            - fake_A: Tensor (B, N, H, W), G_A 从图像生成的实例掩码
            - rec_A: Tensor (B, N, H, W), 循环重建的实例掩码
            
        图像张量 (B域):
            - real_B: Tensor (B, C, H, W), 真实图像
            - fake_B: Tensor (B, C, H, W), G_B 从掩码生成的图像
            - rec_B: Tensor (B, C, H, W), 循环重建的图像
            - real_C: Tensor (B, C, H, W), 背景图像
    
    网络定义:
        - G_A (Mask2Former): 图像 → 实例掩码 (B, N, H, W)
        - G_B: 单个实例掩码 → 图像（对每个实例分别生成后叠加）
        - D_B: 判别图像真假
    
    循环一致性:
        路径1: real_B → G_A → fake_A → G_B → rec_B ≈ real_B
        路径2: real_A → G_B → fake_B → G_A → rec_A ≈ real_A
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='template')
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--score_thresh', type=float, default=0.5, help='score threshold for detections')
            parser.add_argument('--mask_thresh', type=float, default=0.5, help='threshold for binarizing masks')
            parser.add_argument('--lr_G_A', type=float, default=0.0001, help='learning rate for Mask2Former')
            parser.add_argument('--lr_G_B', type=float, default=0.001, help='learning rate for Generator')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # 损失名称
        self.loss_names = ['cycle_A', 'D_B', 'G_B', 'cycle_B', 'G']
        
        # 可视化张量名称
        visual_names_A = ['real_A', 'fake_A', 'rec_A']
        visual_names_B = ['real_B', 'fake_B', 'rec_B', 'real_C']
        self.visual_names = visual_names_A + visual_names_B
        
        # 模型名称
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        distributed = getattr(opt, 'distributed', False)
        self.score_thresh = getattr(opt, 'score_thresh', 0.5)
        self.mask_thresh = getattr(opt, 'mask_thresh', 0.5)
        self.max_instances = getattr(opt, 'max_instances', 10)
        
        # G_A: Mask2Former (图像 → 实例掩码)
        self.netG_A = networks.define_Mask2Former(
            pretrained=False, gpu_ids=self.gpu_ids,
            distributed=distributed, num_queries=self.max_instances,
            score_thresh=self.score_thresh, mask_thresh=self.mask_thresh
        )
        
        # G_B: 单个实例掩码 → 图像
        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
            thres=True, distributed=distributed
        )
        
        if self.isTrain:
            # D_B: 判别图像
            self.netD_B = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids, distributed=distributed
            )
            
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            # 损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionInstanceCycle = networks.InstanceCycleLoss(
                cost_mask=5.0,      # 匹配时 BCE 权重（Mask2Former 默认 5.0）
                cost_dice=5.0,      # 匹配时 Dice 权重（Mask2Former 默认 5.0）
                lambda_mask=5.0,    # 最终损失 BCE 权重（Mask2Former 默认 5.0）
                lambda_dice=5.0,    # 最终损失 Dice 权重（Mask2Former 默认 5.0）
                lambda_class=2.0,   # 最终损失类别权重（区分前景/no-object，Mask2Former 默认 2.0）
                num_points=12544    # 采样点数（112x112）
            )
            
            lr_G_A = getattr(opt, 'lr_G_A', opt.lr * 0.1)
            lr_G_B = getattr(opt, 'lr_G_B', opt.lr)
            
            # Mask2Former: 小学习率 + AdamW
            self.optimizer_G_A = torch.optim.AdamW(
                self.netG_A.parameters(),
                lr=lr_G_A,
                betas=(opt.beta1, 0.999),
                weight_decay=0.01
            )
            
            # ResnetGenerator: 正常学习率 + Adam
            self.optimizer_G_B = torch.optim.Adam(
                self.netG_B.parameters(),
                lr=lr_G_B,
                betas=(opt.beta1, 0.999)
            )
            
            # Discriminator
            self.optimizer_D = torch.optim.Adam(
                self.netD_B.parameters(), 
                lr=opt.lr, 
                betas=(opt.beta1, 0.999)
            )
            
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """设置输入张量"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # print(self.real_A.shape, self.real_B.shape, self.real_C.shape)

    def _instances_to_image(self, instances, background):
        """将实例掩码张量通过 G_B 转换为图像张量
            instance (B, N, H, W)
            background (B, C, H, W), C = 3 or 1    
        """
        if background.shape[1]>1:
            background = background.mean(dim=1, keepdim=True)
        B, C, H, W = background.shape
        N = instances.shape[1]
        
        masks_flat = instances.view(B * N, 1, H, W)
        masks_input = masks_flat * 2 - 1
        
        with torch.set_grad_enabled(self.isTrain):
            inst_images_flat = self.netG_B(masks_input)
        
        inst_images = inst_images_flat.view(B, N, C, H, W)
        masks_expanded = instances.unsqueeze(2).expand(-1, -1, C, -1, -1)
        
        image = background.clone()
        for i in range(N):
            mask_i = masks_expanded[:, i, :, :, :]
            inst_i = inst_images[:, i, :, :, :]
            image = image * (1 - mask_i) + inst_i * mask_i
        
        return image

    def forward(self):
        """统一前向传播（推理时使用）"""
        self.forward_path_BAB()
        self.forward_path_ABA()

    def forward_path_BAB(self):
        """路径1: real_B → G_A → fake_A → G_B → rec_B"""
        B, C, H, W = self.real_B.shape
        fake_A_outputs = self.netG_A(self.real_B)
        masks = fake_A_outputs['mask_logits']  # (B, num_queries, H', W')
        self.fake_A_classes = fake_A_outputs['class_logits']  # (B, num_queries, num_classes + 1)

        B, N, H_low, W_low = masks.shape
        if (H_low, W_low) != (H, W):
            self.fake_A = F.interpolate(
                masks, size=(H, W), mode='bilinear', align_corners=False
            ) 
        else:
            self.fake_A = masks

        self.rec_B = self._instances_to_image(self.fake_A, self.real_C)

    def forward_path_ABA(self):
        """路径2: real_A → G_B → fake_B → G_A → rec_A"""
        B, N, H, W = self.real_A.shape
        self.fake_B = self._instances_to_image(self.real_A, self.real_C)

        rec_A_outputs = self.netG_A(self.fake_B)
        masks = rec_A_outputs['mask_logits']  # (B, num_queries, H', W')
        self.rec_A_classes = rec_A_outputs['class_logits']  # (B, num_queries, num_classes+1)

        B, N, H_low, W_low = masks.shape
        if (H_low, W_low) != (H, W):
            self.rec_A = F.interpolate(
                masks, size=(H, W), mode='bilinear', align_corners=False
            ) 
        else:
            self.rec_A = masks

    def optimize_parameters(self):
        # 1. 清零所有梯度
        self.set_requires_grad([self.netD_B], False)
        self.optimizer_G_A.zero_grad()
        self.optimizer_G_B.zero_grad()

        # 检查是否使用DDP
        use_ddp = (self.opt.distributed and hasattr(self.netG_A, 'no_sync') and hasattr(self.netG_B, 'no_sync'))

        # 2. 路径1：real_B → G_A → fake_A → G_B → rec_B
        self.forward_path_BAB()
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        loss_path1 = self.loss_cycle_B 
        
        if use_ddp: # DDP模式：第一次backward不同步梯度（节省通信开销）
            with self.netG_A.no_sync(), self.netG_B.no_sync():
                loss_path1.backward(retain_graph=False)
        else: # 非DDP模式：正常backward
            loss_path1.backward(retain_graph=False)
        
        # 保存路径1结果用于可视化
        fake_A_vis = self.fake_A.detach()
        rec_B_vis = self.rec_B.detach()

        # 3. 路径2：real_A → G_B → fake_B → G_A → rec_A
        self.forward_path_ABA()
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.loss_cycle_A = self.criterionInstanceCycle(self.rec_A, self.real_A, self.rec_A_classes) * self.opt.lambda_A
        
        loss_path2 = self.loss_G_B + self.loss_cycle_A
        loss_path2.backward(retain_graph=False) # 第二次backward会同步所有累积的梯度（DDP自动处理）

        # 4. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), 5.0)
        
        # 5. 更新参数
        self.optimizer_G_A.step()
        self.optimizer_G_B.step()

        # 恢复可视化张量
        self.fake_A = fake_A_vis
        self.rec_B = rec_B_vis
        self.loss_G = (self.loss_cycle_B + self.loss_G_B + self.loss_cycle_A).detach()        

        self.set_requires_grad([self.netD_B], True)
        self.optimizer_D.zero_grad()
        # 使用 detach 的 fake_B 来训练判别器
        fake_B = self.fake_B_pool.query(self.fake_B.detach())

        # 判别真实图像
        pred_real = self.netD_B(self.real_B)
        loss_D_real = self.criterionGAN(pred_real, True)
        # 判别生成图像
        pred_fake = self.netD_B(fake_B)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D_B.backward()
        self.optimizer_D.step()

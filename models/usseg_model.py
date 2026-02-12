import torch
import torch.nn.functional as F
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from .networks import generator, discriminator
from torchvision.transforms.functional import gaussian_blur

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

class USSEGModel(BaseModel):
    """
    实例分割 CycleGAN 模型 (使用 Mask2Former)
    
    张量定义:
        实例掩码张量 (A域):
            - real_A: Tensor (B, N, H, W), N 为最大的掩码标签数量
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
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_GB', type=float, default=1.0, help='weight for G_B loss')
            parser.add_argument('--lambda_DB', type=float, default=1.0, help='weight for D_B_loss')
            parser.add_argument('--lambda_GA', type=float, default=1.0, help='weight for G_A_loss')
            parser.add_argument('--lambda_DA', type=float, default=1.0, help='weight for D_A_loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # 损失名称
        self.loss_names = ['cycle_A', 'D_A', 'G_A', 'D_B', 'G_B', 'cycle_B', 'G', 'D']
        
        # 可视化张量名称
        visual_names_A = ['real_A', 'fake_A', 'rec_A']
        visual_names_B = ['real_B', 'fake_B', 'rec_B', 'real_C']
        self.visual_names = visual_names_A + visual_names_B
        
        # 模型名称
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        distributed = getattr(opt, 'distributed', False)
        self.max_instances = getattr(opt, 'max_instances', 10)
        self.preseg = getattr(opt, 'preseg', None)
        self.pregen = getattr(opt, 'pregen', None)
        
        # G_A: Mask2Former (图像 → 实例掩码)
        pre_G_A = f'{self.preseg}_net_G_A.pth' if self.preseg else None
        self.netG_A = generator.define_Mask2Former(
            pre_G_A, self.max_instances,  
            opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids, distributed=distributed
        )
        
        # G_B: 单个实例掩码 → 图像
        pre_G_B = f'{self.pregen}_net_G_B.pth' if self.pregen else None
        self.netG_B = generator.define_G(
            pre_G_B, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
            thres=True, distributed=distributed
        )
        
        if self.isTrain:
            # D_A: 判别实例掩码 (输入通道数为 1，因为每个掩码是单通道)
            pre_D_A = f'{self.preseg}_net_D_A.pth' if self.preseg else None
            self.netD_A = discriminator.define_D(
                pre_D_A, 1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids, distributed=distributed
            )
            # D_B: 判别图像
            pre_D_B = f'{self.pregen}_net_D_B.pth' if self.pregen else None
            self.netD_B = discriminator.define_D(
                pre_D_B, opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids, distributed=distributed
            )

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            self.criterionGAN = discriminator.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            # self.criterionInstanceCycle = networks.InstanceCycleLoss()
            
            lr_G_A = opt.lr * 0.5
            lr_G_B = opt.lr
            lr_D_A = opt.lr * 0.5
            lr_D_B = opt.lr * 0.5
            
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
            
            # Discriminators
            self.optimizer_D_A = torch.optim.Adam(
                self.netD_A.parameters(), 
                lr=lr_D_A, 
                betas=(opt.beta1, 0.999)
            )
            
            self.optimizer_D_B = torch.optim.Adam(
                self.netD_B.parameters(), 
                lr=lr_D_B, 
                betas=(opt.beta1, 0.999)
            )
            
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)

    def set_input(self, input):
        """设置输入张量"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # {0, 1}
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # [-1, 1]
        self.real_C = input['C'].to(self.device) # [-1, 1]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.shape_A = self.real_A.shape

    def instances_to_image(self, instances, background, blur_kernel_size=[5, 5], blur_sigma=[1.0, 1.0], dilation_size=1):
        """将实例掩码张量通过 G_B 转换为图像张量
            instance (B, N, H, W)
            background (B, C, H, W), C = 3 or 1    
        """
        if background.shape[1]>1:
            background = background.mean(dim=1, keepdim=True)
        
        mask_merged, _ = torch.max(instances, dim=1, keepdim=True)
        masks_input = mask_merged * 2 - 1 # [-1, 1]
        with torch.set_grad_enabled(self.isTrain):
            inst_image = self.netG_B(masks_input)
            
        # inst_image = inst_image + torch.randn_like(inst_image) * 0.01 
        mask_expanded = mask_merged.expand_as(background)
        image = inst_image * mask_expanded + background * (1 - mask_expanded)

        dilated_mask = F.max_pool2d(
            mask_expanded, 
            kernel_size=2*dilation_size+1, 
            stride=1, 
            padding=dilation_size
        )
        # 对整个图像进行高斯模糊
        blurred_image = gaussian_blur(image, kernel_size=blur_kernel_size, sigma=blur_sigma)
        # 只在膨胀后的mask区域使用模糊图像
        image = blurred_image * dilated_mask + image * (1 - dilated_mask)
        '''
        B, C, H, W = background.shape
        N = instances.shape[1]

        masks_flat = instances.view(B * N, 1, H, W) # [0, 1]
        masks_input = masks_flat * 2 - 1            # [-1, 1]
        
        with torch.set_grad_enabled(self.isTrain):
            inst_images_flat = self.netG_B(masks_input)
        
        inst_images = inst_images_flat.view(B, N, C, H, W)
        masks_expanded = instances.unsqueeze(2).expand(-1, -1, C, -1, -1)
        
        image = background.clone()
        for i in range(N):
            mask_i = masks_expanded[:, i, :, :, :]
            inst_i = inst_images[:, i, :, :, :]
            image = image * (1 - mask_i) + inst_i * mask_i
        '''
        return image

    def forward(self):
        """统一前向传播（推理时使用）"""
        self.forward_path_BAB()
        self.forward_path_ABA()

    def forward_path_BAB(self):
        """路径1: real_B → G_A → fake_A → G_B → rec_B"""
        B, C, H, W = self.real_B.shape
        fake_A_outputs = self.netG_A(self.real_B)
        masks_logits = fake_A_outputs['mask_logits']  # (B, num_queries, H', W')
        self.fake_A_classes = fake_A_outputs['class_logits']  # (B, num_queries, num_classes + 1)

        B, N, H_low, W_low = masks_logits.shape
        if (H_low, W_low) != (H, W):
            self.fake_A_logits = F.interpolate(
                masks_logits, size=(H, W), mode='bilinear', align_corners=False
            ) 
        else:
            self.fake_A_logits = masks_logits
        
        self.fake_A = torch.sigmoid(self.fake_A_logits) # 对 logits 应用 sigmoid，转换到 [0, 1] 概率范围
        self.rec_B = self.instances_to_image(self.fake_A, self.real_B) # [-1, 1]

    def forward_path_ABA(self):
        """路径2: real_A → G_B → fake_B → G_A → rec_A"""
        B, N, H, W = self.real_A.shape
        self.fake_B = self.instances_to_image(self.real_A, self.real_C) # [-1, 1]

        mask_list, class_list = tensor2list(self.real_A)
        rec_A_outputs = self.netG_A(self.fake_B, mask_list, class_list)
        
        self.loss_cycle_A = rec_A_outputs['loss']
        masks_logits = rec_A_outputs['mask_logits']  # (B, num_queries, H', W')
        self.rec_A_classes = rec_A_outputs['class_logits']  # (B, num_queries, num_classes+1)

        B, N, H_low, W_low = masks_logits.shape
        if (H_low, W_low) != (H, W):
            self.rec_A_logits = F.interpolate(
                masks_logits, size=(H, W), mode='bilinear', align_corners=False
            ) 
        else:
            self.rec_A_logits = masks_logits
        
        self.rec_A = torch.sigmoid(self.rec_A_logits) # [0, 1]

    def optimize_parameters(self):
        # 1. 清零所有梯度
        self.set_requires_grad([self.netD_B], False)
        self.optimizer_G_A.zero_grad()
        self.optimizer_G_B.zero_grad()
        use_ddp = (self.opt.distributed and hasattr(self.netG_A, 'no_sync') and hasattr(self.netG_B, 'no_sync'))

        # ========== 生成器训练 (使用AMP) ==========
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                # 2. 路径1：real_B → G_A → fake_A → G_B → rec_B
                self.forward_path_BAB() 

                B, N, H, W = self.fake_A.shape
                fake_A_flat = self.fake_A.view(B * N, 1, H, W)
                pred_fake_A = self.netD_A(fake_A_flat * 2 - 1) # 判别器输入范围需要[-1, 1]
                self.loss_G_A = self.criterionGAN(pred_fake_A, True) * self.opt.lambda_GA

                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
                loss_path1 = self.loss_G_A + self.loss_cycle_B
            
            # 使用scaler进行backward
            if use_ddp:
                with self.netG_A.no_sync(), self.netG_B.no_sync():
                    self.scaler.scale(loss_path1).backward(retain_graph=False)
            else:
                self.scaler.scale(loss_path1).backward(retain_graph=False)

            with torch.amp.autocast("cuda"):
                # 3. 路径2：real_A → G_B → fake_B → G_A → rec_A
                self.forward_path_ABA()
                self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True) * self.opt.lambda_GB
                loss_path2 = self.loss_G_B + self.loss_cycle_A
            
            self.scaler.scale(loss_path2).backward(retain_graph=False)

            # 4. 梯度裁剪 (需要先unscale)
            self.scaler.unscale_(self.optimizer_G_A)
            self.scaler.unscale_(self.optimizer_G_B)
            torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), 5.0)
            
            # 5. 更新参数
            self.scaler.step(self.optimizer_G_A)
            self.scaler.step(self.optimizer_G_B)
            self.scaler.update()
        else:
            raise NotImplementedError
            self.forward_path_BAB()
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
            loss_path1 = self.loss_cycle_B 
            
            if use_ddp:
                with self.netG_A.no_sync(), self.netG_B.no_sync():
                    loss_path1.backward(retain_graph=False)
            else:
                loss_path1.backward(retain_graph=False)

            self.forward_path_ABA()
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True) * self.opt.lambda_GB
            # 使用 logits 计算损失（InstanceCycleLoss 内部会应用 sigmoid）
            # self.loss_cycle_A = self.criterionInstanceCycle(self.rec_A_logits, self.real_A, self.rec_A_classes) * self.opt.lambda_A
            
            loss_path2 = self.loss_G_B + self.loss_cycle_A
            loss_path2.backward(retain_graph=False)

            torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), 5.0)
            
            self.optimizer_G_A.step()
            self.optimizer_G_B.step()

        self.loss_G = (self.loss_G_A + self.loss_cycle_B + self.loss_G_B + self.loss_cycle_A).detach()        

        # ========== 判别器训练 (使用AMP) ==========
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_B.zero_grad()
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        fake_B = self.fake_B_pool.query(self.fake_B.detach())

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                # ========== D_A: 判别实例掩码 ==========
                B, N, H, W = self.real_A.shape
                real_A_flat = self.real_A.view(B * N, 1, H, W)
                tensor_sum = real_A_flat.sum(dim=[1, 2, 3])
                real_A_filtered = real_A_flat[tensor_sum > 10] # 过滤掉全零掩码
                pred_real_A = self.netD_A(real_A_filtered * 2 - 1) # 判别器输入范围需要[-1, 1]
                loss_D_A_real = self.criterionGAN(pred_real_A, True)
                
                B, N, H, W = fake_A.shape
                fake_A_flat = fake_A.view(B * N, 1, H, W)
                pred_fake_A = self.netD_A(fake_A_flat * 2 - 1) # 判别器输入范围需要[-1, 1]
                loss_D_A_fake = self.criterionGAN(pred_fake_A, False)
                self.loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5 * self.opt.lambda_DA

                # ========== D_B: 判别图像 ==========
                pred_real = self.netD_B(self.real_B)
                loss_D_real = self.criterionGAN(pred_real, True)
                pred_fake = self.netD_B(fake_B)
                loss_D_fake = self.criterionGAN(pred_fake, False)
                self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_DB

            self.scaler.scale(self.loss_D_A).backward()
            self.scaler.scale(self.loss_D_B).backward()
            self.scaler.step(self.optimizer_D_A)
            self.scaler.step(self.optimizer_D_B)
            self.scaler.update()
        else:
            raise NotImplementedError
            # 判别真实图像
            pred_real = self.netD_B(self.real_B)
            loss_D_real = self.criterionGAN(pred_real, True)
            # 判别生成图像
            pred_fake = self.netD_B(fake_B)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_DB
            self.loss_D_B.backward()
            self.optimizer_D.step()

        self.loss_D = (self.loss_D_A + self.loss_D_B).detach()

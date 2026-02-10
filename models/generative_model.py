import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool

class GenerativeModel(BaseModel):
    """
    使用 AMP 加速的 Mask2Image 渲染模型。
    功能：将实例掩码 (real_A) 渲染到背景 (real_C) 上，并适配目标域 (real_B)。
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='template')
        if is_train:
            parser.add_argument('--lambda_GB', type=float, default=1.0, help='weight for GAN loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_B', 'D_B']
        self.visual_names = ['real_A_vis', 'real_C', 'fake_B', 'real_B']
        self.model_names = ['G_B', 'D_B'] if self.isTrain else ['G_B']

        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
            thres=True
        )

        if self.isTrain:
            # 初始化 D_B
            self.netD_B = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids
            )
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            
            # 优化器
            self.optimizer_G = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # AMP Scaler 初始化
            self.use_amp = getattr(opt, 'use_amp', False)
            if self.use_amp:
                self.scaler = torch.amp.GradScaler('cuda')

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_A_vis = self.real_A.sum(dim=1, keepdim=True).clamp(0, 1)

    def forward(self):
        """渲染逻辑：Masks + BG -> Fake_B"""
        bg = self.real_C # [-1, 1]
        # 确保通道数匹配
        if bg.shape[1] > 1:
            bg_render = bg.mean(dim=1, keepdim=True) if self.opt.input_nc == 1 else bg
        else:
            bg_render = bg

        merged_mask, _ = torch.max(self.real_A, dim=1, keepdim=True)
        masks_input = merged_mask * 2 - 1
        
        inst_image = self.netG_B(masks_input)
        
        # 扩展 Mask 通道以匹配图像 (B, C, H, W)
        mask_expanded = merged_mask.expand_as(bg_render)
        self.fake_B = inst_image * mask_expanded + bg_render * (1 - mask_expanded)

    def optimize_parameters(self):
        """包含 AMP 逻辑的训练循环"""
        if self.use_amp:
            # --- 训练生成器 G ---
            self.set_requires_grad(self.netD_B, False)
            self.optimizer_G.zero_grad()
            
            with torch.amp.autocast('cuda'):
                self.forward()
                pred_fake = self.netD_B(self.fake_B)
                self.loss_G_B = self.criterionGAN(pred_fake, True) * self.opt.lambda_GB
            
            self.scaler.scale(self.loss_G_B).backward()
            self.scaler.step(self.optimizer_G)
            
            # --- 训练判别器 D ---
            self.set_requires_grad(self.netD_B, True)
            self.optimizer_D.zero_grad()
            
            with torch.amp.autocast('cuda'):
                fake_B = self.fake_B_pool.query(self.fake_B.detach())
                # Real
                pred_real = self.netD_B(self.real_B)
                loss_D_real = self.criterionGAN(pred_real, True)
                # Fake
                pred_fake = self.netD_B(fake_B)
                loss_D_fake = self.criterionGAN(pred_fake, False)
                self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            
            self.scaler.scale(self.loss_D_B).backward()
            self.scaler.step(self.optimizer_D)
            
            # 更新 Scaler
            self.scaler.update()
            
        else:
            # 标准 FP32 训练逻辑
            self.forward()
            # G
            self.set_requires_grad(self.netD_B, False)
            self.optimizer_G.zero_grad()
            pred_fake = self.netD_B(self.fake_B)
            self.loss_G_B = self.criterionGAN(pred_fake, True) * self.opt.lambda_GB
            self.loss_G_B.backward()
            self.optimizer_G.step()
            # D
            self.set_requires_grad(self.netD_B, True)
            self.optimizer_D.zero_grad()
            fake_B = self.fake_B_pool.query(self.fake_B.detach())
            loss_D_real = self.criterionGAN(self.netD_B(self.real_B), True)
            loss_D_fake = self.criterionGAN(self.netD_B(fake_B), False)
            self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            self.loss_D_B.backward()
            self.optimizer_D.step()

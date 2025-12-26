from re import T
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import functools
from torch.optim import lr_scheduler
from .Linknet import UNet11, LinkNet34
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, Mask2FormerConfig, SwinConfig
from scipy.optimize import linear_sum_assignment
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.2)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # Only print on main process (rank 0) or when not using DDP
    if not dist.is_initialized() or dist.get_rank() == 0:
        print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], distributed=False):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        distributed (bool) -- whether to use DistributedDataParallel

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if distributed and dist.is_initialized():
            # Use DistributedDataParallel for DDP training，device_ids should only contain the current process's GPU
            net = DDP(net, device_ids=[gpu_ids[0]], find_unused_parameters=True)
        elif len(gpu_ids) > 1:
            # Fallback to DataParallel for multi-GPU without DDP
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], thres=False, distributed=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        distributed (bool) -- whether to use DistributedDataParallel

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=10, thres=thres)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'LinkNet34':
        net = LinkNet34(num_classes=output_nc, num_channels=input_nc, pretrained=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, distributed)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], distributed=False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        distributed (bool) -- whether to use DistributedDataParallel

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids, distributed)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',thres=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.thres=thres
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        #model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        outs=self.model(input)
        if self.thres:
            predict = torch.where(outs>0.5,torch.ones_like(outs),torch.zeros_like(outs))
        else:
            predict=outs
        return outs



class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


###############################################################################
# Mask2Former for Instance Segmentation
###############################################################################

class Mask2FormerWrapper(nn.Module):
    """Wrapper for Mask2Former to be used in CycleGAN-like architecture.
    Mask2Former 使用 Transformer 架构进行实例分割。
    
    单类别配置：只检测前景对象，所有实例都属于同一类别（前景 vs 背景）。
    
    输出格式统一为 list[dict]，每个 dict 包含:
        - 'masks': Tensor (N, 1, H, W), 实例掩码（训练时保持梯度）
        - 'mask_logits': Tensor (N, 1, H, W), 掩码的原始 logits（未经 sigmoid）
        - 'scores': Tensor (N,), 置信度分数
        - 'labels': Tensor (N,), 类别标签（全部为0，表示前景）
    """
    
    def __init__(self, pretrained=False, num_queries=10, score_thresh=0.5, mask_thresh=0.5, soft_thresh_temp=10.0):
        """
        Parameters:
            pretrained (bool) -- 是否使用预训练权重
            score_thresh (float) -- 置信度阈值
            mask_thresh (float) -- 掩码二值化阈值
            soft_thresh_temp (float) -- 软阈值温度参数，越大越接近硬阈值
        """
        super(Mask2FormerWrapper, self).__init__()
        self.soft_thresh_temp = soft_thresh_temp
        
        self.score_thresh = score_thresh
        self.mask_thresh = mask_thresh
        
        # 配置为单类别（只有前景）
        if pretrained:
            model_name = "facebook/mask2former-swin-tiny-coco-instance"
            config = Mask2FormerConfig.from_pretrained(model_name)
            config.num_labels = 1
            config.num_queries = num_queries
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name, 
                config=config,
                ignore_mismatched_sizes=True  # 忽略分类头的大小不匹配
            )
        else:
            config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
            config.num_labels = 1
            config.num_queries = num_queries
            config.backbone_config = SwinConfig.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224", 
                out_features=['stage1', 'stage2', 'stage3', 'stage4'], 
                use_pretrained_backbone = True
            )
            self.model = Mask2FormerForUniversalSegmentation(config)
            
            # self.processor = Mask2FormerImageProcessor(num_labels=1)
    
    def forward(self, images):
        """前向传播
        Parameters:
            images (Tensor) -- 输入图像 (B, C, H, W)，C可以是1或3
        Returns:
            训练时: dict，批量处理的张量
        """
        # Mask2Former backbone 需要 3 通道输入，将单通道复制3次变成3通道 (B, 1, H, W) -> (B, 3, H, W)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        with torch.set_grad_enabled(self.training):
            outputs = self.model(pixel_values=images)

        pred_masks_logits = outputs.masks_queries_logits  # (B, num_queries, H', W')
        pred_class_logits = outputs.class_queries_logits  # (B, num_queries, num_classes + 1)
        # hidden_states = outputs.hidden_states 
        # attentions = outputs.attentions
    
        if self.training:
            return {
                'mask_logits': pred_masks_logits,  # (B, num_queries, H', W')
                'class_logits': pred_class_logits,  # (B, num_queries, num_classes + 1)
            }
        else:
            return NotImplementedError

def define_Mask2Former(pretrained=False, gpu_ids=[], distributed=False, num_queries=10,
                       score_thresh=0.5, mask_thresh=0.5, soft_thresh_temp=10.0):
    """ 
    配置为单类别前景分割：所有检测到的实例都是前景对象。
    Parameters:
        pretrained (bool) -- 是否使用预训练权重（会被修改为单类别）
        gpu_ids (list) -- GPU IDs
        distributed (bool) -- 是否使用分布式训练
        num_queries (int) -- 查询数量（最大实例数）
        score_thresh (float) -- 置信度阈值
        mask_thresh (float) -- 掩码二值化阈值
        soft_thresh_temp (float) -- 软阈值温度参数
    Returns:
        Mask2FormerWrapper instance (单类别配置)
    """
    net = Mask2FormerWrapper(pretrained=pretrained, num_queries=num_queries, score_thresh=score_thresh, 
                              mask_thresh=mask_thresh, soft_thresh_temp=soft_thresh_temp)
    if len(gpu_ids) > 0: # Move to GPU
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if distributed and dist.is_initialized():
            net = DDP(net, device_ids=gpu_ids, output_device=gpu_ids[0], find_unused_parameters=True)
        elif len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    return net

class InstanceCycleLoss(nn.Module):
    """
    单类别实例分割的循环一致性损失 - 基于 Mask2Former 的匹配和损失机制
    核心思想（单类别前景版本）：
    1. 过滤掉0填充的实例（target 和 pred 都可能有 padding）
    2. 使用匈牙利算法匹配有效的预测和目标实例（仅基于 mask，单类别不需要类别代价）
    3. 对匹配的实例对计算 Dice Loss + BCE Loss + Class Loss
    4. 对未匹配的预测 queries 计算 no-object 类别损失
    注意：
    - 单类别任务：所有实例都是前景（类别0），不需要在匹配时考虑类别代价
    - target_masks 的 shape 为 (B, N, H, W)，不足的实例用全零填充
    - 通过面积阈值 (0.1% * H * W) 自动过滤掉填充的实例
    """
    def __init__(self, 
                 cost_mask=1.0, #匹配时 BCE 代价的权重
                 cost_dice=1.0, #匹配时 Dice 代价的权重
                 lambda_mask=1.0, #最终损失 BCE 权重
                 lambda_dice=1.0, #最终损失 Dice 权重
                 lambda_class=1.0,  #最终损失中类别的权重（区分前景/no-object）
                 num_points=12544):  # Mask2Former 默认采样点数
        super(InstanceCycleLoss, self).__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.lambda_mask = lambda_mask
        self.lambda_dice = lambda_dice
        self.lambda_class = lambda_class
        self.num_points = num_points
    
    def _batch_dice_loss(self, inputs, targets):
        """
        批量计算 Dice Loss（用于匹配）
        Args:
            inputs: (N, num_points) 预测掩码采样点
            targets: (M, num_points) 目标掩码采样点
        Returns:
            cost_matrix: (N, M) Dice 代价矩阵
        """
        inputs = inputs.sigmoid()
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss
    
    def _batch_sigmoid_ce_loss(self, inputs, targets):
        """
        批量计算 BCE Loss（用于匹配）
        Args:
            inputs: (N, num_points) 预测掩码采样点 logits
            targets: (M, num_points) 目标掩码采样点
        Returns:
            cost_matrix: (N, M) BCE 代价矩阵
        """
        num_points = inputs.shape[1]
        
        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )
        
        loss = torch.einsum("nc,mc->nm", pos, targets) + \
               torch.einsum("nc,mc->nm", neg, (1 - targets))
        
        return loss / num_points
    
    def _hungarian_match(self, pred_masks, target_masks):
        """
        使用匈牙利算法匹配实例（单类别版本，仅基于掩码匹配）
        单类别优化：所有目标都是同一类别（前景），类别代价对所有目标相同，不影响匹配结果，因此匹配时只考虑 mask 和 dice 代价。
        Args:
            pred_masks: (N, H, W) 预测实例掩码 logits
            target_masks: (M, H, W) 目标实例掩码（二值）
        Returns:
            matched_pred_idx: list - 匹配的预测索引
            matched_target_idx: list - 匹配的目标索引
        """
        N, H, W = pred_masks.shape
        M = target_masks.shape[0]
        
        if N == 0 or M == 0:
            return [], []
        
        # 随机采样点（内存高效，与 Mask2Former 一致）
        point_coords = torch.rand(1, self.num_points, 2, device=pred_masks.device)
        
        # 采样预测掩码
        pred_points = self._point_sample(
            pred_masks.unsqueeze(1),  # (N, 1, H, W)
            point_coords.repeat(N, 1, 1)
        ).squeeze(1)  # (N, num_points)
        
        # 采样目标掩码
        target_points = self._point_sample(
            target_masks.unsqueeze(1),  # (M, 1, H, W)
            point_coords.repeat(M, 1, 1)
        ).squeeze(1)  # (M, num_points)
        
        # 计算代价矩阵（仅基于掩码）
        with torch.amp.autocast('cuda', enabled=False):
            pred_points = pred_points.float()
            target_points = target_points.float()
            
            cost_mask = self._batch_sigmoid_ce_loss(pred_points, target_points)
            cost_dice = self._batch_dice_loss(pred_points, target_points)
        
        # 总代价（单类别：不需要类别代价）
        C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
        # 匈牙利匹配不需要梯度，detach后转换为numpy
        C = C.detach().cpu().numpy()
        
        # 匈牙利匹配
        pred_idx, target_idx = linear_sum_assignment(C)
        return pred_idx.tolist(), target_idx.tolist()
    
    def _point_sample(self, masks, point_coords):
        """
        在掩码上采样点（简化版 point_sample）
        Args:
            masks: (B, 1, H, W) 掩码
            point_coords: (B, num_points, 2) 归一化坐标 [0, 1]
        Returns:
            sampled: (B, 1, num_points) 采样值
        """
        # 转换为 grid_sample 需要的格式 [-1, 1]
        point_coords = 2.0 * point_coords - 1.0
        sampled = F.grid_sample(
            masks.float(),
            point_coords.unsqueeze(1),  # (B, 1, num_points, 2)
            mode='bilinear',
            align_corners=False
        )
        return sampled.squeeze(2)  # (B, 1, num_points)
    
    def _compute_loss(self, pred_masks, target_masks, pred_idx, target_idx, 
                     pred_classes=None, valid_pred_mask=None, num_total_queries=None):
        """
        计算匹配实例对的损失（与 Mask2Former 的 loss_masks 一致）
        Args:
            pred_masks: (N', H, W) 有效的预测掩码 logits
            target_masks: (M', H, W) 有效的目标掩码
            pred_idx: list - 匹配的预测索引（在有效预测中的索引）
            target_idx: list - 匹配的目标索引（在有效目标中的索引）
            pred_classes: (N_total, num_classes+1) 所有 queries 的预测类别 logits，可选
            valid_pred_mask: (N_total,) 布尔张量，标记哪些是有效预测
            num_total_queries: int - 总查询数（用于类别损失）
        Returns:
            loss_mask: BCE Loss
            loss_dice: Dice Loss
            loss_class: Class Loss
        """
        device = pred_masks.device
        
        if len(pred_idx) == 0:
            # 没有匹配：掩码和dice损失为0，但仍需计算类别损失。如果有类别预测，所有 queries 都应该预测为 no-object
            if pred_classes is not None and num_total_queries is not None:
                # 所有 queries 的目标类别都是 no-object（最后一个类别）
                num_classes = pred_classes.shape[-1] - 1
                target_classes = torch.full(
                    (num_total_queries,), num_classes, dtype=torch.long, device=device
                )
                loss_class = F.cross_entropy(pred_classes, target_classes, reduction='mean')
                # 掩码和dice损失为0，但需要是可梯度的张量（与loss_class相同的计算图）
                loss_mask = pred_classes.sum() * 0.0  # 保持梯度连接
                loss_dice = pred_classes.sum() * 0.0
            else:
                # 没有类别预测时，返回简单的零张量
                loss_mask = torch.tensor(0.0, device=device)
                loss_dice = torch.tensor(0.0, device=device)
                loss_class = torch.tensor(0.0, device=device)
            
            return loss_mask, loss_dice, loss_class
        
        # 选择匹配的掩码
        matched_pred = pred_masks[pred_idx]  # (K, H, W)
        matched_target = target_masks[target_idx]  # (K, H, W)
        
        # 随机采样点
        K = len(pred_idx)
        point_coords = torch.rand(1, self.num_points, 2, device=device)
        
        # 采样预测掩码 logits
        pred_points = self._point_sample(
            matched_pred.unsqueeze(1),  # (K, 1, H, W)
            point_coords.repeat(K, 1, 1)
        ).squeeze(1)  # (K, num_points)
        
        # 采样目标掩码
        target_points = self._point_sample(
            matched_target.unsqueeze(1),  # (K, 1, H, W)
            point_coords.repeat(K, 1, 1)
        ).squeeze(1)  # (K, num_points)
        
        # 计算掩码损失
        loss_mask = F.binary_cross_entropy_with_logits(
            pred_points, target_points, reduction='mean'
        )
        
        loss_dice = self._dice_loss(pred_points, target_points)
        
        # 计算类别损失（单类别：区分前景 vs no-object）
        if pred_classes is not None and num_total_queries is not None:
            num_classes = pred_classes.shape[-1] - 1  # 不包括 no-object
            target_classes = torch.full(
                (num_total_queries,), num_classes, dtype=torch.long, device=device
            )  # 默认所有都是 no-object（类别1）
            
            # 将有效预测中匹配的索引映射回原始所有 queries 的索引
            if valid_pred_mask is not None:
                # 获取有效预测在原始序列中的索引
                valid_indices = torch.where(valid_pred_mask)[0]  # (N',)
                # 匹配的 queries 的目标类别是前景（类别0）
                for idx in pred_idx:
                    original_idx = valid_indices[idx].item()
                    target_classes[original_idx] = 0  # 前景
            else:
                # 如果没有提供 valid_pred_mask，假设 pred_idx 就是原始索引
                for idx in pred_idx:
                    target_classes[idx] = 0  # 前景
            
            # 计算交叉熵损失（2个类别：前景和no-object）
            loss_class = F.cross_entropy(pred_classes, target_classes, reduction='mean')
        else:
            # 没有类别预测时，使用掩码损失的梯度连接（保持梯度图连接）
            loss_class = loss_mask * 0.0
        
        return loss_mask, loss_dice, loss_class
    
    def _dice_loss(self, inputs, targets):
        """
        单对单的 Dice Loss
        Args:
            inputs: (K, num_points) logits
            targets: (K, num_points) 二值标签
        Returns:
            loss: 标量
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss.mean()
    
    def forward(self, pred_masks, target_masks, pred_classes=None):
        """
        计算实例循环一致性损失（完整 Mask2Former 风格）
        Args:
            pred_masks: (B, N, H, W) 预测/重建的实例掩码 logits
            target_masks: (B, N, H, W) 目标/原始实例掩码（二值），不足的用0填充
            pred_classes: (B, N, num_classes+1) 预测类别 logits，可选
        Returns:
            loss: 标量损失
        """
        B, N, H, W = pred_masks.shape
        device = pred_masks.device
        
        # 归一化目标掩码到 [0, 1]
        if target_masks.min() < 0:
            target_masks = (target_masks + 1) / 2
        
        # 初始化损失累积器为张量（确保梯度正确传播）
        total_loss_mask = torch.tensor(0.0, device=device)
        total_loss_dice = torch.tensor(0.0, device=device)
        total_loss_class = torch.tensor(0.0, device=device)
        num_valid_batches = 0
        
        for b in range(B):
            pred_b = pred_masks[b]  # (N, H, W)
            target_b = target_masks[b]  # (N, H, W)
            pred_class_b = pred_classes[b] if pred_classes is not None else None  # (N, num_classes+1)
            
            # 过滤掉空实例（面积过小或0填充的实例）
            # 对于 target，面积几乎为0的认为是padding
            # 对于 pred，sigmoid后面积过小的认为是无效预测
            pred_areas = pred_b.sigmoid().sum(dim=(1, 2))
            target_areas = target_b.sum(dim=(1, 2))
            
            # 使用更小的阈值来检测0填充（0.1%）
            min_area_thresh = 0.001 * H * W
            valid_pred_mask = pred_areas > min_area_thresh
            valid_target_mask = target_areas > min_area_thresh
            
            # 至少需要有目标实例（排除全是0填充的情况）
            if not valid_target_mask.any():
                # 如果没有有效目标，所有 queries 都应该预测为 no-object
                if pred_class_b is not None:
                    num_classes = pred_class_b.shape[-1] - 1
                    target_classes = torch.full(
                        (N,), num_classes, dtype=torch.long, device=pred_b.device
                    )
                    loss_class = F.cross_entropy(pred_class_b, target_classes, reduction='mean')
                    total_loss_class += loss_class
                num_valid_batches += 1
                continue
            
            # 获取有效的目标实例
            valid_target = target_b[valid_target_mask]  # (M', H, W)
            
            if not valid_pred_mask.any():
                # 没有有效预测，所有 queries 都应该预测为 no-object
                if pred_class_b is not None:
                    num_classes = pred_class_b.shape[-1] - 1
                    target_classes = torch.full(
                        (N,), num_classes, dtype=torch.long, device=pred_b.device
                    )
                    loss_class = F.cross_entropy(pred_class_b, target_classes, reduction='mean')
                    total_loss_class += loss_class
                num_valid_batches += 1
                continue
            
            # 获取有效的预测实例
            valid_pred = pred_b[valid_pred_mask]  # (N', H, W)
            
            # 匈牙利匹配（单类别：仅基于掩码匹配，不需要类别信息）
            pred_idx, target_idx = self._hungarian_match(valid_pred, valid_target)
            
            # 计算匹配损失（传入完整的 pred_class_b 和有效预测的索引映射）
            loss_mask, loss_dice, loss_class = self._compute_loss(
                valid_pred, valid_target, pred_idx, target_idx, 
                pred_class_b, valid_pred_mask, num_total_queries=N
            )
            
            total_loss_mask += loss_mask
            total_loss_dice += loss_dice
            total_loss_class += loss_class
            num_valid_batches += 1
        
        if num_valid_batches == 0:
            device = pred_masks.device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 平均并加权
        avg_loss_mask = total_loss_mask / num_valid_batches
        avg_loss_dice = total_loss_dice / num_valid_batches
        avg_loss_class = total_loss_class / num_valid_batches
        
        total_loss = (self.lambda_mask * avg_loss_mask + 
                     self.lambda_dice * avg_loss_dice + 
                     self.lambda_class * avg_loss_class)
        
        return total_loss

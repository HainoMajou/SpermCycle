from .Linknet import LinkNet34
from .resnet import ResnetGenerator
from .unet import UnetGenerator
from .utils import get_norm_layer, init_net
from .mask2former import Mask2FormerWrapper
import torch

def define_G(pretrained, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], thres=False, distributed=False):
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

    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        # state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        net.load_state_dict(state_dict, strict=True)
        net = init_net(net, None, None, gpu_ids, distributed)
        print("Generator model loaded from %s" % pretrained)
    else:
        net = init_net(net, init_type, init_gain, gpu_ids, distributed)
    return net

def define_Mask2Former(pretrained=None, num_queries=10,
                       init_type='normal', init_gain=0.02, gpu_ids=[], distributed=False):
    """ 
    配置为单类别前景分割：所有检测到的实例都是前景对象。
    Parameters:
        pretrained (bool) -- 是否使用预训练权重（会被修改为单类别）
        gpu_ids (list) -- GPU IDs
        distributed (bool) -- 是否使用分布式训练
        num_queries (int) -- 查询数量（最大实例数）
    Returns:
        Mask2FormerWrapper instance (单类别配置)
    """
    net = Mask2FormerWrapper(pretrained=pretrained, num_queries=num_queries)
    net = init_net(net, init_type, init_gain, gpu_ids, distributed)
    
    return net


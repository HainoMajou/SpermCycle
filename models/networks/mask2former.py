import torch.nn as nn
import functools
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

def param_stastic(model):
    pixel_encoder_params = 0
    pixel_decoder_params = 0
    transformer_params = 0
    for name, param in model.named_parameters():
        # print(name, param.shape)
        # print(param.mean().item(), param.std().item())
        if "pixel_level_module.encoder" in name:
            # param.requires_grad = False
            pixel_encoder_params += param.numel()
        elif "pixel_level_module.decoder" in name:
            pixel_decoder_params += param.numel()
        elif "transformer_module" in name:
            transformer_params += param.numel()
    print("pixel_encoder_params: ", pixel_encoder_params)
    print("pixel_decoder_params: ", pixel_decoder_params)
    print("transformer_params: ", transformer_params)

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
    
    def __init__(self, preseg=None, num_queries=10):
        """
        Parameters:
            pretrained (bool) -- 是否使用预训练权重
        """
        super(Mask2FormerWrapper, self).__init__()
        self.num_queries = num_queries
        
        config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
        config.num_labels = 1
        config.num_queries = self.num_queries
        if preseg=='hf':
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-tiny-coco-instance",
                config = config,
                ignore_mismatched_sizes = True
            )
            print("Mask2Former model loaded from facebook/mask2former-swin-tiny-coco-instance")
        elif preseg is not None:
            self.model = Mask2FormerForUniversalSegmentation(config)
            state_dict = torch.load(preseg)
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            print("Mask2Former model loaded from %s" % preseg)
        else:
            config.use_pretrained_backbone = True
            config.backbone = "microsoft/swin-tiny-patch4-window7-224"
            config.backbone_kwargs = {
                "architectures": ["SwinForImageClassification"],
                "attention_probs_dropout_prob": 0.0,
                "depths": [2, 2, 6, 2],
                "drop_path_rate": 0.3,
                "dtype": "float32",
                "embed_dim": 96,
                "encoder_stride": 32,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 768,
                "image_size": 224,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-05,
                "mlp_ratio": 4.0,
                "model_type": "swin",
                "num_channels": 3,
                "num_heads": [3, 6, 12, 24],
                "num_layers": 4,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
                "out_indices": [1, 2, 3, 4],
                "patch_size": 4,
                "path_norm": True,
                "qkv_bias": True,
                "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
                "use_absolute_embeddings": False,
                "window_size": 7   
            }
            config.backbone_config=None
            # config.backbone_config = SwinConfig.from_pretrained(
            #     "microsoft/swin-tiny-patch4-window7-224", 
            #     out_features=['stage1', 'stage2', 'stage3', 'stage4'], 
            # )
            # print(config)
            self.model = Mask2FormerForUniversalSegmentation(config)
            param_stastic(self.model)
            print("Mask2Former model initialized")
    
    def forward(self, images, mask_list=None, class_list=None):
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
            outputs = self.model(pixel_values=images, mask_labels=mask_list, class_labels=class_list)

        instance_loss = outputs.loss
        pred_masks_logits = outputs.masks_queries_logits  # (B, num_queries, H', W')
        pred_class_logits = outputs.class_queries_logits  # (B, num_queries, num_classes + 1)
        # hidden_states = outputs.hidden_states
        # attentions = outputs.attentions

        # Return the same format for both training and inference
        return {
            'loss': instance_loss,
            'mask_logits': pred_masks_logits,  # (B, num_queries, H', W')
            'class_logits': pred_class_logits,  # (B, num_queries, num_classes + 1)
        }
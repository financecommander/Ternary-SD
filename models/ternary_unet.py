"""
Ternary UNet for Stable Diffusion
Converts diffusers UNet2DConditionModel to use ternary quantization
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import logging

from .ternary_layers import (
    TernaryConv2d,
    TernaryLinear,
    calculate_compression_ratio,
    count_ternary_layers
)

logger = logging.getLogger(__name__)

def convert_layer_to_ternary(module, threshold=0.05):
    """Recursively convert Conv2d and Linear layers to ternary"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, TernaryConv2d):
            # Create TernaryConv2d
            ternary_conv = TernaryConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                threshold=threshold
            )
            # Copy weights
            ternary_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ternary_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, ternary_conv)
            
        elif isinstance(child, nn.Linear) and not isinstance(child, TernaryLinear):
            # Create TernaryLinear
            ternary_linear = TernaryLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                threshold=threshold
            )
            # Copy weights
            ternary_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ternary_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, ternary_linear)
        else:
            # Recurse
            convert_layer_to_ternary(child, threshold)
    
    return module

class TernaryUNet2DConditionModel:
    """Factory class for creating ternary UNet models"""
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, threshold=0.05, **kwargs):
        """Load pretrained UNet and convert to ternary"""
        logger.info(f"Loading UNet from {pretrained_model_name_or_path}")
        
        # Load original UNet
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        logger.info(f"Converting to ternary (threshold={threshold})")
        
        # Convert to ternary
        unet = convert_layer_to_ternary(unet, threshold)
        
        # Log statistics
        layer_counts = count_ternary_layers(unet)
        compression = calculate_compression_ratio(unet)
        
        logger.info(f"Conversion complete:")
        logger.info(f"  - Ternary Conv2d: {layer_counts['ternary_conv2d']}")
        logger.info(f"  - Ternary Linear: {layer_counts['ternary_linear']}")
        logger.info(f"  - Compression: {compression['compression_vs_fp16']:.2f}x vs FP16")
        logger.info(f"  - Model size: {compression['ternary_size_mb']:.1f} MB")
        
        return unet
    
    @staticmethod
    def get_stats(unet):
        """Get statistics for a ternary UNet"""
        return {
            **count_ternary_layers(unet),
            **calculate_compression_ratio(unet)
        }
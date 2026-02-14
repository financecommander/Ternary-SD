"""
Ternary Quantization Layers for Neural Networks
Implements {-1, 0, +1} weight quantization with straight-through estimator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryQuantize(torch.autograd.Function):
    """Ternary quantization with straight-through estimator"""
    
    @staticmethod
    def forward(ctx, input, threshold=0.05):
        output = input.clone()
        output[torch.abs(input) < threshold] = 0
        output[input >= threshold] = 1
        output[input <= -threshold] = -1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def ternarize_tensor(tensor, threshold=0.05):
    """Convert tensor to ternary {-1, 0, +1}"""
    return TernaryQuantize.apply(tensor, threshold)

class TernaryConv2d(nn.Conv2d):
    """Ternary Convolutional Layer"""
    
    def __init__(self, *args, threshold=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.register_buffer('weight_scale', torch.ones(1))
        
    def forward(self, x):
        ternary_weight = ternarize_tensor(self.weight, self.threshold)
        with torch.no_grad():
            self.weight_scale = self.weight.abs().mean()
        scaled_weight = ternary_weight * self.weight_scale
        return F.conv2d(x, scaled_weight, self.bias, self.stride, 
                       self.padding, self.dilation, self.groups)

class TernaryLinear(nn.Linear):
    """Ternary Linear Layer"""
    
    def __init__(self, *args, threshold=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.register_buffer('weight_scale', torch.ones(1))
        
    def forward(self, x):
        ternary_weight = ternarize_tensor(self.weight, self.threshold)
        with torch.no_grad():
            self.weight_scale = self.weight.abs().mean()
        scaled_weight = ternary_weight * self.weight_scale
        return F.linear(x, scaled_weight, self.bias)

def calculate_compression_ratio(model):
    """Calculate compression statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    fp32_size = total_params * 4
    fp16_size = total_params * 2
    ternary_size = total_params * 2.5 / 8
    
    return {
        'total_parameters': total_params,
        'fp32_size_mb': fp32_size / (1024**2),
        'fp16_size_mb': fp16_size / (1024**2),
        'ternary_size_mb': ternary_size / (1024**2),
        'compression_vs_fp32': fp32_size / ternary_size,
        'compression_vs_fp16': fp16_size / ternary_size,
    }

def count_ternary_layers(model):
    """Count ternary layers in model"""
    conv_count = sum(1 for m in model.modules() if isinstance(m, TernaryConv2d))
    linear_count = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    return {
        'ternary_conv2d': conv_count,
        'ternary_linear': linear_count,
        'total_ternary': conv_count + linear_count
    }
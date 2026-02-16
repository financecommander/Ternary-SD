"""
Performance benchmark tests for ternary quantization
"""

import pytest
import torch
from models.ternary_layers import ternarize_tensor, TernaryConv2d


def test_ternarize_performance(benchmark):
    """Benchmark ternary quantization performance"""
    test_tensor = torch.randn(1000, 1000)
    
    result = benchmark(ternarize_tensor, test_tensor, threshold=0.05)
    
    assert result.shape == test_tensor.shape


def test_ternary_conv2d_performance(benchmark):
    """Benchmark TernaryConv2d forward pass performance"""
    conv = TernaryConv2d(3, 16, 3, padding=1)
    x = torch.randn(1, 3, 224, 224)
    
    result = benchmark(conv, x)
    
    assert result.shape[0] == 1
    assert result.shape[1] == 16

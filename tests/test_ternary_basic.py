"""
Basic tests for ternary quantization layers
"""

import torch
import pytest
from models.ternary_layers import ternarize_tensor, TernaryConv2d


def test_ternarize_tensor():
    """Test ternary quantization of tensor values"""
    # Create test tensor with specific values
    test_tensor = torch.tensor([-0.1, -0.03, 0.0, 0.03, 0.1])

    # Ternarize with threshold 0.05
    result = ternarize_tensor(test_tensor, threshold=0.05)

    # Expected: [-1, 0, 0, 0, 1]
    expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0])

    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_ternary_conv2d():
    """Test TernaryConv2d forward pass"""
    # Create ternary convolution layer
    conv = TernaryConv2d(3, 16, 3, padding=1)

    # Create random input
    x = torch.randn(1, 3, 32, 32)

    # Forward pass
    output = conv(x)

    # Check output shape
    expected_shape = (1, 16, 32, 32)  # Same spatial dims due to padding
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Check that output is not NaN or Inf
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


if __name__ == "__main__":
    test_ternarize_tensor()
    test_ternary_conv2d()
    print("✅ All tests passed!")
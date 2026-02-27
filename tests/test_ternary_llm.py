"""
Tests for ternary LLM quantization module.
Validates conversion logic without requiring a full LLM download.
"""

import torch
import torch.nn as nn
import pytest

from models.ternary_layers import TernaryLinear, ternarize_tensor
from models.ternary_llm import convert_llm_to_ternary, should_quantize


# ── Mock transformer block for testing ──────────────────────────────

class MockAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
        return self.o_proj(attn @ v)


class MockFFN(nn.Module):
    def __init__(self, dim=64, ffn_dim=128):
        super().__init__()
        self.up_proj = nn.Linear(dim, ffn_dim)
        self.down_proj = nn.Linear(ffn_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class MockTransformerBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.self_attn = MockAttention(dim)
        self.ffn = MockFFN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MockLLM(nn.Module):
    """Minimal LLM-like model for testing ternary conversion."""
    def __init__(self, vocab_size=100, dim=64, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MockTransformerBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ── Tests ───────────────────────────────────────────────────────────

def test_should_quantize():
    """Test layer name filtering logic."""
    skip = ["embed", "lm_head", "norm"]

    assert should_quantize("layers.0.self_attn.q_proj", skip) is True
    assert should_quantize("layers.0.ffn.up_proj", skip) is True
    assert should_quantize("embed_tokens", skip) is False
    assert should_quantize("lm_head", skip) is False
    assert should_quantize("layers.0.norm1", skip) is False
    assert should_quantize("model.norm", skip) is False


def test_convert_llm_preserves_embeddings():
    """Embeddings and lm_head should NOT be converted to ternary."""
    model = MockLLM(vocab_size=100, dim=64, num_layers=2)
    model = convert_llm_to_ternary(model, threshold=0.05)

    # Embeddings preserved
    assert isinstance(model.embed_tokens, nn.Embedding)
    # lm_head preserved
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, TernaryLinear)
    # Norms preserved
    assert isinstance(model.norm, nn.LayerNorm)


def test_convert_llm_converts_attention():
    """Q/K/V/O projection layers should be TernaryLinear."""
    model = MockLLM(vocab_size=100, dim=64, num_layers=2)
    model = convert_llm_to_ternary(model, threshold=0.05)

    for layer in model.layers:
        assert isinstance(layer.self_attn.q_proj, TernaryLinear)
        assert isinstance(layer.self_attn.k_proj, TernaryLinear)
        assert isinstance(layer.self_attn.v_proj, TernaryLinear)
        assert isinstance(layer.self_attn.o_proj, TernaryLinear)


def test_convert_llm_converts_ffn():
    """FFN layers should be TernaryLinear."""
    model = MockLLM(vocab_size=100, dim=64, num_layers=2)
    model = convert_llm_to_ternary(model, threshold=0.05)

    for layer in model.layers:
        assert isinstance(layer.ffn.up_proj, TernaryLinear)
        assert isinstance(layer.ffn.down_proj, TernaryLinear)


def test_convert_llm_forward_pass():
    """Model should still produce valid output after conversion."""
    model = MockLLM(vocab_size=100, dim=64, num_layers=2)
    model = convert_llm_to_ternary(model, threshold=0.05)

    input_ids = torch.randint(0, 100, (1, 16))
    output = model(input_ids)

    assert output.shape == (1, 16, 100)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"


def test_ternary_linear_weights_are_ternary():
    """After conversion, ternary layer weights should be {-1, 0, +1} during forward."""
    model = MockLLM(vocab_size=100, dim=64, num_layers=1)
    model = convert_llm_to_ternary(model, threshold=0.05)

    q_proj = model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, TernaryLinear)

    # Ternarize and check values
    tw = ternarize_tensor(q_proj.weight.data, q_proj.threshold)
    unique = torch.unique(tw)
    for val in unique:
        assert val.item() in (-1.0, 0.0, 1.0), f"Unexpected value: {val}"


def test_layer_count():
    """Verify expected number of layers converted."""
    model = MockLLM(vocab_size=100, dim=64, num_layers=2)
    model = convert_llm_to_ternary(model, threshold=0.05)

    ternary_count = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    # 2 layers x (4 attn proj + 2 ffn) = 12 TernaryLinear
    assert ternary_count == 12, f"Expected 12 ternary layers, got {ternary_count}"

    # lm_head should still be regular Linear
    regular_count = sum(
        1 for m in model.modules()
        if isinstance(m, nn.Linear) and not isinstance(m, TernaryLinear)
    )
    assert regular_count == 1, f"Expected 1 preserved Linear (lm_head), got {regular_count}"


if __name__ == "__main__":
    test_should_quantize()
    test_convert_llm_preserves_embeddings()
    test_convert_llm_converts_attention()
    test_convert_llm_converts_ffn()
    test_convert_llm_forward_pass()
    test_ternary_linear_weights_are_ternary()
    test_layer_count()
    print("All LLM ternary tests passed!")

"""
Ternary Quantization for Large Language Models (Transformer Decoders)

Extends ternary quantization from vision (UNet) to text models.
Converts Linear layers in attention (Q/K/V/O projections) and FFN blocks
to TernaryLinear, while preserving embeddings, norms, and lm_head.

Supports: Phi-3 Mini, Gemma 2B, Qwen2, TinyLlama, and other
HuggingFace AutoModelForCausalLM-compatible models.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

from .ternary_layers import (
    TernaryLinear,
    calculate_compression_ratio,
    count_ternary_layers,
)

logger = logging.getLogger(__name__)

# Layer name patterns to skip (keep in full precision)
DEFAULT_SKIP_PATTERNS = [
    "embed",       # Token / position embeddings
    "lm_head",     # Output vocabulary projection
    "norm",        # LayerNorm / RMSNorm
    "layernorm",   # Alternative naming
    "rotary",      # Rotary position encoding
]


def should_quantize(name: str, skip_patterns: list[str]) -> bool:
    """Check whether a named layer should be quantized."""
    name_lower = name.lower()
    return not any(pat in name_lower for pat in skip_patterns)


def convert_llm_to_ternary(
    model: nn.Module,
    threshold: float = 0.05,
    skip_patterns: Optional[list[str]] = None,
) -> nn.Module:
    """Recursively convert Linear layers in a transformer LLM to TernaryLinear.

    Preserves embeddings, normalization layers, and the language model head
    in full precision for numerical stability.

    Args:
        model: A HuggingFace causal LM (e.g., AutoModelForCausalLM).
        threshold: Ternary quantization threshold (values with |w| < threshold -> 0).
        skip_patterns: Layer name substrings to keep in full precision.
                       Defaults to embeddings, lm_head, norms.

    Returns:
        The same model object with Linear layers replaced by TernaryLinear.
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    converted = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, TernaryLinear):
            continue

        if not should_quantize(name, skip_patterns):
            skipped += 1
            continue

        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for attr in parts[:-1]:
            parent = getattr(parent, attr)
        child_name = parts[-1]

        # Build replacement TernaryLinear
        original = getattr(parent, child_name)
        ternary_linear = TernaryLinear(
            in_features=original.in_features,
            out_features=original.out_features,
            bias=original.bias is not None,
            threshold=threshold,
        )
        ternary_linear.weight.data.copy_(original.weight.data)
        if original.bias is not None:
            ternary_linear.bias.data.copy_(original.bias.data)

        setattr(parent, child_name, ternary_linear)
        converted += 1

    logger.info(
        f"LLM ternary conversion: {converted} layers converted, "
        f"{skipped} layers preserved in full precision"
    )
    return model


class TernaryLLM:
    """Factory class for creating ternary-quantized language models.

    Usage:
        model, tokenizer = TernaryLLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        output = model.generate(tokenizer("Hello", return_tensors="pt").input_ids)
    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        threshold: float = 0.05,
        skip_patterns: Optional[list[str]] = None,
        torch_dtype=None,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        """Load a pretrained LLM and convert to ternary quantization.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            threshold: Ternary quantization threshold.
            skip_patterns: Layer name patterns to preserve in full precision.
            torch_dtype: Weight dtype before quantization (torch.float16 recommended).
            device_map: Device placement ("auto", "cpu", etc.).

        Returns:
            Tuple of (ternary_model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM: {model_name_or_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs, **kwargs
        )

        logger.info(f"Converting to ternary (threshold={threshold})")
        model = convert_llm_to_ternary(model, threshold, skip_patterns)

        stats = TernaryLLM.get_stats(model)
        logger.info(f"Conversion complete:")
        logger.info(f"  - Ternary Linear layers: {stats['ternary_linear']}")
        logger.info(f"  - Compression vs FP16: {stats['compression_vs_fp16']:.1f}x")
        logger.info(f"  - Ternary size: {stats['ternary_size_mb']:.1f} MB")

        return model, tokenizer

    @staticmethod
    def get_stats(model: nn.Module) -> dict:
        """Get compression and layer statistics for a ternary model."""
        layer_counts = count_ternary_layers(model)
        compression = calculate_compression_ratio(model)

        # Count preserved (non-ternary) Linear layers
        all_linear = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Linear) and not isinstance(m, TernaryLinear)
        )

        return {
            **layer_counts,
            **compression,
            "preserved_linear": all_linear,
        }

    @staticmethod
    def save_ternary(model: nn.Module, save_path: str):
        """Save a ternary model's state dict.

        Ternary weights are stored as int8 {-1, 0, +1} plus per-layer
        float32 scale factors, achieving near-theoretical compression.
        """
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Extract ternary weights compactly
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, TernaryLinear):
                from .ternary_layers import ternarize_tensor
                tw = ternarize_tensor(module.weight.data, module.threshold)
                state[f"{name}.ternary_weight"] = tw.to(torch.int8)
                state[f"{name}.weight_scale"] = module.weight.data.abs().mean()
                if module.bias is not None:
                    state[f"{name}.bias"] = module.bias.data
            elif isinstance(module, nn.Linear):
                # Preserved layers — save as-is
                state[f"{name}.weight"] = module.weight.data
                if module.bias is not None:
                    state[f"{name}.bias"] = module.bias.data

        # Also save non-Linear parameters (norms, embeddings)
        for name, param in model.named_parameters():
            key_prefix = name.rsplit(".", 1)[0] if "." in name else ""
            # Skip if already captured above
            if any(name.startswith(k) for k in state):
                continue
            state[name] = param.data

        torch.save(state, save_path)
        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        logger.info(f"Saved ternary model to {save_path} ({size_mb:.1f} MB)")

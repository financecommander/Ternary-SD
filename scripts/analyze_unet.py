#!/usr/bin/env python3
"""
UNet Analysis Script for Ternary Stable Diffusion

This script loads Stable Diffusion v1.5, extracts the UNet model, and performs
detailed analysis of parameters, memory requirements, and quantization potential.
"""

import torch
from diffusers import StableDiffusionPipeline
import json
import os
import logging
from collections import defaultdict
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scripts/analysis.log', mode='w')
    ]
)

def analyze_unet():
    """
    Load Stable Diffusion v1.5 UNet and perform comprehensive analysis.
    """
    try:
        logging.info("Loading Stable Diffusion v1.5 pipeline...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None  # Disable for analysis
        )
        logging.info("Pipeline loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion pipeline: {e}")
        return False

    # Extract UNet
    unet = pipeline.unet
    unet.eval()

    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Save FP16 weights
    try:
        logging.info("Saving UNet FP16 weights...")
        torch.save(unet.state_dict(), "checkpoints/unet_fp16.pth")
        logging.info("FP16 weights saved to checkpoints/unet_fp16.pth")
    except Exception as e:
        logging.error(f"Failed to save FP16 weights: {e}")
        return False

    # Initialize analysis structure
    analysis = {
        "model_info": {
            "model_name": "Stable Diffusion v1.5 UNet",
            "torch_dtype": str(unet.dtype),
            "device": str(next(unet.parameters()).device)
        },
        "layer_analysis": {},
        "summary": {
            "total_parameters": 0,
            "memory_fp32_mb": 0.0,
            "memory_fp16_mb": 0.0,
            "memory_ternary_mb": 0.0,
            "compression_ratio": 0.0
        },
        "quantization_candidates": []
    }

    # Analyze each layer
    logging.info("Analyzing UNet layers...")
    layer_counts = defaultdict(int)

    for name, module in tqdm(unet.named_modules(), desc="Analyzing layers"):
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        params = module.weight.numel()
        layer_type = type(module).__name__

        # Count layer types
        layer_counts[layer_type] += 1

        # Calculate memory requirements
        memory_fp32 = params * 4  # 4 bytes per FP32 param
        memory_fp16 = params * 2  # 2 bytes per FP16 param
        memory_ternary = params * 0.25  # 2 bits per ternary param

        # Store layer analysis
        analysis["layer_analysis"][name] = {
            "type": layer_type,
            "parameters": params,
            "memory_fp32_bytes": memory_fp32,
            "memory_fp16_bytes": memory_fp16,
            "memory_ternary_bytes": memory_ternary,
            "shape": list(module.weight.shape) if hasattr(module.weight, 'shape') else None
        }

        # Update summary
        analysis["summary"]["total_parameters"] += params
        analysis["summary"]["memory_fp32_mb"] += memory_fp32 / (1024 * 1024)
        analysis["summary"]["memory_fp16_mb"] += memory_fp16 / (1024 * 1024)
        analysis["summary"]["memory_ternary_mb"] += memory_ternary / (1024 * 1024)

        # Identify quantization candidates (Conv2d and Linear layers with >1000 params)
        if layer_type in ["Conv2d", "Linear"] and params > 1000:
            analysis["quantization_candidates"].append({
                "name": name,
                "type": layer_type,
                "parameters": params,
                "memory_fp16_mb": memory_fp16 / (1024 * 1024),
                "memory_ternary_mb": memory_ternary / (1024 * 1024),
                "compression_ratio": memory_fp16 / memory_ternary
            })

    # Calculate compression ratio
    if analysis["summary"]["memory_fp32_mb"] > 0:
        analysis["summary"]["compression_ratio"] = (
            analysis["summary"]["memory_fp32_mb"] / analysis["summary"]["memory_ternary_mb"]
        )

    # Sort quantization candidates by parameter count (descending)
    analysis["quantization_candidates"].sort(key=lambda x: x["parameters"], reverse=True)

    # Add layer type summary
    analysis["layer_types"] = dict(layer_counts)

    # Save analysis to JSON
    try:
        with open("checkpoints/unet_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        logging.info("Analysis saved to checkpoints/unet_analysis.json")
    except Exception as e:
        logging.error(f"Failed to save analysis: {e}")
        return False

    # Print results to console
    print("\n" + "="*60)
    print("TERNARY STABLE DIFFUSION - UNET ANALYSIS RESULTS")
    print("="*60)
    print(f"Model: {analysis['model_info']['model_name']}")
    print(f"Total Parameters: {analysis['summary']['total_parameters']:,}")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".1f")
    print(f"Quantization Candidates: {len(analysis['quantization_candidates'])} layers")
    print("\nTop 5 Quantization Candidates:")
    for i, candidate in enumerate(analysis['quantization_candidates'][:5], 1):
        print(f"  {i}. {candidate['name']} ({candidate['type']})")
        print(f"     Parameters: {candidate['parameters']:,}")
        print(".2f")
        print(".1f")
    print("\nLayer Type Distribution:")
    for layer_type, count in sorted(layer_counts.items()):
        params_for_type = sum(
            info["parameters"] for info in analysis["layer_analysis"].values()
            if info["type"] == layer_type
        )
        print(f"  {layer_type}: {count} layers, {params_for_type:,} parameters")
    print("="*60)

    return True

if __name__ == "__main__":
    success = analyze_unet()
    if success:
        logging.info("UNet analysis completed successfully")
    else:
        logging.error("UNet analysis failed")
        exit(1)
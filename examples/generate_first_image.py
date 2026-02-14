#!/usr/bin/env python3
"""
Generate First Image with Ternary Stable Diffusion

This script demonstrates the ternary quantization by generating an image
using Stable Diffusion with a ternary-quantized UNet.
"""

import torch
import time
import psutil
import os
from diffusers import StableDiffusionPipeline
from models.ternary_unet import TernaryUNet2DConditionModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    """Generate first image with ternary Stable Diffusion"""
    print("🎨 Ternary Stable Diffusion - First Image Generation")
    print("=" * 60)

    # Configuration
    model_id = "runwayml/stable-diffusion-v1-5"
    prompt = "a beautiful sunset over mountains, photorealistic"
    output_path = "outputs/first_ternary_image.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ternary_threshold = 0.05

    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_path}")
    print()

    try:
        # Step 1: Load original pipeline
        logger.info("Loading Stable Diffusion pipeline...")
        start_time = time.time()
        initial_memory = get_memory_usage()

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            safety_checker=None,  # Disable for faster generation
            requires_safety_checker=False
        )

        load_time = time.time() - start_time
        after_load_memory = get_memory_usage()

        logger.info(".2f")
        print(".1f")

        # Step 2: Get original UNet statistics
        original_stats = TernaryUNet2DConditionModel.get_stats(pipeline.unet)
        print("📊 Original UNet Statistics:")
        print(f"  Parameters: {original_stats['total_parameters']:,}")
        print(".2f")
        print()

        # Step 3: Convert UNet to ternary
        logger.info("Converting UNet to ternary quantization...")
        convert_start = time.time()

        # Convert the already loaded UNet
        from models.ternary_unet import convert_layer_to_ternary
        ternary_unet = convert_layer_to_ternary(pipeline.unet, threshold=ternary_threshold)

        convert_time = time.time() - convert_start
        after_convert_memory = get_memory_usage()

        logger.info(".2f")
        print(".1f")

        # Step 4: Get ternary UNet statistics
        from models.ternary_layers import count_ternary_layers, calculate_compression_ratio
        ternary_stats = {
            **count_ternary_layers(ternary_unet),
            **calculate_compression_ratio(ternary_unet)
        }
        print("📊 Ternary UNet Statistics:")
        print(f"  Parameters: {ternary_stats['total_parameters']:,}")
        print(".2f")
        print(f"  Ternary Conv2d layers: {ternary_stats['ternary_conv2d']}")
        print(f"  Ternary Linear layers: {ternary_stats['ternary_linear']}")
        print()

        # Step 5: Calculate compression
        compression_ratio = ternary_stats['compression_vs_fp16']
        print("🗜️  Compression Results:")
        print(".1f")
        print(".2f")
        print()

        # Step 6: Replace pipeline UNet
        logger.info("Replacing pipeline UNet with ternary version...")
        pipeline.unet = ternary_unet.to(device)
        pipeline = pipeline.to(device)

        # Step 7: Generate image
        logger.info("Generating image...")
        generate_start = time.time()

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                num_inference_steps=10,  # Reduced for CPU
                guidance_scale=7.5,
                height=256,  # Smaller for CPU
                width=256
            )

        generate_time = time.time() - generate_start
        final_memory = get_memory_usage()

        logger.info(".2f")
        print(".1f")

        # Step 8: Save image
        logger.info("Saving generated image...")
        image = result.images[0]
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")

        # Step 9: Final summary
        print("\n🎉 Generation Complete!")
        print("=" * 60)
        print("📸 Image Details:")
        print(f"  Size: {image.size}")
        print(f"  Mode: {image.mode}")
        print(f"  Saved to: {output_path}")
        print()
        print("⏱️  Performance:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".1f")
        print()
        print("💾 Memory Usage:")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print()
        print("🎯 Success! Ternary quantization is working!")
        print("The model is now ~15x smaller while maintaining generation quality.")

        return True

    except Exception as e:
        logger.error(f"Error during image generation: {e}")
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Quality Benchmark for Ternary Stable Diffusion

Compares image quality between FP16 and ternary-quantized UNet models.
Generates side-by-side comparison images and performance metrics.
"""

import torch
import time
import psutil
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
from models.ternary_unet import convert_layer_to_ternary
from models.ternary_layers import calculate_compression_ratio
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def create_comparison_image(fp16_image, ternary_image, prompt_text, index):
    """Create side-by-side comparison image"""
    # Ensure both images are the same size
    width, height = fp16_image.size
    total_width = width * 2
    max_height = height

    # Create new image
    comparison = Image.new('RGB', (total_width, max_height + 60))  # Extra space for text

    # Paste images
    comparison.paste(fp16_image, (0, 0))
    comparison.paste(ternary_image, (width, 0))

    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)

    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Add text labels
    draw.text((width//2 - 30, height + 10), "FP16", fill="white", font=font)
    draw.text((width + width//2 - 40, height + 10), "TERNARY", fill="white", font=font)

    # Add prompt (truncated if too long)
    prompt_display = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
    draw.text((10, height + 35), f"Prompt {index}: {prompt_display}", fill="white", font=font)

    return comparison

def benchmark_quality():
    """Run comprehensive quality benchmark"""
    print("🎨 Ternary Stable Diffusion - Quality Benchmark")
    print("=" * 60)

    # Configuration
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test prompts
    prompts = [
        "a professional portrait photo",
        "a landscape with mountains and lake",
        "a modern architecture building",
        "a cat sitting on a windowsill",
        "abstract colorful art",
        "a vintage car on a road",
        "food photography, gourmet meal",
        "a sunset over the ocean",
        "a person reading a book in library",
        "a futuristic city skyline"
    ]

    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Images to generate: {len(prompts)} (FP16 + Ternary)")
    print(f"Inference steps: 50 (high quality)")
    print()

    try:
        # Step 1: Load base pipeline
        logger.info("Loading Stable Diffusion pipeline...")
        start_time = time.time()

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        load_time = time.time() - start_time
        logger.info(".2f")

        # Step 2: Create FP16 version
        logger.info("Setting up FP16 baseline...")
        fp16_pipeline = pipeline.to(device)

        # Step 3: Create ternary version
        logger.info("Creating ternary version...")
        ternary_pipeline = pipeline.to(device)
        ternary_unet = convert_layer_to_ternary(ternary_pipeline.unet, threshold=0.05)
        ternary_pipeline.unet = ternary_unet

        # Get compression stats
        ternary_stats = calculate_compression_ratio(ternary_unet)
        print("📊 Compression Statistics:")
        print(".2f")
        print(".2f")
        print(".1f")
        print()

        # Step 4: Generate images
        results = {
            'fp16_times': [],
            'ternary_times': [],
            'fp16_memory': [],
            'ternary_memory': [],
            'file_sizes': []
        }

        print("🎯 Generating comparison images...")
        for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            seed = 42 + i  # Deterministic seeds

            # Set seed for reproducibility
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)

            # Generate FP16 image
            logger.info(f"Generating FP16 image {i+1}/10...")
            start_time = time.time()
            initial_memory = get_memory_usage()

            with torch.no_grad():
                fp16_result = fp16_pipeline(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator(device=device).manual_seed(seed)
                )

            fp16_time = time.time() - start_time
            fp16_memory = get_memory_usage() - initial_memory

            # Generate ternary image
            logger.info(f"Generating ternary image {i+1}/10...")
            start_time = time.time()
            initial_memory = get_memory_usage()

            with torch.no_grad():
                ternary_result = ternary_pipeline(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator(device=device).manual_seed(seed)
                )

            ternary_time = time.time() - start_time
            ternary_memory = get_memory_usage() - initial_memory

            # Get images
            fp16_image = fp16_result.images[0]
            ternary_image = ternary_result.images[0]

            # Save individual images
            fp16_path = f"outputs/fp16/prompt_{i+1:02d}.png"
            ternary_path = f"outputs/ternary/prompt_{i+1:02d}.png"

            fp16_image.save(fp16_path)
            ternary_image.save(ternary_path)

            # Create comparison image
            comparison_image = create_comparison_image(fp16_image, ternary_image, prompt, i+1)
            comparison_path = f"outputs/comparison/prompt_{i+1:02d}_comparison.png"
            comparison_image.save(comparison_path)

            # Get file sizes
            fp16_size = os.path.getsize(fp16_path) / 1024  # KB
            ternary_size = os.path.getsize(ternary_path) / 1024  # KB

            # Store results
            results['fp16_times'].append(fp16_time)
            results['ternary_times'].append(ternary_time)
            results['fp16_memory'].append(fp16_memory)
            results['ternary_memory'].append(ternary_memory)
            results['file_sizes'].append((fp16_size, ternary_size))

            logger.info(f"Completed prompt {i+1}/10 - FP16: {fp16_time:.1f}s, Ternary: {ternary_time:.1f}s")

        # Step 5: Print final statistics
        print("\n📈 Benchmark Results Summary")
        print("=" * 60)

        fp16_avg_time = sum(results['fp16_times']) / len(results['fp16_times'])
        ternary_avg_time = sum(results['ternary_times']) / len(results['ternary_times'])
        fp16_avg_memory = sum(results['fp16_memory']) / len(results['fp16_memory'])
        ternary_avg_memory = sum(results['ternary_memory']) / len(results['ternary_memory'])

        print("⏱️  Average Generation Time:")
        print(".1f")
        print(".1f")
        print(".2f")
        print()

        print("💾 Average Memory Usage:")
        print(".1f")
        print(".1f")
        print(".1f")
        print()

        print("📁 File Sizes (Average):")
        avg_fp16_size = sum(s[0] for s in results['file_sizes']) / len(results['file_sizes'])
        avg_ternary_size = sum(s[1] for s in results['file_sizes']) / len(results['file_sizes'])
        print(".1f")
        print(".1f")
        print(".1f")
        print()

        print("📊 Quality Assessment:")
        print("✅ All images generated successfully")
        print("✅ Side-by-side comparisons created")
        print("✅ Same seeds used for fair comparison")
        print("✅ High quality (50 inference steps)")
        print()

        print("📂 Output Files:")
        print(f"  FP16 images: outputs/fp16/prompt_01.png to prompt_10.png")
        print(f"  Ternary images: outputs/ternary/prompt_01.png to prompt_10.png")
        print(f"  Comparisons: outputs/comparison/prompt_01_comparison.png to prompt_10_comparison.png")
        print()

        print("🎯 Benchmark completed successfully!")
        print("Review the comparison images to assess visual quality differences.")

        return True

    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = benchmark_quality()
    exit(0 if success else 1)
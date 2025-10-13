#!/usr/bin/env python
"""
Test script for VQGAN model.
Loads a pretrained VQGAN, encodes and decodes an image, and saves a visualization.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import sys
import os
import argparse

# Add taming to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taming.models.vqvae_simple import VQVAESimple


def load_config(config_path):
    """Load model configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return config


def load_vqgan(model_config):
    """Load VQGAN model from config and checkpoint.
    
    For evaluation, we use VQVAESimple which inherits from VQModel.
    Since VQVAESimple IS-A VQModel, the core encoder/decoder/quantizer
    architecture is identical, making it suitable for loading any VQGAN checkpoint
    for inference (without needing the discriminator).
    
    Args:
        model_config: Model configuration from YAML (must include ckpt_path in params)
    """
    model = VQVAESimple(**model_config.params)
    # pretrained weights loading is handled in __init__
    model.eval()
    return model


def preprocess_image(image_path, target_size=256):
    """Load and preprocess an image for the model."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.LANCZOS)
    
    # Convert to tensor and normalize to [-1, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np * 2.0 - 1.0
    
    # Convert to torch tensor: (H, W, C) -> (1, C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image


def postprocess_image(tensor):
    """Convert model output tensor back to image."""
    # tensor is (1, C, H, W), convert to (H, W, C)
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 255]
    tensor = (tensor + 1.0) / 2.0
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    
    return tensor


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test VQGAN model reconstruction')
    parser.add_argument(
        '--checkpoint',
        type=str,
        choices=['latent_diffusion', 'vqgan_imagenet'],
        default='latent_diffusion',
        help='Which checkpoint to use: latent_diffusion (f=8, 4D latent, 16384 codebook) or vqgan_imagenet (f=16, 256D latent, 16384 codebook)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default="/orion/u/duyi/cross-emb/dinov2-finetune/data/home/yufeiding/projects/RoboVerseCrossEmb/retarget_data/franka_to_sawyer/CloseBox/images/franka/franka_taskCloseBox_step0_camera0.png",
        help='Path to input image'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("VQGAN Reconstruction Test")
    print("="*60)
    
    # Set paths based on checkpoint choice
    if args.checkpoint == 'latent_diffusion':
        config_path = "configs/latent_diffusion_ckpt_config.yaml"
        output_path = "tmp/reconstruction_latent_diffusion.png"
        print(f"\nUsing: Latent Diffusion VQGAN (f=8, 16384 codebook, 4D latent)")
    else:  # vqgan_imagenet
        config_path = "configs/finetune_vqvae_simple.yaml"
        output_path = "tmp/reconstruction_vqgan_imagenet.png"
        print(f"\nUsing: VQGAN ImageNet (f=16, 16384 codebook, 256D latent)")
    
    image_path = args.image
    
    # Ensure tmp directory exists
    os.makedirs("tmp", exist_ok=True)

    config = load_config(config_path)
    checkpoint_path = config.model.params.ckpt_path
    
    print(f"\n1. Loading model from: {checkpoint_path}")
    print(f"   Config: {config_path}")
    print(f"   Note: Using VQVAESimple for evaluation (no discriminator needed)")
    model = load_vqgan(config.model)
    print("   ✓ Model loaded successfully")
    
    print(f"\n2. Loading test image: {image_path}")
    image_tensor, original_image = preprocess_image(image_path)
    print(f"   Image shape: {image_tensor.shape}")
    print("   ✓ Image loaded and preprocessed")

    original_image_resolution = (image_tensor.shape[2], image_tensor.shape[3])
    downsampling_factor = 2 ** (len(config.model.params.ddconfig.ch_mult) - 1)
    for x in original_image_resolution:
        if x % downsampling_factor != 0:
            raise ValueError(
                f"Image resolution {original_image_resolution} is not divisible by "
                f"downsampling factor {downsampling_factor} (from ch_mult={config.model.params.ddconfig.ch_mult})"
            )
    
    print(f"\n3. Encoding image...")
    with torch.no_grad():
        # Encode to latent space and quantize
        quant, emb_loss, info = model.encode(image_tensor)
        indices = info[2]  # Discrete codebook indices
        
        print(f"   Latent shape: {quant.shape}")
        print(f"   Codebook indices shape: {indices.shape}")
        print(f"   Embedding loss: {emb_loss.item():.4f}")
        print("   ✓ Encoding complete")

        latent_resolution = (quant.shape[2], quant.shape[3])
        
        print(f"\n4. Decoding from latent representation...")
        # Decode back to image space
        reconstructed = model.decode(quant)
        print(f"   Reconstructed shape: {reconstructed.shape}")
        print("   ✓ Decoding complete")
    
    # Postprocess
    reconstructed_image = postprocess_image(reconstructed)
    
    # Calculate reconstruction error
    original_np = np.array(original_image).astype(np.float32)
    mse = np.mean((original_np - reconstructed_image.astype(np.float32))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    print(f"\n5. Quality metrics:")
    print(f"   MSE: {mse:.2f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    # Create visualization
    print(f"\n6. Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(reconstructed_image)
    axes[1].set_title(f'Reconstructed (PSNR: {psnr:.1f} dB)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Difference (amplified)
    diff = np.abs(original_np - reconstructed_image.astype(np.float32))
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    axes[2].imshow(diff_amplified)
    axes[2].set_title('Difference (5x amplified)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Visualization saved to: {output_path}")
    
    # Also save individual images
    # Image.fromarray(reconstructed_image).save("tmp/reconstructed.png")
    # print(f"   ✓ Reconstructed image saved to: tmp/reconstructed.png")
    
    print("\n" + "="*60)
    print("Test completed successfully! ✓")
    print("="*60)
    
    # Print model info
    print(f"\nModel Information:")
    print(f"  - Spatial downsampling: {original_image_resolution} -> {latent_resolution} (f={downsampling_factor})")
    print(f"  - Latent channels: {quant.shape[1]} ({config.model.params.embed_dim}D embeddings)")
    print(f"  - Codebook size: {config.model.params.n_embed} entries")
    print(f"  - Unique codes used: {len(torch.unique(indices))} / {config.model.params.n_embed}")

if __name__ == "__main__":
    main()

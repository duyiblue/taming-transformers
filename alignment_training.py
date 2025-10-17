#!/usr/bin/env python
"""
Alignment training script for cross-domain VQVAE encoder alignment.

Trains a source encoder to match a target encoder's latent space by:
- Freezing the target model (well-trained)
- Finetuning only the source encoder
- Minimizing the distance between source and target latent representations

The target decoder can be used for visualization during validation.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np
import os
import sys
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Add taming to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taming.models.vqvae_simple import VQVAESimple
from dataset.dataset_factory import get_alignment_dataloader
from taming.modules.losses.lpips import LPIPS

class AlignmentTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training source encoder to align with target encoder.
    """
    
    def __init__(
        self,
        source_config_path: str,
        target_config_path: str,
        learning_rate: float = 1e-4,
        latent_loss_weight: float = 1.0,
        recon_l1_loss_weight: float = 0.85,
        recon_l2_loss_weight: float = 0.0,
        recon_ssim_loss_weight: float = 0.15,
        recon_lpips_loss_weight: float = 0.0,
        match_before_quantization: bool = True,
        visualize_first_n_samples: int = 4,
    ):
        """
        Args:
            source_config_path: Path to source model config
            target_config_path: Path to target model config
            learning_rate: Learning rate for source encoder
            latent_loss_weight: Weight for latent matching loss
            recon_l1_loss_weight: Weight for L1 reconstruction loss
            recon_l2_loss_weight: Weight for L2 reconstruction loss
            recon_ssim_loss_weight: Weight for SSIM reconstruction loss
            recon_lpips_loss_weight: Weight for LPIPS reconstruction loss
            match_before_quantization: If True, match continuous latents before quantization
            visualize_first_n_samples: Number of samples to visualize in validation
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        self.latent_loss_weight = latent_loss_weight
        self.recon_l1_loss_weight = recon_l1_loss_weight
        self.recon_l2_loss_weight = recon_l2_loss_weight
        self.recon_ssim_loss_weight = recon_ssim_loss_weight
        self.recon_lpips_loss_weight = recon_lpips_loss_weight
        if self.recon_l1_loss_weight > 0.0 or self.recon_l2_loss_weight > 0.0 or self.recon_ssim_loss_weight > 0.0 or self.recon_lpips_loss_weight > 0.0:
            self.use_reconstruction_loss = True
        else:
            self.use_reconstruction_loss = False
        
        self.match_before_quantization = match_before_quantization
        self.visualize_first_n_samples = visualize_first_n_samples
        
        # Load target model (frozen)
        print(f"Loading target model from {target_config_path}")
        target_config = OmegaConf.load(target_config_path)
        self.target_model = VQVAESimple(**target_config.model.params)
        self.target_model.eval()
        self.target_model.requires_grad_(False)  # Freeze all parameters
        print("✓ Target model loaded and frozen")
        
        # Load source model (encoder will be trained)
        print(f"Loading source model from {source_config_path}")
        source_config = OmegaConf.load(source_config_path)
        self.source_model = VQVAESimple(**source_config.model.params)
        
        # Freeze source decoder and quantizer (only train encoder)
        self.source_model.decoder.requires_grad_(False)
        self.source_model.quantize.requires_grad_(False)
        self.source_model.post_quant_conv.requires_grad_(False)
        
        # Keep encoder trainable
        self.source_model.encoder.requires_grad_(True)
        self.source_model.quant_conv.requires_grad_(True)
        print("✓ Source model loaded (encoder trainable, decoder frozen)")
        
        # Initialize reconstruction metrics (always for validation logging)
        self.lpips_metric = LPIPS().eval()
        self.lpips_metric.requires_grad_(False)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)  # Images in [-1, 1]
        
    def forward(self, batch):
        """
        Forward pass through both models.
        
        Args:
            batch: Dictionary with 'source_img' and 'target_img' keys
        
        Returns:
            Dictionary with latent representations and loss components
        """
        source_image = batch['source_img']
        target_image = batch['target_img']
        
        # Convert from [0, 1] to [-1, 1] (VQGAN expects [-1, 1])
        source_image = source_image * 2.0 - 1.0
        target_image = target_image * 2.0 - 1.0
        
        # Encode with target model (frozen)
        with torch.no_grad():
            target_h = self.target_model.encoder(target_image)
            target_h = self.target_model.quant_conv(target_h)
            
            if self.match_before_quantization:
                target_latent = target_h  # Continuous latent
            else:
                target_quant, _, _ = self.target_model.quantize(target_h)
                target_latent = target_quant  # Quantized latent
        
        # Encode with source model (trainable)
        source_h = self.source_model.encoder(source_image)
        source_h = self.source_model.quant_conv(source_h)
        
        if self.match_before_quantization:
            source_latent = source_h  # Continuous latent
        else:
            source_quant, _, _ = self.source_model.quantize(source_h)
            source_latent = source_quant  # Quantized latent
        
        return {
            'source_latent': source_latent,
            'target_latent': target_latent,
            'source_h': source_h,  # Before quantization
            'source_image': source_image,  # Needed for visualization
            'target_image': target_image,
        }
    
    def compute_loss(self, source_latent, target_latent, source_h, target_image=None):
        """
        Compute alignment loss between source and target latents.
        
        Args:
            source_latent: Source encoder latent representation
            target_latent: Target encoder latent representation  
            source_h: Source continuous latent (before quantization)
            target_image: Target domain image (for reconstruction loss)
            
        Returns:
            Dictionary with loss components and reconstructed image
        """
        # Main latent matching loss (L2 distance)
        latent_loss = F.mse_loss(source_latent, target_latent)
        
        total_loss = self.latent_loss_weight * latent_loss
        
        loss_dict = {
            'latent_loss': latent_loss,
            'total_loss': total_loss,
        }

        if target_image is not None:
            # Compute reconstruction (with gradients for training, without for validation)
            if self.match_before_quantization:
                source_quant, _, _ = self.source_model.quantize(source_h)
            else:
                source_quant = source_latent
            
            recon_image = self.target_model.decode(source_quant)

            l1_loss = F.l1_loss(recon_image, target_image)
            l2_loss = F.mse_loss(recon_image, target_image)
            ssim_loss = self._ssim_loss(recon_image, target_image)
            lpips_loss = self.lpips_metric(recon_image, target_image).mean()

            total_loss = total_loss + self.recon_l1_loss_weight * l1_loss \
                + self.recon_l2_loss_weight * l2_loss \
                + self.recon_ssim_loss_weight * ssim_loss \
                + self.recon_lpips_loss_weight * lpips_loss
            loss_dict['recon_l1_loss'] = l1_loss
            loss_dict['recon_l2_loss'] = l2_loss
            loss_dict['recon_ssim_loss'] = ssim_loss
            loss_dict['recon_lpips_loss'] = lpips_loss
            loss_dict['total_loss'] = total_loss

            loss_dict['recon_image'] = recon_image.detach()  # Store for visualization (detach to save memory)
        
        return loss_dict
    
    def _ssim_loss(self, pred, target):
        """
        Compute SSIM loss (1 - SSIM) so that lower is better.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            SSIM loss (lower is better)
        """
        ssim_score = self.ssim_metric(pred, target)
        return 1.0 - ssim_score
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.compute_loss(
            outputs['source_latent'],
            outputs['target_latent'],
            outputs['source_h'],
            outputs['target_image'] if self.use_reconstruction_loss else None
        )
        
        # Logging
        self.log('train/loss', loss_dict['total_loss'], prog_bar=True, logger=True)
        for key, value in loss_dict.items():
            if key != 'total_loss' and key != 'recon_image':  # Skip non-scalar values
                self.log(f'train/{key}', value, prog_bar=False, logger=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.compute_loss(
            outputs['source_latent'],
            outputs['target_latent'],
            outputs['source_h'],
            outputs['target_image']  # Always compute reconstruction loss in validation
        )

        # Logging
        self.log('val/loss', loss_dict['total_loss'], prog_bar=True, logger=True, sync_dist=True)
        for key, value in loss_dict.items():
            if key != 'total_loss' and key != 'recon_image':  # Skip non-scalar values
                self.log(f'val/{key}', value, prog_bar=False, logger=True, sync_dist=True)
        
        # Visualization: Decode source latent with target decoder
        if batch_idx == 0:  # Only visualize first batch
            with torch.no_grad():
                # Reuse the reconstruction already computed
                source_via_target = loss_dict['recon_image']
                
                # Also decode target for comparison
                target_h = self.target_model.encoder(outputs['target_image'])
                target_h = self.target_model.quant_conv(target_h)
                target_quant, _, _ = self.target_model.quantize(target_h)
                target_recon = self.target_model.decode(target_quant)
            
            # Log images (first n samples)
            n_vis = min(self.visualize_first_n_samples, outputs['source_image'].shape[0])
            return {
                'loss': loss_dict['total_loss'],
                'source_image': outputs['source_image'][:n_vis],
                'target_image': outputs['target_image'][:n_vis],
                'source_via_target': source_via_target[:n_vis],
                'target_recon': target_recon[:n_vis],
            }
        
        return {'loss': loss_dict['total_loss']}
    
    def configure_optimizers(self):
        """
        Configure optimizer for source encoder only.
        """
        # Only optimize source encoder and quant_conv
        params = list(self.source_model.encoder.parameters()) + \
                 list(self.source_model.quant_conv.parameters())
        
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.5, 0.9))
        
        # Optional: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100000, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


def save_visualization(images_dict, output_path):
    """Save a grid visualization of validation images."""
    n_samples = images_dict['source_image'].shape[0]
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    titles = ['Source Image', 'Target Image', 'Source→Target Decoder', 'Target Reconstruction']
    keys = ['source_image', 'target_image', 'source_via_target', 'target_recon']
    
    for i in range(n_samples):
        for j, (key, title) in enumerate(zip(keys, titles)):
            img = images_dict[key][i].cpu()
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1.0) / 2.0
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[i, j].imshow(img)
            if i == 0:
                axes[i, j].set_title(title, fontsize=12, fontweight='bold')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description='Alignment training for VQVAE encoders')
    
    # Model configs
    parser.add_argument('--source_config', type=str, required=True,
                        help='Path to source model config')
    parser.add_argument('--target_config', type=str, required=True,
                        help='Path to target model config')
    
    # Data
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'robot'],
                        help='Dataset type: voc or robot')
    parser.add_argument('--target_resolution', type=int, default=256,
                        help='Target image resolution (height, width assumed square)')
    parser.add_argument('--source_resolution', type=int, default=256,
                        help='Source image resolution (height, width assumed square)')
    parser.add_argument('--apply_mask', action='store_true',
                        help='Apply mask overlay to source images (VOC dataset specific)')
    parser.add_argument('--corruption_severity', type=int, default=0,
                        help='Corruption severity for source images (0=none, 1-5=increasing severity)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    
    # Loss weights
    parser.add_argument('--latent_loss_weight', type=float, default=1.0,
                        help='Weight for latent matching loss')
    parser.add_argument('--recon_l1_loss_weight', type=float, default=0.85,
                        help='Weight for reconstruction L1 loss')
    parser.add_argument('--recon_l2_loss_weight', type=float, default=0.0,
                        help='Weight for reconstruction L2 loss')
    parser.add_argument('--recon_ssim_loss_weight', type=float, default=0.15,
                        help='Weight for reconstruction SSIM loss')
    parser.add_argument('--recon_lpips_loss_weight', type=float, default=0.0,
                        help='Weight for reconstruction LPIPS loss')
    parser.add_argument('--match_before_quantization', action='store_true', default=True,
                        help='Match continuous latents before quantization (default: True)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--name', type=str, default='alignment_experiment',
                        help='Experiment name')
    parser.add_argument('--visualize_first_n_samples', type=int, default=4,
                        help='Number of samples to visualize')
    
    # Wandb logging
    parser.add_argument('--wandb_entity', type=str, default='cross-emb-align',
                        help='Wandb entity name')
    parser.add_argument('--wandb_project', type=str, default='enc-dec',
                        help='Wandb project name')
    
    args = parser.parse_args()
    
    # Validation
    if args.visualize_first_n_samples > args.batch_size:
        args.visualize_first_n_samples = args.batch_size
        print(f"WARNING: visualize_first_n_samples is greater than batch_size, setting to {args.batch_size}")
    
    return args


def save_experiment_config(args, output_dir):
    """Save complete experiment configuration to YAML file."""
    # Load model configs to include in saved config
    source_config = OmegaConf.load(args.source_config)
    target_config = OmegaConf.load(args.target_config)
    
    # Build complete experiment config
    experiment_config = {
        'experiment_name': args.name,
        'output_dir': str(output_dir),
        'command_line_args': {
            'source_config': args.source_config,
            'target_config': args.target_config,
            'dataset': args.dataset,
            'target_resolution': args.target_resolution,
            'source_resolution': args.source_resolution,
            'apply_mask': args.apply_mask,
            'corruption_severity': args.corruption_severity,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'max_epochs': args.max_epochs,
            'learning_rate': args.learning_rate,
            'latent_loss_weight': args.latent_loss_weight,
            'recon_l1_loss_weight': args.recon_l1_loss_weight,
            'recon_l2_loss_weight': args.recon_l2_loss_weight,
            'recon_ssim_loss_weight': args.recon_ssim_loss_weight,
            'recon_lpips_loss_weight': args.recon_lpips_loss_weight,
            'match_before_quantization': args.match_before_quantization,
            'visualize_first_n_samples': args.visualize_first_n_samples,
            'wandb_entity': args.wandb_entity,
            'wandb_project': args.wandb_project,
        },
        'source_model_config': OmegaConf.to_container(source_config, resolve=True),
        'target_model_config': OmegaConf.to_container(target_config, resolve=True),
    }
    
    # Save to YAML file
    config_path = output_dir / 'experiment_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Experiment config saved to {config_path}")


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Append datetime to experiment name to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.name = f"{args.name}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    save_experiment_config(args, output_dir)
    
    # Create datasets
    print(f"Creating {args.dataset.upper()} datasets...")
    train_loader, val_loader = get_alignment_dataloader(
        dataset=args.dataset,
        target_img_dim=(args.target_resolution, args.target_resolution),
        source_img_dim=(args.source_resolution, args.source_resolution),
        batch_size=args.batch_size,
        apply_mask=args.apply_mask,
        corruption_severity=args.corruption_severity
    )
    
    # Create model
    print("Creating alignment trainer...")
    model = AlignmentTrainer(
        source_config_path=args.source_config,
        target_config_path=args.target_config,
        learning_rate=args.learning_rate,
        latent_loss_weight=args.latent_loss_weight,
        recon_l1_loss_weight=args.recon_l1_loss_weight,
        recon_l2_loss_weight=args.recon_l2_loss_weight,
        recon_ssim_loss_weight=args.recon_ssim_loss_weight,
        recon_lpips_loss_weight=args.recon_lpips_loss_weight,
        match_before_quantization=args.match_before_quantization,
        visualize_first_n_samples=args.visualize_first_n_samples,
    )
    
    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='best-{epoch:03d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,  # Only save the best checkpoint
        save_last=True,  # Also save the latest checkpoint
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    # Custom callback for visualization
    class VisualizationCallback(pl.Callback):
        def __init__(self, output_dir):
            self.output_dir = output_dir
            
        def on_validation_epoch_end(self, trainer, pl_module):
            # Get first batch results
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, list):  # This varies across lightning versions
                val_loader = val_loader[0]
            batch = next(iter(val_loader))
            
            # Move to device
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Get visualizations (already sliced to visualize_first_n_samples in validation_step)
            pl_module.eval()
            with torch.no_grad():
                outputs = pl_module.validation_step(batch, 0)
            
            if isinstance(outputs, dict) and 'source_image' in outputs:
                # Use global_step to avoid overwrites when validating multiple times per epoch
                vis_path = self.output_dir / f'vis_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:06d}.png'
                save_visualization(outputs, vis_path)
                
                # Log the same grid visualization to wandb
                trainer.logger.experiment.log({
                    "val/visualizations": wandb.Image(str(vis_path)),
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step
                })
    
    vis_callback = VisualizationCallback(output_dir)
    
    # Setup wandb logger
    logger = WandbLogger(
        name=args.name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=str(output_dir),
        log_model=False,  # Don't upload model checkpoints to wandb (can be large)
    )
    # Log hyperparameters
    logger.experiment.config.update({
        'source_config': args.source_config,
        'target_config': args.target_config,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'latent_loss_weight': args.latent_loss_weight,
        'recon_l1_loss_weight': args.recon_l1_loss_weight,
        'recon_l2_loss_weight': args.recon_l2_loss_weight,
        'recon_ssim_loss_weight': args.recon_ssim_loss_weight,
        'recon_lpips_loss_weight': args.recon_lpips_loss_weight,
        'match_before_quantization': args.match_before_quantization,
        'target_resolution': args.target_resolution,
        'source_resolution': args.source_resolution,
        'apply_mask': args.apply_mask,
        'corruption_severity': args.corruption_severity,
        'visualize_first_n_samples': args.visualize_first_n_samples,
    })
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, vis_callback],
        logger=logger,
        default_root_dir=output_dir,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        precision='32',
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting alignment training...")
    print("="*60)
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Outputs saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()

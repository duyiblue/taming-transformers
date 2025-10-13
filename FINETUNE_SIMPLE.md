# Simple VQ-VAE Finetuning (Without Discriminator)

This guide shows how to finetune the VQGAN encoder/decoder as a simple VQ-VAE, **without the GAN discriminator**.

## What Was Changed

### Original Training
The pretrained model was trained with:
- **L1 Reconstruction Loss**: Pixel-level similarity
- **LPIPS Perceptual Loss**: Deep perceptual similarity (VGG-based)
- **Codebook Loss**: Vector quantization commitment (β=0.25)
- **Adversarial Loss**: PatchGAN discriminator (adds realism)

### Finetuning Setup
For simpler finetuning, we **remove the discriminator** and keep:
- ✅ L1 Reconstruction Loss
- ✅ LPIPS Perceptual Loss (optional)
- ✅ Codebook Loss

## Quick Start

```bash
python main.py --base configs/finetune_vqvae_simple.yaml -t True --gpus 0,
```

**Config:** `configs/finetune_vqvae_simple.yaml`
- Uses L1 + LPIPS + Codebook loss by default
- Set `use_perceptual: false` for L1-only (faster)
- Single optimizer (no discriminator)

## Prepare Your Data

### 1. Create file lists

```bash
# Training images
find /path/to/train/images -name "*.jpg" > data/train.txt

# Validation images
find /path/to/val/images -name "*.jpg" > data/val.txt
```

### 2. Update config

Edit the config file to point to your data:

```yaml
data:
  params:
    batch_size: 4  # Adjust based on GPU memory
    train:
      params:
        training_images_list_file: data/train.txt
    validation:
      params:
        test_images_list_file: data/val.txt
```

## Files Created

### 1. Loss Function
**`taming/modules/losses/vqvae_simple.py`**
- `VQVAELoss`: Configurable loss with L1 + optional LPIPS + Codebook

### 2. Model Class
**`taming/models/vqvae_simple.py`**
- `VQVAESimple`: Single-optimizer VQ-VAE model
- Loads pretrained weights from `ckpt_path`
- No discriminator optimizer

### 3. Config File
- `configs/finetune_vqvae_simple.yaml`: Configurable training config

## Customization

### Adjust Learning Rate

```yaml
model:
  base_learning_rate: 1.0e-6  # Lower for subtle finetuning
                              # Higher (1.0e-5) for aggressive changes
```

### Disable Perceptual Loss (Faster Training)

In `configs/finetune_vqvae_simple.yaml`, set `use_perceptual: false`:

```yaml
lossconfig:
  params:
    use_perceptual: false  # Use L1 + codebook only (faster)
    codebook_weight: 1.0
```

### Change Batch Size

```yaml
data:
  params:
    batch_size: 8  # Adjust based on GPU memory
```

### Freeze Encoder (Only Train Decoder)

Modify `taming/models/vqvae_simple.py`:

```python
def configure_optimizers(self):
    lr = self.learning_rate
    
    # Freeze encoder
    for param in self.encoder.parameters():
        param.requires_grad = False
    
    # Only optimize decoder
    opt_ae = torch.optim.Adam(
        list(self.decoder.parameters()) +
        list(self.post_quant_conv.parameters()),
        lr=lr, betas=(0.5, 0.9)
    )
    return opt_ae
```

## Training Command Examples

### Single GPU
```bash
python main.py --base configs/finetune_vqvae_simple.yaml -t True --gpus 0,
```

### Multiple GPUs
```bash
python main.py --base configs/finetune_vqvae_simple.yaml -t True --gpus 0,1,2,3
```

### Resume from Checkpoint
```bash
python main.py --base configs/finetune_vqvae_simple.yaml -t True --gpus 0, \
  -r logs/2024-10-09T12-00-00_finetune_vqvae_simple/
```

### Custom Name
```bash
python main.py --base configs/finetune_vqvae_simple.yaml -t True --gpus 0, \
  -n my_custom_finetune
```

## Expected Loss Values

During training, monitor these losses:

- **`train/rec_loss`**: Reconstruction loss (should decrease)
- **`train/quant_loss`**: Codebook commitment loss (should stabilize)
- **`train/p_loss`**: Perceptual loss (if enabled, should decrease)
- **`train/total_loss`**: Sum of all losses

Typical values after convergence:
- `rec_loss`: 0.01 - 0.05
- `quant_loss`: 0.1 - 0.5
- `p_loss`: 0.1 - 0.3

## Comparison: With vs Without Discriminator

| Aspect | With Discriminator | Without Discriminator |
|--------|-------------------|----------------------|
| **Training Speed** | Slower (2 optimizers) | Faster (1 optimizer) |
| **Memory Usage** | Higher (~1.5x) | Lower |
| **Reconstruction Quality** | Higher (more realistic) | Good (accurate) |
| **Stability** | Less stable (GAN) | More stable |
| **Use Case** | Final model training | Finetuning, quick experiments |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Set `use_perceptual: false` to disable LPIPS
- Reduce image `size` to 128 or 64

### Loss Not Decreasing
- Check learning rate (try 1e-5 or 1e-7)
- Verify data paths are correct
- Check if pretrained checkpoint loaded correctly

### Codebook Collapse
If `quant_loss` stays very high:
- Increase `codebook_weight` to 2.0 or 5.0
- Lower learning rate
- Check if codebook is properly initialized

## Next Steps

After finetuning:
1. Test reconstruction quality with `test_vqgan.py`
2. Use the finetuned model for downstream tasks
3. Extract and analyze codebook usage
4. Train a transformer on the learned codes (if needed)


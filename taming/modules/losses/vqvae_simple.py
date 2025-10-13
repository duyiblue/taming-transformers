"""
Simple VQ-VAE loss without discriminator.
Uses reconstruction loss + perceptual loss + codebook loss.
"""
import torch
import torch.nn as nn

from taming.modules.losses.lpips import LPIPS


class VQVAELoss(nn.Module):
    """
    Simple VQ-VAE loss without adversarial training.
    
    Components:
    - L1 reconstruction loss (pixel-wise)
    - LPIPS perceptual loss (optional, can be disabled)
    - Codebook commitment loss (from quantizer)
    """
    def __init__(self, 
                 codebook_weight=1.0, 
                 pixelloss_weight=1.0,
                 perceptual_weight=1.0,
                 use_perceptual=True):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        
        if self.use_perceptual:
            self.perceptual_loss = LPIPS().eval()
        
    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx=0,
                global_step=0, last_layer=None, split="train"):
        """
        Args:
            codebook_loss: VQ commitment loss from quantizer
            inputs: Original images
            reconstructions: Reconstructed images from decoder
            optimizer_idx: Ignored (for compatibility with VQModel)
            global_step: Current training step (ignored)
            last_layer: Last layer of decoder (ignored)
            split: "train" or "val"
        
        Returns:
            loss: Total loss
            log_dict: Dictionary of loss components for logging
        """
        # Pixel-wise reconstruction loss (L1)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # Perceptual loss (optional)
        if self.use_perceptual and self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        
        # Mean over all dimensions
        nll_loss = torch.mean(rec_loss)
        
        # Total loss: reconstruction + codebook
        total_loss = nll_loss + self.codebook_weight * codebook_loss.mean()
        
        # Logging dictionary
        log_dict = {
            f"{split}/total_loss": total_loss.clone().detach(),
            f"{split}/quant_loss": codebook_loss.detach().mean(),
            f"{split}/rec_loss": nll_loss.detach(),
            f"{split}/p_loss": p_loss.detach().mean() if self.use_perceptual else torch.tensor(0.0),
        }
        
        return total_loss, log_dict


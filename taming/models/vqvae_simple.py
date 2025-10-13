"""
Simple VQ-VAE model for finetuning without discriminator.
Inherits from VQModel but uses single optimizer.
"""
import torch
import pytorch_lightning as pl

from taming.models.vqgan import VQModel


class VQVAESimple(VQModel):
    """
    VQ-VAE model without discriminator for simple finetuning.
    
    Key differences from VQModel:
    - Single optimizer (no discriminator optimizer)
    - Simplified training_step (no optimizer_idx handling)
    - Compatible with simple loss functions
    """
    
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        
        # Call loss function (no optimizer_idx needed)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 
                                       optimizer_idx=0, 
                                       global_step=self.global_step,
                                       last_layer=self.get_last_layer(), 
                                       split="train")
        
        # Logging
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, 
                on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, 
                     on_step=True, on_epoch=True)
        
        return aeloss
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 
                                       optimizer_idx=0,
                                       global_step=self.global_step,
                                       last_layer=self.get_last_layer(), 
                                       split="val")
        
        rec_loss = log_dict_ae.get("val/rec_loss", aeloss)
        
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, 
                on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, 
                on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, 
                     on_step=False, on_epoch=True)
        
        return self.log_dict
    
    def configure_optimizers(self):
        """Single optimizer for encoder, decoder, quantizer."""
        lr = self.learning_rate
        
        # Optimizer for autoencoder components only
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, 
            betas=(0.5, 0.9)
        )
        
        return opt_ae


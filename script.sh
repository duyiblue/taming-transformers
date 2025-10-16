#! /bin/bash

python alignment_training.py \
    --source_config /orion/u/duyi/cross-emb/taming-transformers/configs/latent_diffusion_ckpt_config.yaml \
    --target_config /orion/u/duyi/cross-emb/taming-transformers/configs/latent_diffusion_ckpt_config.yaml \
    --apply_mask \
    --corruption_severity 0 \
    --batch_size 12
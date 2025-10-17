#! /bin/bash

python alignment_training.py \
    --source_config $(pwd)/configs/latent_diffusion_ckpt_config.yaml \
    --target_config $(pwd)/configs/latent_diffusion_ckpt_config.yaml \
    --dataset voc \
    --batch_size 12
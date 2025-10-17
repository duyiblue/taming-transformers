This repo is based on [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers). For clarity and conciseness, I removed their original README content, but I suggest you read the original one first before you read my updates.

## My updates
1. Wrote a simple test script (`test_vqgan.py`) that takes in a single image and does inference with a provided checkpoint. Experimented with two checkpoints: `VQGAN ImageNet (f=16)` and `VQGAN OpenImages (f=8), 16384` (from latent diffusion), finding that the later performs better.

2. Because the discriminator (GAN) is used only in training (as a supplementary loss function), it is not required in inference. In the future, we also want to finetune the model without the GAN. Therefore, I implemented a `VQVAESimple` class, which is a simplified version of the VQ-GAN model (`VQModel`) without GAN loss.
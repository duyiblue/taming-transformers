This repo is based on [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers). For clarity and conciseness, I removed their original README content, but I suggest you read the original one first before you read my updates.

## Overview
Goal: Finetune a (source) encoder model, so that its output aligns with that of another (target) encoder model. The source and target models receive paired inputs, where two images are "identical" in essence (depending how you define essence), but differ in domain or appearance. For example:
- **VOC mask experiment**: Source model receives images overlayed with semi-transparent segmentation maps, while target model receives the corresponding original images.
- **Pose encoder experiment**: Source model receives images of robot embodiment A, while target model receives images of robot embodiment B in the same pose. (Here the "essence" of an image is defined as the pose of the robot arm in the image.)

We adopt VQGAN's encoder, because previous robotics work (RoboCat) reports that using VQGAN encoder enhances their policy's cross-embodiment generalization. VQGAN also comes with a decoder, which helps us with validation and visualization (how well our alignment is doing).

## Setup Guide
### Setup conda environment
Different people have different systems, with varying hardware, OS, and CUDA versions. Therefore, we don't provide a single fixed requirement.txt. Instead, we provide the following prompt that you can copy to your Cursor Agent (recommend Claude-4.5-Sonnet or higher), and it should automatically setup the environment for you:
```
Currently, we are in a repo that repurposes VQGAN (from the taming transformer paper) for alignment training. See @README.md and @alignment_training.py for a high-level understanding. 

Please help me create a conda environment and install all necessary packages so that I can run the alignment training experiment. You should read @alignment_training.py , the dataset classes, the model, and all other files carefully to install all dependencies that they used. The most subtle part is the version compatibility between the machine's OS, CUDA, torch, and torchvision's versions. You must check thoroughly and think carefully before you make a move.

After everything is installed, you should be able to run `./script.sh`. 
```

### Download pretrained weights
```
mkdir -p tmp
cd tmp/
wget https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
unzip vq-f8.zip
mv model.ckpt latent_diffusion.ckpt
cd ..
```

## What has been done
1. Wrote a simple test script (`test_vqgan.py`) that takes in a single image and does inference with a provided checkpoint. Experimented with two checkpoints: `VQGAN ImageNet (f=16)` and `VQGAN OpenImages (f=8), 16384` (from latent diffusion), finding that the later performs better.

2. Because the discriminator (GAN) is used only in training (as a supplementary loss function), it is not required in inference. In the future, we also want to finetune the model without the GAN. Therefore, I implemented a `VQVAESimple` class, which is a simplified version of the VQ-GAN model (`VQModel`) without GAN loss.

3. Implemented dataset classes for our alignment training.

4. Implemented `alignment_training.py`, the main alignment training script. 
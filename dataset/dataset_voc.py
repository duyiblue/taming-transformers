import os
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # Hugging Face
import matplotlib.pyplot as plt

from .utils import overlay_image_mask, get_corruption_transforms

class VOCDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        transform_target: Optional[A.Compose] = None,
        transform_source: Optional[A.Compose] = None,
        apply_mask: bool = True,
    ):
        # Load VOC dataset from Hugging Face
        hf_home_path = Path(__file__).resolve().parent.parent / "tmp" / "huggingface"  # Do not use "../tmp/huggingface", as it starts from CWD, not where this file is.
        os.makedirs(hf_home_path, exist_ok=True)  # Create this directory if it doesn't exist.
        os.environ["HF_HOME"] = str(hf_home_path)

        self.ds = load_dataset("jxie/pascal-voc-2012")
        self.ds = self.ds[split]  # "train" or "val"

        self.transform_target = transform_target
        self.transform_source = transform_source
        self.apply_mask = apply_mask

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        pose = 0  # This is just a placeholder for pose

        img1_pil = self.ds[index]["image"].convert("RGB")

        if self.apply_mask:
            mask_pil = self.ds[index]["mask"]
            mask_np = np.array(mask_pil, dtype=np.uint8)  # [H,W], 0..20, 255=ignore
            img2_pil = overlay_image_mask(img1_pil, mask_np, alpha=0.5)
        else:
            img2_pil = img1_pil.copy()
        
        # Convert PIL to numpy array for albumentations
        img1_np = np.array(img1_pil)
        img2_np = np.array(img2_pil)
        
        # Apply albumentations transform
        if self.transform_target is not None:
            transformed = self.transform_target(image=img1_np)
            img1_np = transformed["image"]
        if self.transform_source is not None:
            transformed = self.transform_source(image=img2_np)
            img2_np = transformed["image"]
        
        # Convert to tensor format [C, H, W] and normalize
        img1_tensor = torch.from_numpy(img1_np).float().permute(2, 0, 1) / 255.0
        img2_tensor = torch.from_numpy(img2_np).float().permute(2, 0, 1) / 255.0
        
        return {
            "target_img": img1_tensor,
            "source_img": img2_tensor,
            "pose": pose,
        }

def get_alignment_dataloader(
    target_img_dim: Tuple[int, int], 
    source_img_dim: Tuple[int, int], 
    batch_size: int=12, 
    apply_mask: bool=True, 
    corruption_severity: int=0
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for alignment training.
    Currently we use VOC as the underlying dataset.

    Args:
        target_img_dim (Tuple[int, int]): The input size of the images for the target model.
        source_img_dim (Tuple[int, int]): The input size of the images for the source model.
        batch_size (int, optional): The batch size of the dataloader. Defaults to 12.

    Returns:
        Tuple[DataLoader, DataLoader]: The train and validation loader respectively.
    """
    transform_target = A.Compose([A.Resize(height=target_img_dim[0], width=target_img_dim[1])])
    transform_source = get_corruption_transforms(source_img_dim, severity=corruption_severity)

    train_dataset = VOCDataset(split="train", transform_target=transform_target, transform_source=transform_source, apply_mask=apply_mask)
    val_dataset = VOCDataset(split="val", transform_target=transform_target, transform_source=transform_source, apply_mask=apply_mask)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader


# Below is a test for debugging. Not used in real deployment.
# To run the test, run `python -m dataset.dataset_voc` in the codebase's root.
if __name__ == "__main__":
    train_dataloader, val_dataloader = get_alignment_dataloader(
        target_img_dim=(224, 224),
        source_img_dim=(224, 224),
        batch_size=12,
        apply_mask=True,
        corruption_severity=0,
    )
    
    batch = next(iter(train_dataloader))
    
    # Extract from dictionary format
    target_img = batch["target_img"][0]
    source_img = batch["source_img"][0]
    pose = batch["pose"][0]

    output_dir = Path(__file__).resolve().parent.parent / "tmp"
    output_dir.mkdir(exist_ok=True)
    
    # Convert tensors back to numpy arrays for visualization
    # Tensors are in format [C, H, W] and normalized to [0, 1]
    target_img_np = (target_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    source_img_np = (source_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(target_img_np)
    axes[0].set_title(f'Target Image (Original)')
    axes[0].axis('off')
    
    axes[1].imshow(source_img_np)
    axes[1].set_title(f'Source Image (Transformed)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / "sample_images_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
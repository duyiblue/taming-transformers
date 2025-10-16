import os
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import pickle
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image


class RobotDataset(Dataset):
    def __init__(
        self,
        target_path: str,
        source_path: str,
        transform_target: Optional[A.Compose] = None,
        transform_source: Optional[A.Compose] = None,
        split: str = "train",
        seed: int = 42,
    ):
        self.target_path = target_path
        self.source_path = source_path
        self.transform_target = transform_target
        self.transform_source = transform_source

        target_image_files = os.listdir(target_path)
        target_image_files = [file for file in target_image_files if file.endswith(".png")]

        source_image_files = os.listdir(source_path)
        source_image_files = [file for file in source_image_files if file.endswith(".png")]

        if len(target_image_files) != len(source_image_files):
            raise ValueError("Number of target and source images must be the same")

        target_image_files = sorted(target_image_files)
        source_image_files = sorted(source_image_files)

        # random shuffle while preserving pairing
        np.random.seed(seed)
        indices = np.arange(len(target_image_files))
        np.random.shuffle(indices)
        
        # Apply the same shuffled indices to both lists to preserve pairing
        target_image_files = [target_image_files[i] for i in indices]
        source_image_files = [source_image_files[i] for i in indices]

        train_size = int(len(target_image_files) * 0.8)
        val_size = len(target_image_files) - train_size

        if split == "train":
            self.target_image_files = target_image_files[:train_size]
            self.source_image_files = source_image_files[:train_size]
        elif split == "val":
            self.target_image_files = target_image_files[train_size:]
            self.source_image_files = source_image_files[train_size:]
        else:
            raise ValueError("Split must be either train or val")

    def __len__(self):
        return len(self.target_image_files)

    def __getitem__(self, index):
        target_image_path = os.path.join(self.target_path, self.target_image_files[index])
        source_image_path = os.path.join(self.source_path, self.source_image_files[index])
        
        pose = 0  # This is just a placeholder for pose
        
        # Load images using PIL
        target_img_pil = Image.open(target_image_path).convert("RGB")
        source_img_pil = Image.open(source_image_path).convert("RGB")
        
        # Convert PIL to numpy array
        target_img_np = np.array(target_img_pil)
        source_img_np = np.array(source_img_pil)
        
        # Apply albumentations transform
        if self.transform_target is not None:
            transformed = self.transform_target(image=target_img_np)
            target_img_np = transformed["image"]
        if self.transform_source is not None:
            transformed = self.transform_source(image=source_img_np)
            source_img_np = transformed["image"]
        
        # Convert to tensor format [C, H, W] and normalize
        target_img_tensor = torch.from_numpy(target_img_np).float().permute(2, 0, 1) / 255.0
        source_img_tensor = torch.from_numpy(source_img_np).float().permute(2, 0, 1) / 255.0
        
        return {
            "target_img": target_img_tensor,
            "source_img": source_img_tensor,
            "pose": pose,
            "meta": {
                "target_img_path": target_image_path,
                "source_img_path": source_image_path,
            }
        }

def get_alignment_dataloader(
    target_img_dim: Tuple[int, int],
    source_img_dim: Tuple[int, int],
    batch_size: int=12
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for alignment training using robot data.
    Currently uses hardcoded robot data paths.

    Args:
        target_img_dim (Tuple[int, int]): The input size of the images for the target model.
        source_img_dim (Tuple[int, int]): The input size of the images for the source model.
        batch_size (int, optional): The batch size of the dataloader. Defaults to 12.

    Returns:
        Tuple[DataLoader, DataLoader]: The train and validation loader respectively.
    """
    # Hardcoded robot data paths
    target_path = "/orion/u/duyi/cross-emb/dinov2-finetune/data/home/yufeiding/projects/RoboVerseCrossEmb/retarget_data/franka_to_sawyer/CloseBox/images/franka"
    source_path = "/orion/u/duyi/cross-emb/dinov2-finetune/data/home/yufeiding/projects/RoboVerseCrossEmb/retarget_data/franka_to_sawyer/CloseBox/images/sawyer"
    
    # Create transforms for resizing (no corruption for robot data)
    transform_target = A.Compose([A.Resize(height=target_img_dim[0], width=target_img_dim[1])])
    transform_source = A.Compose([A.Resize(height=source_img_dim[0], width=source_img_dim[1])])
    
    train_dataset = RobotDataset(
        target_path=target_path,
        source_path=source_path,
        transform_target=transform_target,
        transform_source=transform_source,
        split="train",
    )
    val_dataset = RobotDataset(
        target_path=target_path,
        source_path=source_path,
        transform_target=transform_target,
        transform_source=transform_source,
        split="val",
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader

if __name__ == "__main__":
    train_dataloader, val_dataloader = get_alignment_dataloader(
        target_img_dim=(256, 256),
        source_img_dim=(224, 224),
        batch_size=12,
    )
    
    batch = next(iter(train_dataloader))
    
    # Extract from dictionary format
    target_img = batch["target_img"][0]
    source_img = batch["source_img"][0] 
    pose = batch["pose"][0]
    meta = batch["meta"]

    output_dir = Path(__file__).resolve().parent.parent / "tmp"
    output_dir.mkdir(exist_ok=True)
    
    # Convert tensors back to numpy arrays for visualization
    # Tensors are in format [C, H, W] and normalized to [0, 1]
    target_img_np = (target_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    source_img_np = (source_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(target_img_np)
    axes[0].set_title(f'Target Image (Embodiment A)')
    axes[0].axis('off')
    
    axes[1].imshow(source_img_np)
    axes[1].set_title(f'Source Image (Embodiment B)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / "sample_images_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    print(f"Target image path: {meta['target_img_path'][0]}")
    print(f"Source image path: {meta['source_img_path'][0]}")
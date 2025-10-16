import os
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import albumentations as A

def voc_palette():
    """
    Pascal VOC palette for 21 classes (0..20). 255 is 'ignore' and will be transparent in overlay.
    Colors follow the standard VOC colormap.
    """
    palette = [
        (0, 0, 0),        # 0=background
        (128, 0, 0),      # 1=aeroplane
        (0, 128, 0),      # 2=bicycle
        (128, 128, 0),    # 3=bird
        (0, 0, 128),      # 4=boat
        (128, 0, 128),    # 5=bottle
        (0, 128, 128),    # 6=bus
        (128, 128, 128),  # 7=car
        (64, 0, 0),       # 8=cat
        (192, 0, 0),      # 9=chair
        (64, 128, 0),     # 10=cow
        (192, 128, 0),    # 11=diningtable
        (64, 0, 128),     # 12=dog
        (192, 0, 128),    # 13=horse
        (64, 128, 128),   # 14=motorbike
        (192, 128, 128),  # 15=person
        (0, 64, 0),       # 16=pottedplant
        (128, 64, 0),     # 17=sheep
        (0, 192, 0),      # 18=sofa
        (128, 192, 0),    # 19=train
        (0, 64, 128),     # 20=tv/monitor
    ]
    # Flatten to 768 entries (256*3) as required by PIL palettes
    flat = []
    for i in range(256):
        if i < len(palette):
            flat.extend(palette[i])
        else:
            flat.extend((0, 0, 0))
    return flat

def colorize_mask(mask_np):
    """
    Convert a [H,W] integer mask (0..20; 255=ignore) to a P-mode PIL image with VOC palette.
    """
    mask_img = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    mask_img.putpalette(voc_palette())
    return mask_img

def overlay_image_mask(img_rgb, mask_np, alpha=0.5):
    """
    Blend RGB image with colored mask. Ignore-index (255) is transparent.

    Args:
        img_rgb: PIL RGB image (PIL.Image.Image object, mode="RGB", shape=(H,W,3), dtype=uint8, range=[0,255])
        mask_np: [H,W] uint8 with class ids (0...20, 255=ignore)
        alpha: float, opacity of the mask (in [0,1])

    Returns:
        PIL RGB image
    """
    # Colorized mask -> RGBA
    color_mask = colorize_mask(mask_np).convert("RGBA")
    # Make ignore pixels transparent
    ignore = (mask_np == 255)
    if ignore.any():
        cm = np.array(color_mask)  # [H,W,4]
        cm[ignore] = [0, 0, 0, 0]
        color_mask = Image.fromarray(cm, mode="RGBA")

    img_rgba = img_rgb.convert("RGBA")
    # Adjust alpha channel of mask
    cm = np.array(color_mask).astype(np.float32)
    cm[..., 3] = (cm[..., 3] * alpha).clip(0, 255)  # scale existing alpha
    color_mask = Image.fromarray(cm.astype(np.uint8), mode="RGBA")

    blended = Image.alpha_composite(img_rgba, color_mask)
    return blended.convert("RGB")

def get_corruption_transforms(img_dim: Tuple[int, int], severity: int):
    """Augmentation pipeline to recreate the ImageNet-C dataset to evaluate the robustness of
    the DINOv2 backbone. Not all augmentations are available in Albumentations, so only the
    available augmentations are included if reasonable.

        Args:
            img_dim (Tuple[int, int]): The height and width input tuple.
            severity (int): A severity level ranging from 1 to 5.

        Returns:
            A.Compose: An augmentation pipeline
    """
    if severity == 0:
        return A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])
    
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussNoise(
                        std_range=[(0, 0.02), (0.02, 0.05), (0.05, 0.09), (0.09, 0.15), (0.15, 0.25)][
                            severity - 1
                        ],
                        p = 1.0,
                    ),
                    A.ISONoise(
                        intensity=[
                            (0.0, 0.3),
                            (0.5, 0.7),
                            (0.6, 0.9),
                            (0.6, 0.9),
                            (0.9, 1.2),
                        ][severity - 1],
                        p = 1.0,
                    ),
                    A.GaussianBlur(
                        blur_limit=[(0, 1), (1, 3), (1, 3), (3, 5), (5, 7)][
                            severity - 1
                        ],
                        p = 1.0,
                    ),
                    A.GlassBlur(
                        sigma=[0.7, 0.9, 1, 1.1, 1.5][severity - 1],
                        max_delta=[1, 2, 2, 3, 4][severity - 1],
                        iterations=[2, 1, 3, 2, 2][severity - 1],
                        p = 1.0,
                    ),
                    A.Defocus(
                        radius=[3, 4, 6, 8, 10][severity - 1],
                        alias_blur=[0.1, 0.5, 0.5, 0.5, 0.5][severity - 1],
                        p = 1.0,
                    ),
                    A.MotionBlur(
                        blur_limit=[3, 5, 5, 9, 13][severity - 1],
                        p = 1.0,
                    ),
                    A.ZoomBlur(
                        max_factor=[1.11, 1.16, 1.21, 1.26, 1.31][severity - 1],
                        p = 1.0,
                    ),
                    A.RandomSnow(
                        snow_point_range = ([0.1, 0.2, 0.45, 0.45, 0.55][severity - 1], [0.1, 0.2, 0.45, 0.45, 0.55][severity - 1]),
                        p = 1.0,
                    ),
                    A.ImageCompression(
                        quality_range=([25, 18, 15, 10, 7][severity - 1], [25, 18, 15, 10, 7][severity - 1]),
                        p = 1.0,
                    ),
                    A.ElasticTransform(
                        alpha=[488, 288, 244 * 0.05, 244 * 0.07, 244 * 0.12][
                            severity - 1
                        ],
                        sigma=[
                            244 * 0.7,
                            244 * 0.08,
                            244 * 0.01,
                            244 * 0.01,
                            244 * 0.01,
                        ][severity - 1],
                        p = 1.0,
                    ),
                ],
                p=1.0,
            ),
            A.Resize(height=img_dim[0], width=img_dim[1]),
        ]
    )

import os
from pathlib import Path
import numpy as np
from PIL import Image

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
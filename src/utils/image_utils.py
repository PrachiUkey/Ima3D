 # Masking, augmentations
"""
src/utils/image_utils.py

Utilities to load real-world images with masks, preprocess them for training,
and apply optional augmentations.

Dependencies:
    pip install pillow numpy opencv-python torchvision
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T


def load_image(path: str) -> Image.Image:
    """
    Load an image from disk.

    Args:
        path: path to the image file

    Returns:
        PIL.Image in RGB mode
    """
    img = Image.open(path).convert("RGB")
    return img


def load_mask(path: str) -> np.ndarray:
    """
    Load a binary mask from disk.

    Args:
        path: path to the mask image (white = foreground, black = background)

    Returns:
        np.ndarray (H, W), dtype=uint8 in {0, 1}
    """
    mask = Image.open(path).convert("L")  # grayscale
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)  # binarize
    return mask


def apply_mask(image: Image.Image, mask: np.ndarray, background: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Apply binary mask to an image. Background is replaced with a given color.

    Args:
        image: PIL.Image (RGB)
        mask: np.ndarray (H, W), binary 0/1
        background: RGB tuple to fill background

    Returns:
        PIL.Image with background removed
    """
    img_np = np.array(image)
    if mask.shape != img_np.shape[:2]:
        raise ValueError("Mask and image dimensions do not match")

    bg = np.zeros_like(img_np)
    bg[:, :] = background
    out = np.where(mask[:, :, None] == 1, img_np, bg)
    return Image.fromarray(out)


def preprocess_image(image: Image.Image, image_size: int = 224, augment: bool = False) -> torch.Tensor:
    """
    Resize and normalize an image for model input.

    Args:
        image: PIL.Image
        image_size: output size (square)
        augment: whether to apply random augmentations

    Returns:
        torch.FloatTensor of shape (3, H, W), normalized to [-1, 1]
    """
    if augment:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1,1]
        ])
    else:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1,1]
        ])
    return transform(image)


def load_and_preprocess(image_path: str, mask_path: Optional[str] = None, image_size: int = 224,
                        augment: bool = False) -> torch.Tensor:
    """
    Full pipeline: load image, apply mask (if provided), resize + normalize.

    Args:
        image_path: path to RGB image
        mask_path: path to binary mask (optional)
        image_size: size for resizing
        augment: whether to apply augmentations

    Returns:
        torch.FloatTensor (3, H, W)
    """
    img = load_image(image_path)

    if mask_path is not None:
        mask = load_mask(mask_path)
        img = apply_mask(img, mask)

    img_tensor = preprocess_image(img, image_size=image_size, augment=augment)
    return img_tensor

"""
src/data/image_dataset.py

Dataset for loading real-world images with optional binary masks.

Dependencies:
    pip install torch torchvision Pillow numpy
"""

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    """
    Dataset for loading images and optional masks.
    """

    def __init__(self, image_dir: str, mask_dir: str = None, transform=None, split="train", split_ratio=0.8):
        """
        Args:
            image_dir: path to folder with images
            mask_dir: path to folder with masks (optional)
            transform: torchvision transform for preprocessing
            split: 'train' or 'val'
            split_ratio: ratio of data used for training
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.files.sort()

        split_idx = int(len(self.files) * split_ratio)
        if split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        # Default transform (resize + normalize)
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        # Load image
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Apply mask if available
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, filename)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")  # grayscale
                mask = np.array(mask) > 0
                image_np = np.array(image)
                image_np[~mask] = 0
                image = Image.fromarray(image_np)

        # Transform
        image = self.transform(image)

        # Label from parent folder (class name)
        label = os.path.basename(os.path.dirname(img_path))

        return image, label

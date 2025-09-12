# src/utils/image_utils.py

import torch
import torchvision.transforms as T
from PIL import Image

# Default image size for encoder
IMAGE_SIZE = (128, 128)

# Compose standard transforms
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),            # Converts HxWxC to CxHxW and scales [0,255] -> [0,1]
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # Optional normalization
])

def preprocess_image(img_path: str, device='cpu'):
    """
    Load image from disk, resize, and convert to tensor.
    Returns a batch tensor of shape (1, 3, H, W).
    """
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def flatten_tensor(x: torch.Tensor):
    """
    Flatten conv feature maps for linear layer input.
    """
    return x.view(x.size(0), -1)

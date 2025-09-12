# src/models/image_encoder.py

import torch
import torch.nn as nn
from src.utils.image_utils import flatten_tensor

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 128->64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64->32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 32->16
            nn.ReLU(),
        )
        self.latent_dim = latent_dim
        self.fc = None  # will initialize dynamically

    def forward(self, x):
        x = self.conv(x)
        x = flatten_tensor(x)  # from utils
        if self.fc is None:
            # dynamically create linear layer
            self.fc = nn.Linear(x.shape[1], self.latent_dim).to(x.device)
        x = self.fc(x)
        return x

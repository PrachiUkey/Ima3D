# src/models/image_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        super(ImageEncoder, self).__init__()

        # Load ResNet backbone
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Remove classification head (use feature extractor)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        # Projection to embedding space
        self.fc = nn.Linear(in_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input: (B, 3, H, W)
        x = self.feature_extractor(x)   # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 512)
        x = self.fc(x)                  # (B, embed_dim)
        x = self.norm(x)
        x = F.normalize(x, dim=1)       # L2 normalize
        return x

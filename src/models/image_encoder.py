# src/models/image_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        super(ImageEncoder, self).__init__()

        # Use ResNet50 backbone
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        # Projection head (DINOv2 style)
        self.proj = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.feature_extractor(x)       # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)          # (B, 2048)
        x = self.proj(x)
        x = F.normalize(x, dim=1)
        return x

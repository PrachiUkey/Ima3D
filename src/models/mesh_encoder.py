# src/models/mesh_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(MeshEncoder, self).__init__()

        # PointNet-style encoder
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)
        self.mlp4 = nn.Linear(256, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, N, 3) point cloud
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = self.mlp4(x)  # (B, N, embed_dim)

        # Global max pooling across points
        x, _ = torch.max(x, dim=1)  # (B, embed_dim)

        # Normalize embedding
        x = self.norm(x)
        x = F.normalize(x, dim=1)
        return x

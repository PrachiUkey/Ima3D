# cad encpder
"""
src/models/mesh_encoder.py

PointNet-based encoder for extracting embeddings from 3D meshes (via point clouds).

Dependencies:
    pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        """
        Args:
            embed_dim: output embedding dimension
        """
        super(MeshEncoder, self).__init__()

        # PointNet-style MLPs
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)

        # Global pooling
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, embed_dim)

    def forward(self, x):
        """
        Args:
            x: point cloud [B, N, 3]

        Returns:
            embeddings: [B, embed_dim]
        """
        # Apply MLPs per point
        x = F.relu(self.mlp1(x))   # [B, N, 64]
        x = F.relu(self.mlp2(x))   # [B, N, 128]
        x = F.relu(self.mlp3(x))   # [B, N, 256]

        # Global max pooling over points
        x = torch.max(x, dim=1)[0]  # [B, 256]

        # Projection head
        x = F.relu(self.fc1(x))     # [B, 256]
        x = self.fc2(x)             # [B, embed_dim]

        # Normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        return x


if __name__ == "__main__":
    # Quick test
    model = MeshEncoder(embed_dim=128)
    pc = torch.randn(4, 1024, 3)   # batch of 4, each 1024 points
    out = model(pc)
    print("Embedding shape:", out.shape)  # [4, 128]

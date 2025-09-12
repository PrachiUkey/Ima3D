import torch
import torch.nn as nn

class MeshEncoder(nn.Module):
    def __init__(self, num_points=5000, latent_dim=128):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = self.mlp(x)  # [B, N, latent]
        x = torch.max(x, dim=1)[0]  # global max pool
        return x

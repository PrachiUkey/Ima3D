import torch
import torch.nn as nn

class MeshEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_points = cfg["mesh"]["num_points"]
        self.latent_dim = cfg["model"]["latent_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        )

    def forward(self, x):
        # x: (B, N, 3)
        x = self.mlp(x)              # (B, N, latent)
        x = x.max(dim=1)[0]          # global max pooling
        return x

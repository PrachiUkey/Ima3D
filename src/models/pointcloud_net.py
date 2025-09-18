import torch
import torch.nn as nn

class PointCloudNet(nn.Module):
    def __init__(self, num_points=1024, latent_dim=512):
        super().__init__()
        self.num_points = num_points

        # Encoder: simple CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Decoder: fully connected to points
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        latent = self.encoder(x).view(batch_size, -1)
        points = self.decoder(latent)
        points = points.view(batch_size, self.num_points, 3)
        return points

import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg["image"]["img_size"]
        self.channels = cfg["image"]["channels"]
        self.latent_dim = cfg["model"]["latent_dim"]

        # simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        conv_out = 128 * (self.img_size//8) * (self.img_size//8)
        self.fc = nn.Linear(conv_out, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

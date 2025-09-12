import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()
        )
        # calculate final flattened size
        conv_out_size = (img_size // 16) ** 2 * 256
        self.fc = nn.Linear(conv_out_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(MeshEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (B, N, 3) point cloud
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.max(x, dim=1)[0]  # global feature pooling
        x = self.fc4(x)
        return F.normalize(x, dim=-1)

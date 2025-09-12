class MeshEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        """
        x: [B, N, 3] point cloud
        returns: [B, latent_dim]
        """
        # Apply MLP to each point
        x = self.mlp(x)  # [B,N,latent_dim]
        # Aggregate points (max pooling)
        x, _ = torch.max(x, dim=1)  # [B, latent_dim]
        x = self.bn(x)
        x = nn.functional.normalize(x, dim=1)
        return x

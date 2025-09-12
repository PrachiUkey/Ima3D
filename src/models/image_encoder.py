import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256, pretrained=True):
        super().__init__()
        # Use ResNet18 backbone
        backbone = models.resnet18(pretrained=pretrained)
        modules = list(backbone.children())[:-1]  # remove final fc layer
        self.feature_extractor = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        """
        x: [B,3,H,W]
        returns: [B, latent_dim]
        """
        features = self.feature_extractor(x)  # [B,512,1,1]
        features = features.view(features.size(0), -1)
        latent = self.fc(features)
        latent = self.bn(latent)
        latent = nn.functional.normalize(latent, dim=1)  # normalize for cosine similarity
        return latent

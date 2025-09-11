"""
src/models/image_encoder.py

CNN-based encoder for extracting feature embeddings from real-world images.

Dependencies:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        """
        Args:
            embed_dim: dimension of the output embedding
            pretrained: use ImageNet-pretrained backbone
        """
        super(ImageEncoder, self).__init__()

        # Load ResNet18 backbone
        backbone = models.resnet18(pretrained=pretrained)

        # Remove the classification head
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # output: [B, 512, 1, 1]

        # Projection layer: 512 -> embed_dim
        self.projection = nn.Linear(512, embed_dim)

    def forward(self, x):
        """
        Args:
            x: image tensor [B, 3, H, W]

        Returns:
            embeddings: [B, embed_dim] normalized
        """
        features = self.feature_extractor(x)       # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]

        embeddings = self.projection(features)     # [B, embed_dim]

        # Normalize to unit length (important for cosine similarity)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


if __name__ == "__main__":
    # Quick test
    model = ImageEncoder(embed_dim=128, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print("Embedding shape:", out.shape)  # [4, 128]

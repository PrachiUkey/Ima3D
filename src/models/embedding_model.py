import torch
import torch.nn as nn
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder

class EmbeddingModel(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        super(EmbeddingModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.mesh_encoder = MeshEncoder(embed_dim=embed_dim)

    def forward(self, img=None, mask=None, mesh=None):
        """
        Forward pass for image+mask OR mesh.
        If both are given, returns both embeddings.
        """
        img_emb = None
        mesh_emb = None

        if img is not None:
            if mask is not None:
                # fuse mask by concatenation (optionally learn fusion later)
                img = torch.cat([img, mask], dim=1)
                img = img[:, :3, :, :]  # drop extra channel (keeps ResNet compatibility)
            img_emb = self.image_encoder(img)

        if mesh is not None:
            mesh_emb = self.mesh_encoder(mesh)

        return img_emb, mesh_emb

 # Combines both encoders
import torch.nn as nn
from .image_encoder import ImageEncoder
from .mesh_encoder import MeshEncoder

class MultimodalModel(nn.Module):
    def __init__(self, latent_dim=128):
        super(MultimodalModel, self).__init__()
        self.image_encoder = ImageEncoder(latent_dim)
        self.mesh_encoder = MeshEncoder(latent_dim)

    def forward(self, img, mesh):
        img_latent = self.image_encoder(img)
        mesh_latent = self.mesh_encoder(mesh)
        return img_latent, mesh_latent

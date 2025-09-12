# src/training/trainer.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.utils.mesh_utils import load_obj_as_pointcloud
from src.utils.image_utils import preprocess_image

# ---- Example Dataset ----
class ImageMeshDataset(Dataset):
    def __init__(self, img_paths, mesh_paths, num_points=1024, device='cpu'):
        self.img_paths = img_paths
        self.mesh_paths = mesh_paths
        self.num_points = num_points
        self.device = device

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_tensor = preprocess_image(self.img_paths[idx], device=self.device)
        points = load_obj_as_pointcloud(self.mesh_paths[idx], num_points=self.num_points)
        mesh_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(self.device)
        return img_tensor.squeeze(0), mesh_tensor.squeeze(0)  # remove batch dim here

# ---- Training Function ----
def train(num_epochs=5, batch_size=4, save_every=2, device='cpu'):
    # Dummy paths (replace with real)
    img_paths = ["data/images/img1.png", "data/images/img2.png"]
    mesh_paths = ["data/model/bed/bed_0/model.obj", "data/model/bed/bed_1/model.obj"]
    dataset = ImageMeshDataset(img_paths, mesh_paths, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    image_encoder = ImageEncoder(latent_dim=128).to(device)
    mesh_encoder = MeshEncoder(latent_dim=128).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(image_encoder.parameters()) + list(mesh_encoder.parameters()), lr=1e-4)

    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        for images, meshes in loader:
            images = images.to(device)
            meshes = meshes.to(device)

            optimizer.zero_grad()
            img_latent = image_encoder(images)
            mesh_latent = mesh_encoder(meshes)
            loss = criterion(img_latent, mesh_latent)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch}/{num_epochs}] Loss = {avg_loss:.4f}")

        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "image_encoder_state_dict": image_encoder.state_dict(),
                "mesh_encoder_state_dict": mesh_encoder.state_dict()
            }, f"outputs/checkpoints/epoch_{epoch}.pth")
            print(f"✅ Saved checkpoint: outputs/checkpoints/epoch_{epoch}.pth")


if __name__ == "__main__":
    train(num_epochs=5, batch_size=2, device='cpu')

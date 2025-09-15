# src/training/decoder.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.trainer import ImageEncoder
from src.dataset.image_dataset import ImageDataset
from src.dataset.mesh_dataset import MeshDataset

# -----------------------------
# Paired Dataset
# -----------------------------
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset, mesh_dataset):
        self.img_dataset = img_dataset
        self.mesh_dataset = mesh_dataset

        # group meshes by class
        self.mesh_by_class = {}
        for i in range(len(mesh_dataset)):
            points, normals, cls = mesh_dataset[i]
            if cls not in self.mesh_by_class:
                self.mesh_by_class[cls] = []
            self.mesh_by_class[cls].append(points)

        # assign mesh to each image (cycle through available meshes)
        self.mesh_indices = []
        for idx in range(len(img_dataset)):
            cls = img_dataset.samples[idx]["class"]
            mesh_list = self.mesh_by_class[cls]
            mesh_idx = idx % len(mesh_list)
            self.mesh_indices.append(mesh_idx)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img, _, cls = self.img_dataset[idx]
        mesh_list = self.mesh_by_class[cls]
        mesh_idx = self.mesh_indices[idx]
        gt_points = mesh_list[mesh_idx]
        return img, gt_points


# -----------------------------
# Decoder Model
# -----------------------------
class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_points=5000):
        super(PointCloudDecoder, self).__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, self.num_points, 3)


# -----------------------------
# Chamfer Distance Loss
# -----------------------------
def chamfer_distance(pc1, pc2):
    diff = torch.cdist(pc1, pc2)  # (B, N, N)
    dist1 = torch.min(diff, dim=2)[0].mean(1)
    dist2 = torch.min(diff, dim=1)[0].mean(1)
    return (dist1 + dist2).mean()


# -----------------------------
# Training Loop
# -----------------------------
def train_decoder(
    encoder_ckpt="checkpoints/embedding_epoch1.pth",
    num_points=5000,
    epochs=5,
    batch_size=2,
    lr=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load frozen encoder
    encoder = ImageEncoder(latent_dim=256).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device)["image_encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Decoder
    decoder = PointCloudDecoder(latent_dim=256, num_points=num_points).to(device)

    # Optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # Datasets
    img_dataset = ImageDataset(
        img_dir="data/img",
        mask_dir="data/mask",
        classes=["bed", "chair"]
    )

    mesh_dataset = MeshDataset(
        mesh_dir="data/model",
        classes=["bed", "chair"],
        num_points=num_points,
        sampling="surface"
    )

    paired_dataset = PairedDataset(img_dataset, mesh_dataset)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for img, gt_points in dataloader:
            img, gt_points = img.to(device), gt_points.to(device)

            with torch.no_grad():
                z = encoder(img)  # latent code

            pred_points = decoder(z)

            loss = chamfer_distance(pred_points, gt_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(decoder.state_dict(), f"checkpoints/decoder_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train_decoder()

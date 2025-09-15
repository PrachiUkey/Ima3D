# src/training/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.dataset.image_dataset import ImageDataset
from src.dataset.mesh_dataset import MeshDataset

# -------------------------
# Paired dataset for embedding training
# -------------------------
class PairedDataset(Dataset):
    def __init__(self, img_dataset, mesh_dataset, seed=42):
        self.img_dataset = img_dataset
        self.mesh_dataset = mesh_dataset

        # Group meshes by class
        self.mesh_by_class = {}
        for points, normals, cls in mesh_dataset:
            if cls not in self.mesh_by_class:
                self.mesh_by_class[cls] = []
            self.mesh_by_class[cls].append(points)

        # Precompute mesh assignment per image
        torch.manual_seed(seed)
        self.mesh_indices = []
        for idx in range(len(img_dataset)):
            cls = img_dataset.samples[idx]['class']
            mesh_list = self.mesh_by_class[cls]
            mesh_idx = idx % len(mesh_list)
            self.mesh_indices.append(mesh_idx)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img, mask, cls = self.img_dataset[idx]
        mesh_list = self.mesh_by_class[cls]
        mesh_idx = self.mesh_indices[idx]
        points = mesh_list[mesh_idx]
        return img, points, cls

# -------------------------
# Cosine embedding loss
# -------------------------
def embedding_loss(img_embed, mesh_embed):
    sim = F.cosine_similarity(img_embed, mesh_embed)
    loss = 1 - sim.mean()
    return loss

# -------------------------
# Training function
# -------------------------
def train_embedding(
    img_dir="data/img",
    mask_dir="data/mask",
    mesh_dir="data/model",
    latent_dim=256,
    num_points=100000,    # 1 lakh points
    batch_size=2,
    epochs=50,
    lr=1e-4,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Datasets
    # -------------------------
    transform = transforms.Compose([transforms.ToTensor()])

    # Automatically detect all folders as classes
    classes = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    print("Detected classes:", classes)

    img_dataset = ImageDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        classes=classes,
        transform=transform
    )

    mesh_dataset = MeshDataset(
        mesh_dir=mesh_dir,
        num_points=num_points,
        sampling="surface",
        classes=classes
    )

    paired_dataset = PairedDataset(img_dataset, mesh_dataset)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------------
    # Models
    # -------------------------
    image_encoder = ImageEncoder(embed_dim=latent_dim).to(device)
    mesh_encoder = MeshEncoder(embed_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(image_encoder.parameters()) + list(mesh_encoder.parameters()),
        lr=lr
    )

    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for step, (img, points, cls) in enumerate(dataloader):
            img = img.to(device)
            points = points.to(device)

            img_embed = image_encoder(img)
            mesh_embed = mesh_encoder(points)

            loss = embedding_loss(img_embed, mesh_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} Step {step+1}/{len(dataloader)} - Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed in {time.time()-start_time:.2f}s - Avg Loss: {avg_loss:.6f}")

        # Save checkpoint every epoch
        torch.save({
            'image_encoder': image_encoder.state_dict(),
            'mesh_encoder': mesh_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, f"checkpoints/embedding_epoch{epoch+1}.pth")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    train_embedding()


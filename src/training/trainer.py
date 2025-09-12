import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from src.dataset.image_dataset import ImageDataset
from src.dataset.mesh_dataset import MeshDataset

# -------------------------
# Paired Dataset
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
            self.mesh_by_class[cls].append((points, normals))

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
        points, normals = mesh_list[mesh_idx]
        return img, points, cls

# -------------------------
# Image Encoder
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        modules = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)  # LayerNorm instead of BatchNorm

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.norm(x)
        x = F.normalize(x, dim=1)
        return x

# -------------------------
# Mesh Encoder (PointNet-style)
# -------------------------
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
        self.norm = nn.LayerNorm(latent_dim)  # LayerNorm instead of BatchNorm

    def forward(self, x):
        x = self.mlp(x)        # [B,N,latent_dim]
        x, _ = torch.max(x, dim=1)  # max pooling
        x = self.norm(x)
        x = F.normalize(x, dim=1)
        return x

# -------------------------
# Cosine embedding loss
# -------------------------
def embedding_loss(img_latent, mesh_latent):
    sim = F.cosine_similarity(img_latent, mesh_latent)
    loss = 1 - sim.mean()
    return loss

# -------------------------
# Training function
# -------------------------
def train_embedding(img_dataset, mesh_dataset, latent_dim=256, batch_size=1, epochs=5, lr=1e-4):
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    paired_dataset = PairedDataset(img_dataset, mesh_dataset)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    image_encoder = ImageEncoder(latent_dim=latent_dim).to(device)
    mesh_encoder = MeshEncoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(mesh_encoder.parameters()), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        start_epoch = time.time()

        for step, (img, points, cls) in enumerate(dataloader):
            batch_start = time.time()
            img = img.to(device)
            points = points.to(device)

            img_latent = image_encoder(img)
            mesh_latent = mesh_encoder(points)

            loss = embedding_loss(img_latent, mesh_latent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Step {step+1}/{len(dataloader)} - Loss: {loss.item():.4f} - Batch time: {time.time() - batch_start:.2f}s")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] completed in {time.time() - start_epoch:.2f}s - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
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
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_dataset = ImageDataset(
        img_dir="data/img",
        mask_dir="data/mask",
        classes=["bed","chair"],  # Add more classes as needed
        transform=transform
    )

    mesh_dataset = MeshDataset(
        mesh_dir="data/model",
        classes=["bed","chair"],  # Add more classes as needed
        num_points=5000,  # Start small
        sampling='surface'
    )

    train_embedding(
        img_dataset=img_dataset,
        mesh_dataset=mesh_dataset,
        latent_dim=256,
        batch_size=1,  # small batch for testing
        epochs=2,      # quick test run
        lr=1e-4
    )


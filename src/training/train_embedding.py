import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.dataset.image_dataset import ImageDataset
from src.dataset.mesh_dataset import MeshDataset

# -------------------------
# Paired dataset
# -------------------------
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset, mesh_dataset, seed=42):
        self.img_dataset = img_dataset
        self.mesh_dataset = mesh_dataset

        # Group meshes by class
        self.mesh_by_class = {}
        for points, cls in mesh_dataset:
            cls = cls.lower()
            if cls not in self.mesh_by_class:
                self.mesh_by_class[cls] = []
            self.mesh_by_class[cls].append(points)

        # Check that all image classes exist in meshes
        img_classes = set([s['class'].lower() for s in img_dataset.samples])
        missing_classes = img_classes - set(self.mesh_by_class.keys())
        if missing_classes:
            raise ValueError(f"Missing meshes for classes: {missing_classes}")

        # Precompute mesh assignment per image
        torch.manual_seed(seed)
        self.mesh_indices = []
        for idx in range(len(img_dataset)):
            cls = img_dataset.samples[idx]['class'].lower()
            mesh_list = self.mesh_by_class[cls]
            mesh_idx = idx % len(mesh_list)
            self.mesh_indices.append(mesh_idx)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img, mask, cls = self.img_dataset[idx]
        cls = cls.lower()
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
    num_points=100000,
    batch_size=2,
    epochs=50,
    lr=1e-4,
    device=None,
    inference_callback=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets
    img_dataset = ImageDataset(img_dir, mask_dir, img_size=(256,256))
    mesh_dataset = MeshDataset(mesh_dir, num_points=num_points, sampling="surface")
    paired_dataset = PairedDataset(img_dataset, mesh_dataset)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Models
    image_encoder = ImageEncoder(embed_dim=latent_dim).to(device)
    mesh_encoder = MeshEncoder(embed_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(image_encoder.parameters()) + list(mesh_encoder.parameters()),
        lr=lr
    )

    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        image_encoder.train()
        mesh_encoder.train()

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
                print(f"Epoch {epoch+1}/{epochs} Step {step+1}/{len(dataloader)} Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed in {time.time()-start_time:.2f}s - Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save({
            'image_encoder': image_encoder.state_dict(),
            'mesh_encoder': mesh_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, f"checkpoints/embedding_epoch{epoch+1}.pth")

        # Run inference every 5 epochs
        if inference_callback and (epoch + 1) % 5 == 0:
            inference_callback(image_encoder, mesh_encoder, mesh_dataset, device, epoch)

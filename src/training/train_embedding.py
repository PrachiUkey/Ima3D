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
# Paired dataset (subclass-aware)
# -------------------------
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset, mesh_dataset, seed=42):
        self.img_dataset = img_dataset
        self.mesh_dataset = mesh_dataset

        # Group meshes by subclass
        self.mesh_by_subclass = {}
        for points, cls in mesh_dataset:  # cls is like "chair_0"
            cls = cls.lower()
            if cls not in self.mesh_by_subclass:
                self.mesh_by_subclass[cls] = []
            self.mesh_by_subclass[cls].append(points)

        # Check all image classes exist
        img_classes = set([s['class'].lower() for s in img_dataset.samples])
        missing = img_classes - set(self.mesh_by_subclass.keys())
        if missing:
            raise ValueError(f"Missing meshes for subclasses: {missing}")

        # Precompute assignments
        torch.manual_seed(seed)
        self.mesh_indices = []
        for idx in range(len(img_dataset)):
            cls = img_dataset.samples[idx]['class'].lower()  # subclass label
            mesh_list = self.mesh_by_subclass[cls]
            mesh_idx = idx % len(mesh_list)
            self.mesh_indices.append(mesh_idx)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img, mask, cls = self.img_dataset[idx]  # cls must be subclass
        cls = cls.lower()
        mesh_list = self.mesh_by_subclass[cls]
        mesh_idx = self.mesh_indices[idx]
        points = mesh_list[mesh_idx]
        return img, points, cls

# -------------------------
# Triplet loss
# -------------------------
def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = 1 - F.cosine_similarity(anchor, positive)
    neg_dist = 1 - F.cosine_similarity(anchor, negative)
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()

# -------------------------
# Training function
# -------------------------
def train_embedding(
    img_dir="data/img",
    mask_dir="data/mask",
    mesh_dir="data/model",
    latent_dim=512,  # bump up embedding dim
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
    img_dataset = ImageDataset(img_dir, mask_dir, img_size=(256,256), subclass_level=True)
    mesh_dataset = MeshDataset(mesh_dir, num_points=num_points, sampling="surface", subclass_level=True)
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

            # --- pick a negative from a different subclass ---
            neg_cls = torch.choice(list(paired_dataset.mesh_by_subclass.keys()))
            while neg_cls == cls[0].lower():  # avoid same subclass
                neg_cls = torch.choice(list(paired_dataset.mesh_by_subclass.keys()))
            neg_points = paired_dataset.mesh_by_subclass[neg_cls][0].to(device)
            neg_embed = mesh_encoder(neg_points.unsqueeze(0))

            # Triplet loss
            loss = triplet_loss(img_embed, mesh_embed, neg_embed)

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

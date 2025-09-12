import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset.image_mesh_dataset import ImageMeshDataset
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
import yaml

# load config
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
BATCH_SIZE = cfg["training"]["batch_size"]
NUM_EPOCHS = cfg["training"]["num_epochs"]
LATENT_DIM = cfg["model"]["latent_dim"]
NUM_POINTS = cfg["mesh"]["num_points"]
IMG_SIZE = cfg["image"]["img_size"]
CHECKPOINT_DIR = cfg["paths"]["checkpoints_dir"]
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

dataset = ImageMeshDataset(cfg)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

image_encoder = ImageEncoder(cfg).to(DEVICE)
mesh_encoder = MeshEncoder(cfg).to(DEVICE)
optimizer = torch.optim.Adam(list(image_encoder.parameters())+list(mesh_encoder.parameters()),
                             lr=cfg["training"]["learning_rate"])
criterion = nn.CosineEmbeddingLoss()

def train():
    for epoch in range(1, NUM_EPOCHS+1):
        total_loss = 0
        for imgs, points, _ in dataloader:
            imgs, points = imgs.to(DEVICE), points.to(DEVICE)
            img_latent = image_encoder(imgs)
            mesh_latent = mesh_encoder(points)
            target = torch.ones(img_latent.size(0), device=DEVICE)
            loss = criterion(img_latent, mesh_latent, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Loss = {avg_loss:.4f}")
        if epoch % cfg["training"]["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "image_encoder_state_dict": image_encoder.state_dict(),
                "mesh_encoder_state_dict": mesh_encoder.state_dict()
            }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    train()

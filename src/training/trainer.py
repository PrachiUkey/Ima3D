# src/training/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.image_mesh_dataset import ImageMeshDataset
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder

# ------------------ CONFIG ------------------
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")  # optional
MODELS_DIR = os.path.join(DATA_DIR, "models")

BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_POINTS = 5000
LATENT_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# --------------------------------------------

# Dataset + DataLoader
dataset = ImageMeshDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    models_dir=MODELS_DIR,
    num_points=NUM_POINTS
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
image_encoder = ImageEncoder(latent_dim=LATENT_DIM).to(DEVICE)
mesh_encoder = MeshEncoder(latent_dim=LATENT_DIM).to(DEVICE)

# Optimizer + Loss
optimizer = optim.Adam(list(image_encoder.parameters()) + list(mesh_encoder.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = 0.0
    for images, meshes, _class in dataloader:
        images = images.to(DEVICE)
        meshes = meshes.to(DEVICE)

        optimizer.zero_grad()
        img_latent = image_encoder(images)
        mesh_latent = mesh_encoder(meshes)
        loss = criterion(img_latent, mesh_latent)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch}/{NUM_EPOCHS}] Loss = {avg_loss:.4f}")

    # Save checkpoint every 2 epochs
    if epoch % 2 == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "image_encoder_state_dict": image_encoder.state_dict(),
            "mesh_encoder_state_dict": mesh_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, ckpt_path)
        print(f"✅ Saved checkpoint: {ckpt_path}")

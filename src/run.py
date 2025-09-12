import torch
import os
import yaml
from src.dataset.image_mesh_dataset import ImageMeshDataset
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ==============================
# Load config
# ==============================
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Paths
DATA_DIR = cfg["paths"]["data_dir"]
CHECKPOINT_DIR = cfg["paths"]["checkpoints_dir"]

# Image settings
IMG_SIZE = cfg["image"]["img_size"]

# Mesh settings
NUM_POINTS = cfg["mesh"]["num_points"]

# Model settings
LATENT_DIM = cfg["model"]["latent_dim"]

# Training / device
DEVICE = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

# ==============================
# Dataset and DataLoader
# ==============================
dataset = ImageMeshDataset(
    data_dir=DATA_DIR,      # pass root folder
    img_size=IMG_SIZE,
    num_points=NUM_POINTS
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ==============================
# Initialize models
# ==============================
image_encoder = ImageEncoder(latent_dim=LATENT_DIM).to(DEVICE)
mesh_encoder = MeshEncoder(latent_dim=LATENT_DIM).to(DEVICE)

# ==============================
# Load latest checkpoint if available
# ==============================
checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    image_encoder.load_state_dict(checkpoint.get("image_encoder_state_dict", {}), strict=False)
    mesh_encoder.load_state_dict(checkpoint.get("mesh_encoder_state_dict", {}), strict=False)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
else:
    print("[INFO] No checkpoint found, running with random weights")

image_encoder.eval()
mesh_encoder.eval()

# ==============================
# Run inference on first sample
# ==============================
sample = dataset[0]
img_tensor = sample["image"].unsqueeze(0).to(DEVICE)   # add batch dim
mesh_tensor = sample["points"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    image_latent = image_encoder(img_tensor)
    mesh_latent = mesh_encoder(mesh_tensor)

    # Cosine similarity
    similarity = F.cosine_similarity(image_latent, mesh_latent)
    print(f"Image latent: {image_latent.shape}")
    print(f"Mesh latent: {mesh_latent.shape}")
    print(f"Cosine similarity: {similarity.item():.4f}")

print("[INFO] Run complete.")

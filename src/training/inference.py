# src/training/inference.py

import os
import torch
import torch.nn.functional as F
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.utils.mesh_utils import load_obj_as_pointcloud
from PIL import Image
import torchvision.transforms as T

# ========================
# CONFIG
# ========================
CHECKPOINT_PATH = "outputs/checkpoints/epoch_4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_POINTS = 2048
LATENT_DIM = 128

# ========================
# UTILS
# ========================
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1,C,H,W)

def load_mesh(path):
    pc = load_obj_as_pointcloud(path, num_points=NUM_POINTS)
    return torch.tensor(pc, dtype=torch.float32).unsqueeze(0)  # (1,N,3)

# ========================
# MODEL
# ========================
image_encoder = ImageEncoder(latent_dim=LATENT_DIM).to(DEVICE)
mesh_encoder = MeshEncoder(latent_dim=LATENT_DIM).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
mesh_encoder.load_state_dict(checkpoint["mesh_encoder_state_dict"])

image_encoder.eval()
mesh_encoder.eval()

# ========================
# SAMPLE INFERENCE
# ========================
img_path = "data/images/bed/0001.png"
mesh_path = "data/models/bed/bed_0/model.obj"

img_tensor = load_image(img_path).to(DEVICE)
mesh_tensor = load_mesh(mesh_path).to(DEVICE)

with torch.no_grad():
    img_latent = image_encoder(img_tensor)
    mesh_latent = mesh_encoder(mesh_tensor)

cos_sim = F.cosine_similarity(img_latent, mesh_latent).item()
print(f"[INFO] Image latent: {img_latent.shape}")
print(f"[INFO] Mesh latent: {mesh_latent.shape}")
print(f"[INFO] Cosine similarity: {cos_sim:.4f}")

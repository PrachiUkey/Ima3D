import torch
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.dataset.image_mesh_dataset import ImageMeshDataset
import yaml
import os
import torch.nn.functional as F

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = cfg["paths"]["checkpoints_dir"]

dataset = ImageMeshDataset(cfg)
image_encoder = ImageEncoder(cfg).to(DEVICE)
mesh_encoder = MeshEncoder(cfg).to(DEVICE)

# load last checkpoint
ckpt_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, ckpt_files[-1]), map_location=DEVICE)
image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
mesh_encoder.load_state_dict(checkpoint["mesh_encoder_state_dict"])
image_encoder.eval(); mesh_encoder.eval()

# inference on first sample
img, points, cls = dataset[0]
img, points = img.unsqueeze(0).to(DEVICE), points.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    img_latent = image_encoder(img)
    mesh_latent = mesh_encoder(points)
cos_sim = F.cosine_similarity(img_latent, mesh_latent)
print(f"[INFO] Image latent: {img_latent.shape}")
print(f"[INFO] Mesh latent: {mesh_latent.shape}")
print(f"[INFO] Cosine similarity: {cos_sim.item():.4f}")

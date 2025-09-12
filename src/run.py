# src/run.py

"""
Run training + inference pipeline for Image → 3D Mesh latent comparison.

- Loads latest checkpoint if available.
- Runs a single inference sample to check latent similarity.
"""

import os
import torch
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.utils.mesh_utils import load_obj_as_pointcloud

# -----------------------------
# Configs
# -----------------------------
NUM_POINTS = 5000
LATENT_DIM = 128
CHECKPOINT_PATH = "outputs/checkpoints/epoch_4.pth"  # Change if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Example data for inference
IMAGE_PATH = "data/img/bed/0001.png"  # Provide your image path
MESH_PATH = "data/model/bed/bed_0/model.obj"  # Provide your mesh path

# -----------------------------
# Helpers
# -----------------------------
def preprocess_image(img_path):
    """Load image and convert to tensor"""
    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return img_tensor.to(DEVICE)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Initialize models
    image_encoder = ImageEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    mesh_encoder = MeshEncoder(latent_dim=LATENT_DIM).to(DEVICE)

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        print(f"[INFO] Loaded checkpoint from {CHECKPOINT_PATH}")

        # Load state dicts safely (ignore missing/unexpected keys)
        image_encoder.load_state_dict(checkpoint.get("image_encoder_state_dict", {}), strict=False)
        mesh_encoder.load_state_dict(checkpoint.get("mesh_encoder_state_dict", {}), strict=False)
    else:
        print("[WARN] No checkpoint found. Models are randomly initialized.")

    image_encoder.eval()
    mesh_encoder.eval()

    # -----------------------------
    # Prepare data
    # -----------------------------
    try:
        img_tensor = preprocess_image(IMAGE_PATH)
    except Exception as e:
        print(f"[WARN] Failed to load image: {e}")
        img_tensor = torch.rand(1, 3, 128, 128).to(DEVICE)  # fallback dummy image

    try:
        mesh_points = load_obj_as_pointcloud(MESH_PATH, num_points=NUM_POINTS, normalize=True)
        mesh_tensor = torch.from_numpy(mesh_points).unsqueeze(0).float().to(DEVICE)  # (1, N, 3)
    except Exception as e:
        print(f"[WARN] Failed to load mesh: {e}")
        mesh_tensor = torch.rand(1, NUM_POINTS, 3).to(DEVICE)  # fallback dummy mesh

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():
        img_latent = image_encoder(img_tensor)
        mesh_latent = mesh_encoder(mesh_tensor)

        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(img_latent, mesh_latent)
        print(f"Image latent: {img_latent.shape}")
        print(f"Mesh latent: {mesh_latent.shape}")
        print(f"Cosine similarity: {cosine_sim.item():.4f}")

    print("[INFO] Run complete.")

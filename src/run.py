import torch
import torch.nn.functional as F
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.dataset.image_mesh_dataset import get_sample_image_mesh

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
LATENT_DIM = 128
IMG_SIZE = 128
NUM_POINTS = 5000

# Initialize models
image_encoder = ImageEncoder(latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(DEVICE)
mesh_encoder = MeshEncoder(num_points=NUM_POINTS, latent_dim=LATENT_DIM).to(DEVICE)

# Load checkpoint if exists
checkpoint_path = "outputs/checkpoints/epoch_4.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    image_state = checkpoint.get("image_encoder_state_dict", None)
    mesh_state = checkpoint.get("mesh_encoder_state_dict", None)
    if image_state:
        image_encoder.load_state_dict(image_state, strict=False)
    if mesh_state:
        mesh_encoder.load_state_dict(mesh_state, strict=False)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
except FileNotFoundError:
    print("[INFO] No checkpoint found, running inference with random weights.")

# Get one sample
img_tensor, mesh_tensor = get_sample_image_mesh(device=DEVICE, image_size=(IMG_SIZE, IMG_SIZE), num_points=NUM_POINTS)

# Forward pass
with torch.no_grad():
    img_latent = image_encoder(img_tensor)
    mesh_latent = mesh_encoder(mesh_tensor)
    cos_sim = F.cosine_similarity(img_latent, mesh_latent).item()

print(f"Image latent: {img_latent.shape}")
print(f"Mesh latent: {mesh_latent.shape}")
print(f"Cosine similarity: {cos_sim:.4f}")
print("[INFO] Run complete.")

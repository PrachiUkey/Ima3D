# src/training/inference.py

import torch
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.utils.image_utils import preprocess_image
from src.utils.mesh_utils import load_obj_as_pointcloud

device = 'cpu'

# Load checkpoint
checkpoint_path = "outputs/checkpoints/epoch_4.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Initialize models
image_encoder = ImageEncoder(latent_dim=128).to(device)
mesh_encoder = MeshEncoder(latent_dim=128).to(device)

image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
mesh_encoder.load_state_dict(checkpoint["mesh_encoder_state_dict"])
image_encoder.eval()
mesh_encoder.eval()

# Example single image & mesh
img_path = "data/images/img1.png"
mesh_path = "data/model/bed/bed_0/model.obj"

img_tensor = preprocess_image(img_path, device=device)
mesh_points = load_obj_as_pointcloud(mesh_path, num_points=1024)
mesh_tensor = torch.tensor(mesh_points, dtype=torch.float32).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    image_latent = image_encoder(img_tensor)
    mesh_latent = mesh_encoder(mesh_tensor)

    cos_sim = torch.nn.functional.cosine_similarity(image_latent, mesh_latent)
    print("Image latent:", image_latent.shape)
    print("Mesh latent:", mesh_latent.shape)
    print("Cosine similarity:", cos_sim.item())

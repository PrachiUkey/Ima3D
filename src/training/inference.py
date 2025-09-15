# src/training/inference.py
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import open3d as o3d

from src.training.trainer import ImageEncoder
from src.training.decoder import PointCloudDecoder


# ------------------------
# Load model weights safely
# ------------------------
def load_encoder_decoder(encoder_ckpt, decoder_ckpt, device):
    # Encoder
    encoder = ImageEncoder(latent_dim=256).to(device)
    ckpt = torch.load(encoder_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["image_encoder"])
    encoder.eval()

    # Decoder
    decoder = PointCloudDecoder(latent_dim=256, num_points=5000).to(device)
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
    decoder.eval()

    return encoder, decoder


# ------------------------
# Generate point cloud
# ------------------------
def generate_pointcloud(img_path, encoder, decoder, device, save_path="output.ply"):
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # Encode → Decode
    with torch.no_grad():
        z = encoder(img)
        pred_points = decoder(z).squeeze(0).cpu().numpy()

    # Save as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_points)
    o3d.io.write_point_cloud(save_path, pcd)

    print(f"Point cloud saved to {save_path}")
    return pred_points


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder_ckpt = "checkpoints/embedding_epoch2.pth"   # latest encoder
    decoder_ckpt = "checkpoints/decoder_epoch5.pth"     # latest decoder
    test_image = "data/img/bed/0011.png"            # example image path

    encoder, decoder = load_encoder_decoder(encoder_ckpt, decoder_ckpt, device)
    generate_pointcloud(test_image, encoder, decoder, device, save_path="chair_pointcloud.ply")

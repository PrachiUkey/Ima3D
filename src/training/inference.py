import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import open3d as o3d
import argparse
import os

# ---------------------------
# Image Encoder
# ---------------------------
class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        modules = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, dim=1)
        return x

# ---------------------------
# Point Cloud Decoder
# ---------------------------
class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_points=20000):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, self.num_points, 3)
        return x

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# ---------------------------
# Inference function
# ---------------------------
def image_to_pointcloud(image_path, encoder, decoder, device='cuda'):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = encoder(img)
        points = decoder(latent)
    return points.squeeze(0).cpu().numpy()

# ---------------------------
# Visualize
# ---------------------------
def visualize_pointcloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to object-focused image")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/embedding_epoch1.pth")
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/decoder.pth")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--num_points", type=int, default=5000)  # reduce for testing
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load models
    latent_dim = 256
    encoder = ImageEncoder(latent_dim=latent_dim).to(device)
    decoder = PointCloudDecoder(latent_dim=latent_dim, num_points=args.num_points).to(device)

    # Load checkpoints
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    encoder.load_state_dict(ckpt['image_encoder'])
    encoder.eval()

    decoder_ckpt = torch.load(args.decoder_ckpt, map_location=device)
    decoder.load_state_dict(decoder_ckpt)
    decoder.eval()

    # Run inference
    points = image_to_pointcloud(args.image, encoder, decoder, device)
    print("Generated point cloud shape:", points.shape)

    # Visualize
    visualize_pointcloud(points)

    # Optionally save PLY
    save_path = os.path.splitext(args.image)[0] + "_pred.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved predicted point cloud: {save_path}")

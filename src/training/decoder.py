import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Encoder from your previous training ---
from src.training.trainer import ImageEncoder
# --- Your dataset class ---
from src.dataset.mesh_dataset import MeshImageDataset  

# -----------------------------
# Decoder Model
# -----------------------------
class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_points=5000):
        super(PointCloudDecoder, self).__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, self.num_points, 3)


# -----------------------------
# Chamfer Distance Loss
# -----------------------------
def chamfer_distance(pc1, pc2):
    diff = torch.cdist(pc1, pc2)  # (B, N, N)
    dist1 = torch.min(diff, dim=2)[0].mean(1)
    dist2 = torch.min(diff, dim=1)[0].mean(1)
    return (dist1 + dist2).mean()


# -----------------------------
# Training Loop
# -----------------------------
def train_decoder(
    encoder_ckpt="checkpoints/embedding_epoch2.pth",
    num_points=5000,
    epochs=5,
    batch_size=2,
    lr=1e-4,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load frozen encoder
    encoder = ImageEncoder(latent_dim=256).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Decoder
    decoder = PointCloudDecoder(latent_dim=256, num_points=num_points).to(device)

    # Optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # Dataset
    dataset = MeshImageDataset(root="data/chairs", num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for img, gt_points in dataloader:
            img, gt_points = img.to(device), gt_points.to(device)

            with torch.no_grad():
                z = encoder(img)  # latent code

            pred_points = decoder(z)

            loss = chamfer_distance(pred_points, gt_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

        # Save checkpoint
        torch.save(decoder.state_dict(), f"checkpoints/decoder_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train_decoder()

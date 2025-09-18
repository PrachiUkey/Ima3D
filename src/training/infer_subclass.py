# verify_embeddings.py
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder
from src.dataset.Paired_Dataset import PairedDataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load dataset
# -----------------------------
img_dir = "data/img"
mask_dir = "data/mask"
mesh_dir = "data/model"

dataset = PairedDataset(img_dir, mask_dir, mesh_dir)  # Make sure mesh_dataset is available
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------------
# Load models
# -----------------------------
image_encoder = ImageEncoder(embed_dim=256).to(device)
mesh_encoder = MeshEncoder(embed_dim=256).to(device)

# -----------------------------
# Load checkpoint
# -----------------------------
checkpoint_path = "checkpoints/embedding_epoch11.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

image_encoder.load_state_dict(checkpoint['image_encoder'])
mesh_encoder.load_state_dict(checkpoint['mesh_encoder'])

image_encoder.eval()
mesh_encoder.eval()

# -----------------------------
# Extract embeddings
# -----------------------------
image_embeddings = []
mesh_embeddings = []
labels = []

with torch.no_grad():
    for img, mask, mesh, cls in dataloader:
        img = img.to(device)
        mesh = mesh.to(device)
        
        img_emb = image_encoder(img)
        mesh_emb = mesh_encoder(mesh)
        
        image_embeddings.append(img_emb.cpu().numpy())
        mesh_embeddings.append(mesh_emb.cpu().numpy())
        labels.append(cls[0])  # Assuming batch_size=1

image_embeddings = np.vstack(image_embeddings)
mesh_embeddings = np.vstack(mesh_embeddings)

# -----------------------------
# t-SNE visualization
# -----------------------------
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(np.vstack([image_embeddings, mesh_embeddings]))
num_samples = len(image_embeddings)

plt.figure(figsize=(10, 8))
# Image embeddings
plt.scatter(emb_2d[:num_samples, 0], emb_2d[:num_samples, 1], c='r', label='Image')
# Mesh embeddings
plt.scatter(emb_2d[num_samples:, 0], emb_2d[num_samples:, 1], c='b', label='Mesh')

# Optional: annotate points with labels
for i, label in enumerate(labels):
    plt.text(emb_2d[i, 0], emb_2d[i, 1], label, fontsize=8)

plt.legend()
plt.title("t-SNE of Image & Mesh Embeddings")
plt.show()

# -----------------------------
# Check nearest neighbor retrieval
# -----------------------------
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(image_embeddings, mesh_embeddings)
nearest_idx = similarity.argmax(axis=1)

for i, idx in enumerate(nearest_idx):
    print(f"Image {labels[i]} -> Nearest Mesh: {labels[idx]} (similarity={similarity[i, idx]:.3f})")

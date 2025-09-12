import torch
from src.models.multimodal_model import MultimodalModel

def main():
    model = MultimodalModel(latent_dim=128)

    # Fake data for demo
    img = torch.randn(1, 3, 64, 64)       # (B, C, H, W)
    mesh = torch.randn(1, 1024, 3)        # (B, N, 3)

    img_latent, mesh_latent = model(img, mesh)

    print("Image latent:", img_latent.shape)
    print("Mesh latent:", mesh_latent.shape)

    # Example similarity
    sim = torch.cosine_similarity(img_latent, mesh_latent)
    print("Cosine similarity:", sim.item())

if __name__ == "__main__":
    main()

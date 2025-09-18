import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.Paired_Dataset import PairedDataset
from src.models.embedding_model import EmbeddingModel

def train_embedding(
    img_dir,
    mask_dir,
    mesh_dir,
    embed_dim=256,
    batch_size=4,
    epochs=20,
    lr=1e-4,
    device=None,
    inference_callback=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PairedDataset(img_dir, mask_dir, mesh_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = EmbeddingModel(embed_dim=embed_dim).to(device)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (img, mask, mesh, cls) in enumerate(dataloader):
            img, mask, mesh = img.to(device), mask.to(device), mesh.to(device)

            img_emb, mesh_emb = model(img=img, mask=mask, mesh=mesh)

            # cosine loss expects labels: 1 for similar, -1 for dissimilar
            target = torch.ones(img_emb.size(0), device=device)
            loss = criterion(img_emb, mesh_emb, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        # run inference callback
        if inference_callback is not None:
            inference_callback(model.image_encoder, model.mesh_encoder, dataset.mesh_dataset, device, epoch)

    return model

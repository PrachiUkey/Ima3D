import torch
from torch.utils.data import DataLoader
from dataset.dataset import ShapeDataset
from models.pointcloud_net import PointCloudNet
from utils.chamfer import chamfer_distance
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ShapeDataset("data/renders/", num_points=1024)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

model = PointCloudNet(num_points=1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, pcs in dataloader:
        imgs = imgs.to(device)
        pcs = pcs.to(device)

        optimizer.zero_grad()
        pred_pc = model(imgs)
        loss = chamfer_distance(pred_pc, pcs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset):.6f}")

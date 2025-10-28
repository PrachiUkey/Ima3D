import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
import open3d as o3d
from PIL import Image

from dataset import Image2PointCloudDataset
from model import Image2PointCloud, CombinedLoss

# ============================================================
# GPU CHECK
# ============================================================
if not torch.cuda.is_available():
    print("ERROR: GPU not available! This requires CUDA.")
    sys.exit(1)

device = torch.device("cuda")
print("="*60)
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA: {torch.version.cuda}")
print("="*60)

# ============================================================
# CONFIGURATION - EASY TO SWITCH!
# ============================================================
class Config:
    # Data
    train_dir = "dataset/train"
    
    # ========== CHOOSE YOUR CONFIGURATION ==========
    # Option 1: FPS with 2048 points (FAST, good quality)
    # num_points = 2048
    # hidden_dim = 256
    # num_layers = 4
    # num_heads = 4
    # batch_size = 32
    
    # Option 2: FPS with 4096 points (RECOMMENDED - best balance)
    num_points = 4096
    hidden_dim = 320
    num_layers = 5
    num_heads = 5
    batch_size = 8
    
    # Option 3: FPS with 8192 points (HIGH DETAIL, slower)
    # num_points = 8192
    # hidden_dim = 384
    # num_layers = 6
    # num_heads = 6
    # batch_size = 12
    
    # Option 4: FPS with 10240 points (MAXIMUM DETAIL, very slow)
    # num_points = 10240
    # hidden_dim = 512
    # num_layers = 6
    # num_heads = 8
    # batch_size = 8
    # ===============================================
    
    # Sampling method
    sampling_method = 'fps'  # 'fps' (best), 'stratified', or 'random'
    
    # Training
    num_epochs = 100
    learning_rate = 2e-4
    weight_decay = 1e-4
    num_workers = 4
    
    # Loss weights
    chamfer_weight = 1.0
    spread_weight = 0.01
    
    # Checkpoint & Visualization
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    vis_dir = "training_vis"
    save_every = 5
    vis_samples = 5
    
    device = device


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_chamfer = 0
    total_spread = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        gt_points = batch["points"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred_points = model(images)
        
        loss, chamfer, spread = criterion(pred_points, gt_points)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_chamfer += chamfer.item()
        total_spread += spread.item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "CD": f"{chamfer.item():.4f}"
        })
    
    return (total_loss / len(train_loader), 
            total_chamfer / len(train_loader), 
            total_spread / len(train_loader))


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_chamfer = 0
    total_spread = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device, non_blocking=True)
            gt_points = batch["points"].to(device, non_blocking=True)
            
            pred_points = model(images)
            loss, chamfer, spread = criterion(pred_points, gt_points)
            
            total_loss += loss.item()
            total_chamfer += chamfer.item()
            total_spread += spread.item()
    
    return (total_loss / len(val_loader), 
            total_chamfer / len(val_loader), 
            total_spread / len(val_loader))


def visualize_predictions(model, dataset, device, epoch, vis_dir, num_samples=5, scale=10.0):
    model.eval()
    
    epoch_dir = os.path.join(vis_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(os.path.join(epoch_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(epoch_dir, "predicted_plys"), exist_ok=True)
    os.makedirs(os.path.join(epoch_dir, "gt_plys"), exist_ok=True)
    
    print(f"\n  📸 Visualizing {num_samples} samples...")
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            
            image_tensor = sample["image"].unsqueeze(0).to(device)
            pred_points = model(image_tensor)
            pred_points = pred_points.squeeze(0).cpu().numpy()
            gt_points = sample["points"].numpy()
            
            pred_points_scaled = pred_points * scale
            gt_points_scaled = gt_points * scale
            
            # Check quality
            unique_pred = len(np.unique(np.round(pred_points, decimals=4), axis=0))
            pred_std = pred_points.std()
            
            print(f"    {idx}: {sample['object']} - Unique: {unique_pred}/{len(pred_points)}, Std: {pred_std:.4f}")
            
            # Save image
            img_tensor = sample["image"]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_pil = Image.fromarray((img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img_pil.save(os.path.join(epoch_dir, "images", f"sample_{idx:02d}_{sample['object']}.png"))
            
            # Save predicted PLY
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_points_scaled)
            colors = np.zeros_like(pred_points_scaled)
            y_min, y_max = pred_points_scaled[:, 1].min(), pred_points_scaled[:, 1].max()
            if y_max - y_min > 1e-6:
                y_norm = (pred_points_scaled[:, 1] - y_min) / (y_max - y_min)
                colors[:, 1] = y_norm
                colors[:, 0] = 1.0 - y_norm
            else:
                colors[:, :] = [0.5, 0.5, 0.5]
            pred_pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(
                os.path.join(epoch_dir, "predicted_plys", f"sample_{idx:02d}_pred.ply"), 
                pred_pcd
            )
            
            # Save GT PLY
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_points_scaled)
            colors_gt = np.zeros_like(gt_points_scaled)
            y_min, y_max = gt_points_scaled[:, 1].min(), gt_points_scaled[:, 1].max()
            if y_max - y_min > 1e-6:
                y_norm = (gt_points_scaled[:, 1] - y_min) / (y_max - y_min)
                colors_gt[:, 1] = y_norm
                colors_gt[:, 0] = 1.0 - y_norm
            else:
                colors_gt[:, :] = [0.5, 0.5, 0.5]
            gt_pcd.colors = o3d.utility.Vector3dVector(colors_gt)
            o3d.io.write_point_cloud(
                os.path.join(epoch_dir, "gt_plys", f"sample_{idx:02d}_gt.ply"), 
                gt_pcd
            )
    
    print(f"  ✓ Saved to {epoch_dir}/")
    model.train()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": {
            "num_points": config.num_points,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads
        }
    }
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"  💾 {filename}")


def main():
    config = Config()
    
    print("\n" + "="*60)
    print("Image-to-PointCloud Training")
    print("SMART SAMPLING + COORDINATE QUERIES")
    print("="*60)
    print(f"Points: {config.num_points}")
    print(f"Sampling: {config.sampling_method}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Batch size: {config.batch_size}")
    print("="*60)
    
    os.makedirs(config.vis_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_dataset = Image2PointCloudDataset(
        config.train_dir, 
        num_points=config.num_points, 
        augment=True,
        sampling_method=config.sampling_method
    )
    val_dataset = Image2PointCloudDataset(
        config.train_dir, 
        num_points=config.num_points, 
        augment=False,
        sampling_method=config.sampling_method
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, 
        persistent_workers=True if config.num_workers > 0 else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Model
    print("\nInitializing model...")
    model = Image2PointCloud(
        num_points=config.num_points,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = CombinedLoss(config.chamfer_weight, config.spread_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(config.log_dir, f"run_{timestamp}"))
    
    best_val_loss = float('inf')
    
    print("\n🚀 Starting training...\n")
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_chamfer, train_spread = train_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        val_loss, val_chamfer, val_spread = validate(
            model, val_loader, criterion, config.device
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Chamfer/train", train_chamfer, epoch)
        writer.add_scalar("Chamfer/val", val_chamfer, epoch)
        writer.add_scalar("Learning_rate", current_lr, epoch)
        
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train - Loss: {train_loss:.6f}, CD: {train_chamfer:.6f}")
        print(f"  Val   - Loss: {val_loss:.6f}, CD: {val_chamfer:.6f}")
        print(f"  LR: {current_lr:.2e}")
        
        if epoch % config.save_every == 0:
            visualize_predictions(model, val_dataset, config.device, epoch, 
                                config.vis_dir, config.vis_samples, scale=10.0)
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          config, f"checkpoint_epoch_{epoch}.pth")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          config, "best_model.pth")
            print(f"  ✓ New best! ({val_loss:.6f})")
    
    visualize_predictions(model, val_dataset, config.device, config.num_epochs, 
                        config.vis_dir, 10, scale=10.0)
    save_checkpoint(model, optimizer, config.num_epochs, train_loss, val_loss, 
                  config, "final_model.pth")
    
    writer.close()
    print("\n" + "="*60)
    print(f"✓ Done! Best: {best_val_loss:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
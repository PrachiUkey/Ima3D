import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import open3d as o3d

from model import Image2PointCloud

# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_40.pth"
IMAGE_PATH = "dataset/train/images/00016.png"  # Change this to your image
OUTPUT_PATH = "debug_output.ply"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("DEBUG INFERENCE SCRIPT")
print("="*60)

# ============================================================
# STEP 1: Load Checkpoint
# ============================================================
print(f"\n[STEP 1] Loading checkpoint...")
print(f"  Path: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: Checkpoint not found!")
    exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(f"  ✓ Checkpoint loaded")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
print(f"  Config: {checkpoint['config']}")

# ============================================================
# STEP 2: Create Model
# ============================================================
print(f"\n[STEP 2] Creating model...")

config = checkpoint["config"]
model = Image2PointCloud(
    num_points=config["num_points"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"  ✓ Model created")
print(f"  Device: {DEVICE}")
print(f"  Num Points: {config['num_points']}")
print(f"  Hidden Dim: {config['hidden_dim']}")

# ============================================================
# STEP 3: Load and Preprocess Image
# ============================================================
print(f"\n[STEP 3] Loading image...")
print(f"  Path: {IMAGE_PATH}")

if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found!")
    exit(1)

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"  ✓ Image loaded")
print(f"  Original size: {image.size}")
print(f"  Mode: {image.mode}")

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0).to(DEVICE)
print(f"  ✓ Image preprocessed")
print(f"  Tensor shape: {image_tensor.shape}")
print(f"  Tensor dtype: {image_tensor.dtype}")
print(f"  Tensor device: {image_tensor.device}")
print(f"  Tensor min/max: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")

# ============================================================
# STEP 4: Forward Pass
# ============================================================
print(f"\n[STEP 4] Running inference...")

with torch.no_grad():
    pred_points = model(image_tensor)

print(f"  ✓ Inference complete")
print(f"  Output shape: {pred_points.shape}")
print(f"  Output dtype: {pred_points.dtype}")
print(f"  Output device: {pred_points.device}")

# ============================================================
# STEP 5: Analyze Raw Predictions
# ============================================================
print(f"\n[STEP 5] Analyzing predictions...")

points_cpu = pred_points.squeeze(0).cpu().numpy()
print(f"  Points shape: {points_cpu.shape}")
print(f"  Points dtype: {points_cpu.dtype}")
print(f"  Total points: {len(points_cpu)}")

print(f"\n  Statistics:")
print(f"    X range: [{points_cpu[:, 0].min():.6f}, {points_cpu[:, 0].max():.6f}]")
print(f"    Y range: [{points_cpu[:, 1].min():.6f}, {points_cpu[:, 1].max():.6f}]")
print(f"    Z range: [{points_cpu[:, 2].min():.6f}, {points_cpu[:, 2].max():.6f}]")

print(f"\n  Mean: {points_cpu.mean(axis=0)}")
print(f"  Std:  {points_cpu.std(axis=0)}")

# Check if all points are the same
unique_points = np.unique(points_cpu, axis=0)
print(f"\n  Unique points: {len(unique_points)}")
if len(unique_points) == 1:
    print(f"  ⚠️  WARNING: All points are identical!")
    print(f"  Point value: {unique_points[0]}")
elif len(unique_points) < 100:
    print(f"  ⚠️  WARNING: Very few unique points ({len(unique_points)})")

# Check for NaN or Inf
has_nan = np.isnan(points_cpu).any()
has_inf = np.isinf(points_cpu).any()
print(f"\n  Contains NaN: {has_nan}")
print(f"  Contains Inf: {has_inf}")

if has_nan or has_inf:
    print(f"  ⚠️  ERROR: Invalid values detected!")
    exit(1)

# Show first 10 points
print(f"\n  First 10 points:")
for i in range(min(10, len(points_cpu))):
    print(f"    Point {i}: [{points_cpu[i, 0]:.6f}, {points_cpu[i, 1]:.6f}, {points_cpu[i, 2]:.6f}]")

# ============================================================
# STEP 6: Check Point Spread
# ============================================================
print(f"\n[STEP 6] Checking point distribution...")

# Calculate pairwise distances for first 100 points
sample_size = min(100, len(points_cpu))
sample_points = points_cpu[:sample_size]

distances = []
for i in range(sample_size):
    for j in range(i+1, sample_size):
        dist = np.linalg.norm(sample_points[i] - sample_points[j])
        distances.append(dist)

if distances:
    distances = np.array(distances)
    print(f"  Pairwise distances (sample of {sample_size} points):")
    print(f"    Min distance: {distances.min():.6f}")
    print(f"    Max distance: {distances.max():.6f}")
    print(f"    Mean distance: {distances.mean():.6f}")
    print(f"    Std distance: {distances.std():.6f}")
    
    if distances.max() < 0.001:
        print(f"  ⚠️  WARNING: Points are extremely close together!")
        print(f"  This will appear as a single point in viewer")

# ============================================================
# STEP 7: Save Without Scaling (Normalized)
# ============================================================
print(f"\n[STEP 7] Saving normalized PLY...")

output_normalized = OUTPUT_PATH.replace(".ply", "_normalized.ply")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_cpu)

# Add colors
colors = np.zeros_like(points_cpu)
if points_cpu[:, 1].max() - points_cpu[:, 1].min() > 1e-9:
    y_norm = (points_cpu[:, 1] - points_cpu[:, 1].min()) / (points_cpu[:, 1].max() - points_cpu[:, 1].min())
    colors[:, 1] = y_norm  # Green
    colors[:, 0] = 1.0 - y_norm  # Red
else:
    colors[:, :] = [0.5, 0.5, 0.5]  # Gray if no variation

pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(output_normalized, pcd)
print(f"  ✓ Saved: {output_normalized}")

# Verify saved file
try:
    pcd_loaded = o3d.io.read_point_cloud(output_normalized)
    print(f"  Verification:")
    print(f"    Points in saved file: {len(pcd_loaded.points)}")
    print(f"    Has colors: {pcd_loaded.has_colors()}")
except Exception as e:
    print(f"  ⚠️  Error verifying saved file: {e}")

# ============================================================
# STEP 8: Save With Scaling
# ============================================================
print(f"\n[STEP 8] Saving scaled PLYs...")

scales = [1.0, 5.0, 10.0, 50.0, 100.0]

for scale in scales:
    output_scaled = OUTPUT_PATH.replace(".ply", f"_scale{scale:.0f}.ply")
    
    points_scaled = points_cpu * scale
    pcd_scaled = o3d.geometry.PointCloud()
    pcd_scaled.points = o3d.utility.Vector3dVector(points_scaled)
    pcd_scaled.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(output_scaled, pcd_scaled)
    
    print(f"  ✓ Scale {scale:>5.0f}x: {output_scaled}")
    print(f"      Range: X=[{points_scaled[:, 0].min():.2f}, {points_scaled[:, 0].max():.2f}], "
          f"Y=[{points_scaled[:, 1].min():.2f}, {points_scaled[:, 1].max():.2f}], "
          f"Z=[{points_scaled[:, 2].min():.2f}, {points_scaled[:, 2].max():.2f}]")

# ============================================================
# STEP 9: Model Architecture Check
# ============================================================
print(f"\n[STEP 9] Model architecture check...")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Check if model has point_queries
if hasattr(model, 'point_queries'):
    print(f"  ✓ Model has point_queries")
    print(f"    Shape: {model.point_queries.weight.shape}")
else:
    print(f"  ⚠️  WARNING: Model missing point_queries!")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n" + "="*60)
print(f"SUMMARY")
print(f"="*60)
print(f"Model: {CHECKPOINT_PATH}")
print(f"Image: {IMAGE_PATH}")
print(f"Generated: {len(points_cpu)} points")
print(f"Unique points: {len(unique_points)}")
print(f"Point spread: {distances.max():.6f}" if distances else "N/A")
print(f"\nOutput files created:")
for scale in [None] + scales:
    if scale is None:
        fname = output_normalized
    else:
        fname = OUTPUT_PATH.replace(".ply", f"_scale{scale:.0f}.ply")
    if os.path.exists(fname):
        print(f"  ✓ {fname}")

print(f"\nNEXT STEPS:")
print(f"1. Open each PLY file in VSCode (right-click → Open With → 3D Viewer)")
print(f"2. Check which scale looks best")
print(f"3. If all appear as 1 point, model may not be trained properly")
print(f"="*60)
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d
from torchvision import transforms
from scipy.spatial import cKDTree


def compute_curvature_importance(points, k=20):
    """
    Compute importance scores based on local curvature.
    Higher scores = more geometric detail (edges, corners, bends).
    """
    if len(points) < k:
        return np.ones(len(points))
    
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)
    
    importance = np.zeros(len(points))
    
    for i in range(len(points)):
        neighbors = points[indices[i]]
        
        # Covariance matrix for local geometry
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        
        # Eigenvalues indicate local variation
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Curvature estimate
        if eigenvalues[0] > 1e-6:
            curvature = eigenvalues[2] / eigenvalues[0]
        else:
            curvature = 0
        
        importance[i] = curvature
    
    # Normalize
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    return importance


def adaptive_fps_sampling(points, num_samples, curvature_weight=0.6):
    """
    FPS with curvature bias - samples more from high-detail regions.
    """
    N = points.shape[0]
    
    if N <= num_samples:
        if N == 0:
            return np.zeros((num_samples, 3), dtype=np.float32)
        indices = np.random.choice(N, num_samples, replace=True)
        return points[indices]
    
    print(f"  Computing curvature importance for {N} points...")
    importance = compute_curvature_importance(points, k=min(20, N))
    
    sampled_indices = []
    distances = np.ones(N) * 1e10
    farthest = np.argmax(importance)
    
    for i in range(num_samples):
        sampled_indices.append(farthest)
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=-1)
        mask = dist < distances
        distances[mask] = dist[mask]
        scores = distances * (1 + curvature_weight * importance)
        farthest = np.argmax(scores)
        
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{num_samples} points sampled")
    
    return points[np.array(sampled_indices)]


def multi_scale_sampling(points, num_samples):
    """
    Multi-scale sampling: combines global structure with local detail.
    """
    if len(points) == 0:
        return np.zeros((num_samples, 3), dtype=np.float32)
    
    if len(points) <= num_samples:
        indices = np.random.choice(len(points), num_samples, replace=True)
        return points[indices]
    
    print(f"  Multi-scale sampling: {num_samples} points")
    
    global_samples = num_samples // 2
    detail_samples = num_samples - global_samples
    
    print(f"    Global structure: {global_samples} points (FPS)")
    global_points = farthest_point_sampling(points, global_samples)
    
    print(f"    Detail sampling: {detail_samples} points (curvature-based)")
    importance = compute_curvature_importance(points, k=min(20, len(points)))
    
    probs = importance / (importance.sum() + 1e-8)
    detail_indices = np.random.choice(len(points), detail_samples, replace=False, p=probs)
    detail_points = points[detail_indices]
    
    sampled_points = np.vstack([global_points, detail_points])
    
    return sampled_points


def farthest_point_sampling(points, num_samples):
    """Standard FPS for comparison."""
    N = points.shape[0]
    
    if N <= num_samples:
        if N == 0:
            return np.zeros((num_samples, 3), dtype=np.float32)
        indices = np.random.choice(N, num_samples, replace=True)
        return points[indices]
    
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        sampled_indices[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=-1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
        
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i+1}/{num_samples} points sampled")
    
    return points[sampled_indices]


def analyze_sample_quality(original_points, sampled_points, method_name):
    """Analyze sampling quality metrics."""
    print(f"\n  Quality Analysis ({method_name}):")
    print(f"    Original points: {len(original_points)}")
    print(f"    Sampled points: {len(sampled_points)}")
    print(f"    Sampling ratio: {len(sampled_points)/len(original_points)*100:.1f}%")
    
    tree = cKDTree(sampled_points)
    distances, _ = tree.query(original_points, k=1)
    avg_coverage = distances.mean()
    max_gap = distances.max()
    
    print(f"    Average coverage error: {avg_coverage:.6f}")
    print(f"    Maximum gap: {max_gap:.6f}")
    
    if len(sampled_points) > 1:
        tree_sampled = cKDTree(sampled_points)
        distances_internal, _ = tree_sampled.query(sampled_points, k=2)
        nn_dist = distances_internal[:, 1]
        uniformity = nn_dist.std()
        print(f"    Point uniformity (lower=better): {uniformity:.6f}")


def process_ply_direct(ply_path, num_points=8192, output_suffix=""):
    """
    Process a PLY file directly - compare all sampling methods.
    """
    print("\n" + "="*70)
    print(f"PROCESSING PLY: {ply_path}")
    print("="*70)
    
    if not os.path.exists(ply_path):
        print(f"ERROR: PLY file not found at {ply_path}")
        return
    
    ply_filename = os.path.basename(ply_path)
    ply_name = os.path.splitext(ply_filename)[0]
    
    print(f"\nFile: {ply_filename}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    original_points = np.asarray(pcd.points, dtype=np.float32)
    
    print(f"\nOriginal Point Cloud:")
    print(f"  Points: {len(original_points)}")
    print(f"  Bounds: X[{original_points[:, 0].min():.3f}, {original_points[:, 0].max():.3f}]")
    print(f"          Y[{original_points[:, 1].min():.3f}, {original_points[:, 1].max():.3f}]")
    print(f"          Z[{original_points[:, 2].min():.3f}, {original_points[:, 2].max():.3f}]")
    
    # Create output directory
    ply_dir = os.path.dirname(ply_path)
    output_dir = os.path.join(ply_dir, f"analysis_{ply_name}{output_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    orig_file = os.path.join(output_dir, f"{ply_name}_ORIGINAL_{len(original_points)}pts.ply")
    o3d.io.write_point_cloud(orig_file, pcd)
    print(f"\n✓ Saved original: {orig_file}")
    
    # Test different sampling methods
    methods = {
        'adaptive_fps': ('Adaptive FPS (Curvature-Aware)', adaptive_fps_sampling),
        'multi_scale': ('Multi-Scale (Global+Detail)', multi_scale_sampling),
        'fps': ('Standard FPS', farthest_point_sampling),
        'random': ('Random Sampling', None)
    }
    
    print(f"\n{'='*70}")
    print(f"TESTING SAMPLING METHODS (Target: {num_points} points)")
    print(f"{'='*70}")
    
    for method_key, (method_name, method_func) in methods.items():
        print(f"\n[{method_name}]")
        
        if method_func:
            sampled_points = method_func(original_points.copy(), num_points)
        else:
            # Random sampling
            if len(original_points) > num_points:
                indices = np.random.choice(len(original_points), num_points, replace=False)
                sampled_points = original_points[indices]
            else:
                sampled_points = original_points
        
        # Normalize to unit sphere
        centroid = sampled_points.mean(axis=0)
        sampled_points_norm = sampled_points - centroid
        max_dist = np.max(np.sqrt(np.sum(sampled_points_norm**2, axis=1)))
        if max_dist > 0:
            sampled_points_norm = sampled_points_norm / max_dist
        
        # Quality analysis
        analyze_sample_quality(original_points, sampled_points, method_name)
        
        # Save sampled version
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(sampled_points_norm)
        out_file = os.path.join(
            output_dir,
            f"{ply_name}_{method_key}_{num_points}pts.ply"
        )
        o3d.io.write_point_cloud(out_file, pcd_out)
        print(f"  ✓ Saved: {out_file}")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}/")
    print(f"\n🔍 Open these PLY files in MeshLab or CloudCompare to compare:")
    print(f"   1. {ply_name}_ORIGINAL - Full resolution ground truth")
    print(f"   2. {ply_name}_adaptive_fps - Best for edges/corners/bends")
    print(f"   3. {ply_name}_multi_scale - Best for complex geometry")
    print(f"   4. {ply_name}_fps - Good overall coverage")
    print(f"   5. {ply_name}_random - Baseline comparison")
    print(f"\n💡 Recommended: Use 'adaptive_fps' with 8192-10240 points")
    print(f"{'='*70}\n")


class Image2PointCloudDataset(Dataset):
    """
    Dataset with ADVANCED sampling for superior structure capture.
    """
    
    def __init__(self, data_dir, num_points=8192, augment=False, 
                 sampling_method='adaptive_fps'):
        """
        Args:
            data_dir: Path to train/ or test/ directory
            num_points: Points to sample (4096, 8192, 10240 recommended)
            augment: Whether to apply data augmentation
            sampling_method: 'adaptive_fps' (best!), 'multi_scale', 'fps', 'random'
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.augment = augment
        self.sampling_method = sampling_method
        
        # Load manifest
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        print(f"\nDataset: {data_dir}")
        print(f"  Samples: {len(self.manifest)}")
        print(f"  Points per sample: {num_points}")
        print(f"  Sampling method: {sampling_method}")
        print(f"  Augmentation: {augment}")
        
        if num_points < 4096:
            print(f"  ⚠️  WARNING: {num_points} points may be insufficient!")
        
        # Image preprocessing
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Data augmentation
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ]) if augment else None
    
    def __len__(self):
        return len(self.manifest)
    
    def sample_points(self, points):
        """Apply selected sampling method."""
        if self.sampling_method == 'adaptive_fps':
            return adaptive_fps_sampling(points, self.num_points, curvature_weight=0.6)
        elif self.sampling_method == 'multi_scale':
            return multi_scale_sampling(points, self.num_points)
        elif self.sampling_method == 'fps':
            return farthest_point_sampling(points, self.num_points)
        else:  # random
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                return points[indices]
            elif len(points) < self.num_points:
                pad_size = self.num_points - len(points)
                if len(points) > 0:
                    indices = np.random.choice(len(points), pad_size, replace=True)
                    padding = points[indices]
                else:
                    padding = np.zeros((pad_size, 3), dtype=np.float32)
                return np.vstack([points, padding])
            return points
    
    def __getitem__(self, idx):
        sample = self.manifest[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, sample["image"])
        image = Image.open(img_path).convert("RGB")
        
        if self.augment and self.aug_transform:
            image = self.aug_transform(image)
        
        image = self.img_transform(image)
        
        # Load point cloud
        ply_path = os.path.join(self.data_dir, sample["ply"])
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Sample points
        points = self.sample_points(points)
        
        # Normalize to unit sphere
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        points = torch.from_numpy(points).float()
        
        return {
            "image": image,
            "points": points,
            "id": sample["id"],
            "object": sample["object"]
        }


if __name__ == "__main__":
    # Process the specific PLY file directly
    ply_path = "dataset/train/plys/00056.ply"
    
    print("\n" + "="*70)
    print("POINT CLOUD SAMPLING ANALYSIS")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(ply_path):
        print(f"\nERROR: PLY file not found at: {ply_path}")
        print("\nPlease update the 'ply_path' variable in the script to point to your PLY file.")
        print("Example: ply_path = 'dataset/train/plys/00001.ply'")
    else:
        # Process with 8192 points
        process_ply_direct(ply_path, num_points=8192)
        
        # Also try with higher resolution for maximum detail
        print("\n\n" + "="*70)
        print("BONUS: Testing with 10240 points (maximum detail)")
        print("="*70)
        process_ply_direct(ply_path, num_points=10240, output_suffix="_10240")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open the analysis_* folder in dataset/train/plys/")
    print("2. Load all PLY files in MeshLab or CloudCompare")
    print("3. Compare visual quality of different sampling methods")
    print("4. Choose the best method for your training!")
    print("="*70)
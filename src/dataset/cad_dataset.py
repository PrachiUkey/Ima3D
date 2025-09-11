# Loads OBJ → point clouds
 # Loads real-world images + masks
"""
src/data/mesh_dataset.py

Dataset for loading 3D CAD meshes, sampling point clouds, 
and computing signed distance fields (SDFs).

Dependencies:
    pip install torch trimesh numpy
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.mesh_utils import load_mesh, sample_points_from_mesh


class MeshDataset(Dataset):
    """
    Dataset for loading 3D meshes (OBJ/GLB) and sampling training data.
    """

    def __init__(self, mesh_dir: str, num_points: int = 2048, sdf: bool = False, split: str = "train", split_ratio=0.8):
        """
        Args:
            mesh_dir: directory containing mesh files (.obj/.glb)
            num_points: number of 3D points to sample per mesh
            sdf: whether to compute SDF values (True) or only sample surface points (False)
            split: "train" or "val"
            split_ratio: fraction of data used for training
        """
        self.mesh_dir = mesh_dir
        self.num_points = num_points
        self.sdf = sdf

        self.files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith((".obj", ".glb"))]
        self.files.sort()

        # Train/val split
        split_idx = int(len(self.files) * split_ratio)
        if split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_path = self.files[idx]
        mesh = load_mesh(mesh_path)

        # Sample point cloud
        points, normals = sample_points_from_mesh(mesh, self.num_points)

        if self.sdf:
            # Sample random 3D points around bounding box
            bbox_min, bbox_max = mesh.bounds
            rand_points = np.random.uniform(bbox_min, bbox_max, size=(self.num_points, 3))

            # Compute SDF: positive outside, negative inside
            signed_distance = mesh.nearest.signed_distance(rand_points)

            points = rand_points
            sdf_values = signed_distance.astype(np.float32)
            return torch.from_numpy(points).float(), torch.from_numpy(sdf_values).float()

        # Default: return surface point cloud
        return torch.from_numpy(points).float(), torch.from_numpy(normals).float()

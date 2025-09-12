# src/dataset/mesh_dataset.py

import os
import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np

class MeshDataset(Dataset):
    def __init__(self, mesh_dir, classes, num_points=200000, sampling='surface', cache=True):
        """
        mesh_dir: root folder containing class folders
        classes: list of classes to load
        num_points: number of points to sample per mesh
        sampling: 'surface' or 'importance'
        cache: whether to save/load sampled points to speed up
        """
        self.mesh_dir = mesh_dir
        self.classes = classes
        self.num_points = num_points
        self.sampling = sampling
        self.cache = cache
        self.samples = []

        for cls in classes:
            class_dir = os.path.join(mesh_dir, cls)
            if not os.path.exists(class_dir):
                continue

            obj_folders = sorted(os.listdir(class_dir))
            for obj_folder in obj_folders:
                folder_path = os.path.join(class_dir, obj_folder)
                if not os.path.isdir(folder_path):
                    continue

                obj_files = [f for f in os.listdir(folder_path) if f.endswith(".obj")]
                if not obj_files:
                    print(f"[Warning] No .obj file found in {folder_path}, skipping")
                    continue

                obj_path = os.path.join(folder_path, obj_files[0])

                # rename to model.obj if necessary
                model_path = os.path.join(folder_path, "model.obj")
                if obj_files[0] != "model.obj":
                    os.rename(obj_path, model_path)
                    obj_path = model_path

                points_path = os.path.join(folder_path, f"points_{num_points}.npy")

                self.samples.append({
                    "path": obj_path,
                    "class": cls,
                    "points_path": points_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.cache and os.path.exists(sample["points_path"]):
            points = np.load(sample["points_path"])
        else:
            mesh = trimesh.load(sample["path"], force='mesh')
            if mesh is None or mesh.is_empty:
                raise ValueError(f"Failed to load a valid mesh from {sample['path']}")

            if self.sampling == 'surface':
                points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            elif self.sampling == 'importance':
                points, _ = self.importance_sample(mesh, self.num_points)
            else:
                raise ValueError(f"Unknown sampling type: {self.sampling}")

            if self.cache:
                np.save(sample["points_path"], points)

        points = torch.from_numpy(points.astype(np.float32))
        normals = torch.zeros_like(points)
        return points, normals, sample["class"]

    def importance_sample(self, mesh, num_points):
        """Simple importance sampling based on vertex curvature"""
        try:
            curvature = mesh.vertex_defects
        except:
            curvature = np.ones(len(mesh.vertices))
        curvature = np.abs(curvature)
        probs = curvature / curvature.sum()
        indices = np.random.choice(len(mesh.vertices), size=num_points, p=probs)
        points = mesh.vertices[indices]
        return points, None

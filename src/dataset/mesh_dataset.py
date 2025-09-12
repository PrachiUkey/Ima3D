import os
import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np

class MeshDataset(Dataset):
    def __init__(self, mesh_dir, classes, num_points=200000, sampling='surface'):
        """
        mesh_dir: path to mesh folder
        classes: list of classes
        num_points: number of points to sample
        sampling: 'surface' (uniform) or 'importance' (curvature-based)
        """
        self.mesh_dir = mesh_dir
        self.classes = classes
        self.num_points = num_points
        self.sampling = sampling

        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(mesh_dir, cls)
            subdirs = sorted(os.listdir(cls_dir))
            for sub in subdirs:
                mesh_file = os.path.join(cls_dir, sub, "model.obj")
                self.samples.append({
                    "mesh": mesh_file,
                    "class": cls
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mesh = trimesh.load(sample["mesh"], force='mesh')
        if mesh.is_empty:
            raise ValueError(f"Empty mesh at {sample['mesh']}")

        if self.sampling == 'surface':
            points, face_idx = trimesh.sample.sample_surface(mesh, self.num_points)
        elif self.sampling == 'importance':
            curvature = self.compute_vertex_curvature(mesh)
            face_curvature = curvature[mesh.faces].mean(axis=1)
            face_prob = face_curvature / face_curvature.sum()
            face_idx = np.random.choice(len(mesh.faces), size=self.num_points, p=face_prob)
            triangles = mesh.triangles[face_idx]
            r1 = np.sqrt(np.random.rand(self.num_points, 1))
            r2 = np.random.rand(self.num_points, 1)
            points = (1 - r1) * triangles[:, 0] + r1 * (1 - r2) * triangles[:, 1] + r1 * r2 * triangles[:, 2]
        else:
            raise ValueError(f"Unknown sampling mode {self.sampling}")

        points = torch.from_numpy(points.astype('float32'))

        # Normals
        if self.sampling == 'surface':
            normals = mesh.face_normals[face_idx]
        else:
            v0, v1, v2 = triangles[:,0], triangles[:,1], triangles[:,2]
            normals = np.cross(v1 - v0, v2 - v0)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normals = torch.from_numpy(normals.astype('float32'))

        return points, normals, sample["class"]

    def compute_vertex_curvature(self, mesh):
        """
        Simple curvature approximation: mean difference of vertex normals
        """
        vertex_normals = mesh.vertex_normals
        curvature = np.zeros(len(mesh.vertices))
        counts = np.zeros(len(mesh.vertices))
        for face in mesh.faces:
            n0, n1, n2 = vertex_normals[face]
            diff0 = np.linalg.norm(n0 - n1) + np.linalg.norm(n0 - n2)
            diff1 = np.linalg.norm(n1 - n0) + np.linalg.norm(n1 - n2)
            diff2 = np.linalg.norm(n2 - n0) + np.linalg.norm(n2 - n1)
            curvature[face[0]] += diff0
            curvature[face[1]] += diff1
            curvature[face[2]] += diff2
            counts[face] += 2
        curvature /= counts
        curvature += 1e-6
        return curvature

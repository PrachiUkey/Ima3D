# src/dataset/PairedDataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import trimesh
from torchvision import transforms

class PairedDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mesh_dir, num_points=2048, img_size=224):
        """
        img_dir: path to images (data/img/<class>/<image.png>)
        mask_dir: path to masks (data/mask/<class>/<mask.png>)
        mesh_dir: path to meshes (data/model/<class>/<subclass>/model.obj)
        num_points: number of points to sample per mesh
        img_size: resize images to this size
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mesh_dir = mesh_dir
        self.num_points = num_points

        # Gather all images and their class
        self.img_files = []
        for cls in os.listdir(img_dir):
            cls_path = os.path.join(img_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.img_files.append((cls, fname))

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        cls, fname = self.img_files[idx]
        base = os.path.splitext(fname)[0]

        # --- Image ---
        img_path = os.path.join(self.img_dir, cls, fname)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # --- Mask ---
        mask_path = os.path.join(self.mask_dir, cls, base + ".png")
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)

        # --- Mesh: pick a random subclass in that class ---
        mesh_cls_dir = os.path.join(self.mesh_dir, cls)
        subclasses = [d for d in os.listdir(mesh_cls_dir) if os.path.isdir(os.path.join(mesh_cls_dir, d))]
        chosen_subclass = np.random.choice(subclasses)
        mesh_path = os.path.join(mesh_cls_dir, chosen_subclass, "model.obj")

        mesh = trimesh.load(mesh_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'geometry'):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)

        # --- Sample fixed number of vertices ---
        if vertices.shape[0] >= self.num_points:
            choice = np.random.choice(vertices.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(vertices.shape[0], self.num_points, replace=True)
        vertices = vertices[choice, :]

        return img, mask, vertices, cls

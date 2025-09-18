import os
import torch
from torch.utils.data import Dataset
import cv2
import trimesh
import numpy as np

def sample_pointcloud(obj_path, num_points=1024):
    mesh = trimesh.load(obj_path, process=True)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)

class ShapeDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, transform=None):
        """
        root_dir: data/renders/
        Each subclass folder has model.obj and rendered views
        """
        self.samples = []
        self.num_points = num_points
        self.transform = transform

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for subclass_name in os.listdir(class_path):
                subclass_path = os.path.join(class_path, subclass_name)
                obj_path = os.path.join(subclass_path, "model.obj")
                if not os.path.exists(obj_path):
                    continue
                # Collect all rendered views
                views = sorted([os.path.join(subclass_path, f)
                                for f in os.listdir(subclass_path)
                                if f.endswith(".png")])
                for img_path in views:
                    self.samples.append({
                        "img": img_path,
                        "obj": obj_path
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["img"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W

        pc = sample_pointcloud(sample["obj"], self.num_points)
        pc = torch.from_numpy(pc)

        return img, pc

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import trimesh
import numpy as np

class ImageMeshDataset(Dataset):
    def __init__(self, images_dir, models_dir, classes=None, image_size=(128,128), num_points=5000):
        """
        images_dir: data/images
        models_dir: data/models
        classes: list of classes, if None, read from folders
        """
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.image_size = image_size
        self.num_points = num_points

        if classes is None:
            self.classes = sorted(os.listdir(images_dir))
        else:
            self.classes = classes

        # build image paths and labels
        self.samples = []
        for cls in self.classes:
            img_folder = os.path.join(images_dir, cls)
            imgs = sorted(os.listdir(img_folder))
            for img_file in imgs:
                img_path = os.path.join(img_folder, img_file)
                # pick first CAD model of this class, or None
                model_cls_folder = os.path.join(models_dir, cls)
                if os.path.exists(model_cls_folder):
                    model_subs = sorted(os.listdir(model_cls_folder))
                    if model_subs:
                        model_file = os.path.join(model_cls_folder, model_subs[0], "model.obj")
                        if not os.path.exists(model_file):
                            model_file = None
                    else:
                        model_file = None
                else:
                    model_file = None

                self.samples.append((img_path, model_file, cls))

        self.cls_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mesh_path, cls = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # mesh
        if mesh_path is not None:
            mesh = trimesh.load(mesh_path, process=False)
            points = mesh.sample(self.num_points)
            points = torch.tensor(points, dtype=torch.float32)
        else:
            # placeholder random points if mesh missing
            points = torch.rand(self.num_points, 3)

        label = self.cls_to_idx[cls]
        return img_tensor, points, label

# Optional: helper to get a single sample
def get_sample_image_mesh(device="cpu", image_size=(128,128), num_points=5000):
    dataset = ImageMeshDataset("data/images", "data/models", image_size=image_size, num_points=num_points)
    img, mesh, _ = dataset[0]
    return img.unsqueeze(0).to(device), mesh.unsqueeze(0).to(device)

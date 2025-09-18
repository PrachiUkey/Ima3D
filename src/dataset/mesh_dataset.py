import os
import trimesh
from torch.utils.data import Dataset

class MeshDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with coarse class folders (bed, chair, ...)
                            Each contains subclass folders (bed_0, bed_1, ...)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mesh_paths, self.coarse_labels, self.subclass_labels = self._load_meshes()

    def _load_meshes(self):
        mesh_paths, coarse_labels, subclass_labels = [], [], []
        for coarse_class in os.listdir(self.root_dir):
            coarse_path = os.path.join(self.root_dir, coarse_class)
            if not os.path.isdir(coarse_path):
                continue
            for subclass in os.listdir(coarse_path):
                subclass_path = os.path.join(coarse_path, subclass)
                if not os.path.isdir(subclass_path):
                    continue
                for fname in os.listdir(subclass_path):
                    if fname.endswith(".obj"):
                        mesh_paths.append(os.path.join(subclass_path, fname))
                        coarse_labels.append(coarse_class)
                        subclass_labels.append(subclass)  # finer label
        print(f"[MeshDataset] Loaded {len(mesh_paths)} meshes from {self.root_dir}")
        return mesh_paths, coarse_labels, subclass_labels

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        mesh_path = self.mesh_paths[idx]
        coarse_label = self.coarse_labels[idx]
        subclass_label = self.subclass_labels[idx]
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
            if self.transform:
                mesh = self.transform(mesh)
        except Exception as e:
            raise ValueError(f"Failed to load mesh {mesh_path}: {e}")

        return mesh, coarse_label, subclass_label

import os
import torch
import trimesh

class MeshDataset(torch.utils.data.Dataset):
    def __init__(self, mesh_dir, num_points=100000, sampling="surface"):
        """
        mesh_dir: root folder, e.g., data/model/
        expects: data/model/<class>/<class>_*/model.obj
        """
        self.mesh_files = []  # list of tuples: (points_tensor, class)
        self.num_points = num_points
        self.sampling = sampling

        # iterate over classes
        for cls_name in os.listdir(mesh_dir):
            cls_folder = os.path.join(mesh_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue

            # iterate over subfolders like bed_0, bed_1, ...
            for sub in os.listdir(cls_folder):
                subfolder = os.path.join(cls_folder, sub)
                obj_path = os.path.join(subfolder, "model.obj")
                if os.path.exists(obj_path):
                    points = self.load_mesh(obj_path)
                    self.mesh_files.append((points, cls_name.lower()))

        if len(self.mesh_files) == 0:
            raise ValueError(f"No mesh files found in {mesh_dir}!")

    def load_mesh(self, path):
        mesh = trimesh.load(path, process=False)

        # If Scene, merge all geometries
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError(f"Empty Scene: {path}")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        if mesh.is_empty:
            raise ValueError(f"Failed to load mesh: {path}")

        # Sample points
        points = mesh.sample(self.num_points) if self.sampling=="surface" else mesh.vertices
        points = torch.tensor(points, dtype=torch.float32)
        return points

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        points, cls = self.mesh_files[idx]
        return points, cls

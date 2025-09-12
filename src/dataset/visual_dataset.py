import os
import trimesh
from .mesh_dataset import MeshDataset  # relative import

def visualize_and_save(dataset, save_dir="outputs/visualized", top_n=5):
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(top_n, len(dataset))):
        points, normals, cls = dataset[idx]
        print(f"Visualizing {cls} mesh {idx} with {points.shape[0]} points")

        # Convert to trimesh PointCloud
        cloud = trimesh.PointCloud(vertices=points.numpy())
        
        # Show interactive viewer
        cloud.show()

        # Save as PLY
        save_path = os.path.join(save_dir, f"{cls}_{idx}.ply")
        cloud.export(save_path)
        print(f"Saved PLY to {save_path}")

if __name__ == "__main__":
    # Surface sampled dataset
    dataset_surface = MeshDataset(mesh_dir="data/model", classes=["bed","chair"], num_points=200000, sampling='surface')
    print("Visualizing surface sampled point clouds...")
    visualize_and_save(dataset_surface, save_dir="outputs/visualized_surface", top_n=5)

    # Importance sampled dataset
    dataset_importance = MeshDataset(mesh_dir="data/model", classes=["bed","chair"], num_points=200000, sampling='importance')
    print("Visualizing importance sampled point clouds...")
    visualize_and_save(dataset_importance, save_dir="outputs/visualized_importance", top_n=5)

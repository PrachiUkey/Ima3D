"""

Load a mesh (.obj/.ply/.glb), sample a point cloud,
save it (PLY/XYZ), and optionally reconstruct a mesh
via Convex Hull or Marching Cubes.
"""

import os
import numpy as np
import trimesh

from src.utils.mesh_utils import (
    load_obj_as_pointcloud,
    visualize_point_cloud,
    reconstruct_mesh_convex_hull,
    reconstruct_mesh_marching_cubes,
)


# ----------------- Save Functions -----------------
def save_pointcloud_ply(points: np.ndarray, out_path: str):
    """Save (N,3) point cloud as a .ply file (Blender-readable)."""
    cloud = trimesh.PointCloud(points)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cloud.export(out_path)
    print(f"[INFO] Point cloud saved to {out_path}")


def save_pointcloud_xyz(points: np.ndarray, out_path: str):
    """Save (N,3) point cloud as .xyz (Blender-readable)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, points, fmt="%.6f")
    print(f"[INFO] Point cloud saved to {out_path}")


# ----------------- Main -----------------
if __name__ == "__main__":
    # ====== CONFIG ======
    mesh_path = "data/model/bed/bed_18/model.obj"   # Input mesh
    out_cloud_path = "outputs/pointclouds/pointcloud_18.ply"      # Point cloud output
    num_points = 200000
    normalize = True
    visualize = True
    reconstruct = "mc"   # options: "hull", "mc", or None

    # ====== Load + Sample ======
    print(f"[INFO] Looking for mesh at: {os.path.abspath(mesh_path)}")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    points = load_obj_as_pointcloud(mesh_path, num_points=num_points, normalize=normalize)
    print(f"[INFO] Sampled {points.shape[0]} points from mesh")

    # ====== Save Point Cloud ======
    if out_cloud_path.endswith(".ply"):
        save_pointcloud_ply(points, out_cloud_path)
    elif out_cloud_path.endswith(".xyz"):
        save_pointcloud_xyz(points, out_cloud_path)
    else:
        raise ValueError("Output file must end with .ply or .xyz")

    # ====== Reconstruct Mesh ======
    if reconstruct == "hull":
        mesh = reconstruct_mesh_convex_hull(points)
        mesh.export("outputs/reconstructed_hull.obj")
        print("[INFO] Convex Hull mesh saved to outputs/reconstructed_hull.obj")

    elif reconstruct == "mc":
        mesh = reconstruct_mesh_marching_cubes(points, voxel_size=0.05)
        mesh.export("outputs/reconstructed_mc.obj")
        print("[INFO] Marching Cubes mesh saved to outputs/reconstructed_mc.obj")

    # ====== Visualization ======
    if visualize:
        visualize_point_cloud(points, title="Sampled Point Cloud")

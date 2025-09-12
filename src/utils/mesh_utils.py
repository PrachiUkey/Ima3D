"""
src/utils/mesh_utils.py

Extended with reconstruction:
- Convex hull triangulation
- Marching Cubes surface reconstruction
"""

from typing import Optional
import numpy as np
import trimesh
import os
from scipy.spatial import ConvexHull
from skimage import measure  # marching cubes


# ----------------- Existing imports -----------------
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


# ----------------- Mesh Loading -----------------
def load_mesh(path: str) -> trimesh.Trimesh:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load(path, force="mesh")
    if mesh is None or mesh.is_empty:
        raise ValueError(f"Failed to load a valid mesh from {path}")

    if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    return mesh


# ----------------- Sampling -----------------
def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int = 1024) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    if points.shape[0] < num_points:
        if points.shape[0] == 0:
            raise ValueError("Mesh sampling returned zero points.")
        idx = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
        points = np.vstack([points, points[idx]])
    return points.astype(np.float32)


# ----------------- Normalization -----------------
def normalize_point_cloud(points: np.ndarray, to_sphere: bool = True) -> np.ndarray:
    centroid = points.mean(axis=0)
    pts = points - centroid

    if to_sphere:
        max_dist = np.linalg.norm(pts, axis=1).max()
        if max_dist > 0:
            pts /= max_dist
    else:
        max_abs = np.abs(pts).max()
        if max_abs > 0:
            pts /= max_abs

    return pts.astype(np.float32)


# ----------------- Combined Loader -----------------
def load_obj_as_pointcloud(path: str, num_points: int = 1024, normalize: bool = True) -> np.ndarray:
    mesh = load_mesh(path)
    points = sample_point_cloud(mesh, num_points)
    if normalize:
        points = normalize_point_cloud(points)
    return points


# ----------------- Visualization -----------------
def visualize_point_cloud(points: np.ndarray, title: Optional[str] = None, s: int = 2):
    if not _HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib not available.")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)

    plt.show()


# ----------------- Caching -----------------
def cache_point_cloud(mesh_path: str, out_path: str, num_points: int = 1024, normalize: bool = True) -> str:
    points = load_obj_as_pointcloud(mesh_path, num_points=num_points, normalize=normalize)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, points)
    return out_path


# ----------------- NEW: Convex Hull Reconstruction -----------------
def reconstruct_mesh_convex_hull(points: np.ndarray) -> trimesh.Trimesh:
    """
    Reconstruct a mesh from point cloud using convex hull triangulation.
    Works best for convex objects.
    """
    hull = ConvexHull(points)
    mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
    return mesh


# ----------------- NEW: Marching Cubes Reconstruction -----------------
def reconstruct_mesh_marching_cubes(points: np.ndarray, voxel_size: float = 0.05) -> trimesh.Trimesh:
    """
    Reconstruct a mesh using marching cubes from a point cloud.

    Args:
        points: (N,3) point cloud
        voxel_size: resolution of voxel grid

    Returns:
        trimesh.Trimesh reconstructed mesh
    """
    # Create voxel grid
    min_bounds = points.min(axis=0) - voxel_size
    max_bounds = points.max(axis=0) + voxel_size
    dims = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)

    volume = np.zeros(dims, dtype=np.uint8)
    indices = ((points - min_bounds) / voxel_size).astype(int)
    indices = np.clip(indices, 0, dims - 1)
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    # Apply marching cubes
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(voxel_size, voxel_size, voxel_size))
    verts += min_bounds

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

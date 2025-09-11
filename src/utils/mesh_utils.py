"""
src/utils/mesh_utils.py

Utilities to load 3D mesh files (.obj/.ply/.glb), sample point clouds from
the surface, normalize point clouds and visualize them.

Dependencies:
    pip install trimesh numpy matplotlib
"""
from typing import Optional
import numpy as np
import trimesh
import os

# optional import for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from path using trimesh.

    Args:
        path: path to a mesh file (.obj, .ply, .glb, ...)

    Returns:
        trimesh.Trimesh object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load(path, force='mesh')
    if mesh is None or mesh.is_empty:
        raise ValueError(f"Failed to load a valid mesh from {path}")
    if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'dump'):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int = 1024) -> np.ndarray:
    """
    Uniformly sample points over mesh surface.

    Args:
        mesh: trimesh.Trimesh object
        num_points: number of points to sample

    Returns:
        points: np.ndarray of shape (num_points, 3)
    """
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    if points.shape[0] < num_points:
        if points.shape[0] == 0:
            raise ValueError("Mesh sampling returned zero points. Mesh may be invalid.")
        needed = num_points - points.shape[0]
        idx = np.random.choice(points.shape[0], needed, replace=True)
        pad = points[idx]
        points = np.vstack([points, pad])
    return points.astype(np.float32)


def normalize_point_cloud(points: np.ndarray, to_sphere: bool = True) -> np.ndarray:
    """
    Normalize a point cloud.

    - Centers the points at the origin (subtract mean).
    - If to_sphere=True: scales so max distance from origin == 1 (unit sphere).
    - Else: scales to fit in [-1,1] cube.

    Args:
        points: (N,3) array
        to_sphere: whether to normalize to unit sphere

    Returns:
        normalized points: (N,3) np.ndarray
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N,3)")

    centroid = points.mean(axis=0)
    pts = points - centroid

    if to_sphere:
        max_dist = np.linalg.norm(pts, axis=1).max()
        if max_dist > 0:
            pts = pts / max_dist
    else:
        max_abs = np.abs(pts).max()
        if max_abs > 0:
            pts = pts / max_abs

    return pts.astype(np.float32)


def load_obj_as_pointcloud(path: str, num_points: int = 1024, normalize: bool = True) -> np.ndarray:
    """
    Helper that loads a mesh file and returns a normalized point cloud.

    Args:
        path: path to mesh (.obj/.ply/.glb...)
        num_points: number of points to sample
        normalize: whether to center+scale to unit sphere

    Returns:
        points: (num_points, 3) float32
    """
    mesh = load_mesh(path)
    points = sample_point_cloud(mesh, num_points=num_points)
    if normalize:
        points = normalize_point_cloud(points, to_sphere=True)
    return points


def visualize_point_cloud(points: np.ndarray, title: Optional[str] = None, s: int = 2):
    """
    Quick 3D scatter of a point cloud using matplotlib.

    Args:
        points: (N,3) array
        title: optional title
        s: marker size
    """
    if not _HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib not available. Install matplotlib to visualize.")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N,3)")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        ax.set_title(title)

    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def cache_point_cloud(mesh_path: str, out_path: str, num_points: int = 1024, normalize: bool = True) -> str:
    """
    Sample mesh and save point cloud to out_path (.npy).

    Returns the saved path.
    """
    points = load_obj_as_pointcloud(mesh_path, num_points=num_points, normalize=normalize)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, points)
    return out_path

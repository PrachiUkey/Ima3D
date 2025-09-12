# src/utils/mesh_reconstruction.py

import numpy as np
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reconstruct_mesh_from_points(points: np.ndarray, method: str = "marching_cubes", resolution: int = 32):
    """
    Reconstruct a mesh from 3D points.

    Args:
        points: (N,3) point cloud
        method: "marching_cubes" or "delaunay"
        resolution: voxel grid resolution for marching cubes

    Returns:
        trimesh.Trimesh object
    """
    if method == "delaunay":
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
        return mesh

    elif method == "marching_cubes":
        # Create voxel grid
        voxel = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        pts = ((points - points.min(0)) / (points.ptp(0)) * (resolution - 1)).astype(int)
        voxel[pts[:,0], pts[:,1], pts[:,2]] = 1.0

        # Use marching cubes
        verts, faces, _, _ = measure.marching_cubes(voxel, level=0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        return mesh
    else:
        raise ValueError("Method must be 'marching_cubes' or 'delaunay'")

def save_mesh(mesh: trimesh.Trimesh, out_path: str):
    """Save mesh to .ply or .obj file."""
    mesh.export(out_path)
    print(f"[INFO] Mesh saved to {out_path}")

def visualize_mesh(mesh: trimesh.Trimesh):
    """Visualize mesh using matplotlib 3D plot."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    faces = mesh.faces

    # Plot faces
    for face in faces:
        tri = vertices[face]
        ax.add_collection3d(
            plt.Poly3DCollection([tri], facecolor='cyan', edgecolor='k', alpha=0.6)
        )

    scale = mesh.bounds
    ax.set_xlim(scale[0,0], scale[1,0])
    ax.set_ylim(scale[0,1], scale[1,1])
    ax.set_zlim(scale[0,2], scale[1,2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

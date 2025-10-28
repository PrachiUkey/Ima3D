import numpy as np
import trimesh
import os
import torch
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree

from config import config

def get_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([geom for geom in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh


    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    return mesh, o3d_mesh

def get_padded_bounds():
    bounds = np.array(config.data.sdf_bounds)
    padding = np.array(config.data.sdf_padding)
    extent = bounds[1] - bounds[0]
    padded_bounds = np.array([bounds[0] - extent * padding, bounds[1] + extent * padding])
    return padded_bounds

def get_sdf_from_mesh(mesh, resolution):
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    bounds = get_padded_bounds()
    mins, maxs = bounds[0], bounds[1]

    # Create 3D grid
    x = np.linspace(mins[0], maxs[0], resolution)
    y = np.linspace(mins[1], maxs[1], resolution)
    z = np.linspace(mins[2], maxs[2], resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # Compute signed distances
    sdf_tensor = scene.compute_signed_distance(o3d.core.Tensor(grid_coords, dtype=o3d.core.Dtype.Float32))
    sdf_grid = sdf_tensor.cpu().numpy().reshape((resolution, resolution, resolution))

    return sdf_grid, bounds

def get_mesh_from_sdf(sdf_grid, padded_bounds):
    verts, faces, normals, _ = measure.marching_cubes(
        sdf_grid, level=0.0, spacing=(
            (padded_bounds[1][0] - padded_bounds[0][0]) / (sdf_grid.shape[0] - 1),
            (padded_bounds[1][1] - padded_bounds[0][1]) / (sdf_grid.shape[1] - 1),
            (padded_bounds[1][2] - padded_bounds[0][2]) / (sdf_grid.shape[2] - 1)
        )
    )
    verts += padded_bounds[0]

    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def get_voxel_from_mesh(mesh, resolution):
    
    voxelized = mesh.voxelized(pitch=2.0 / resolution)

    # make dense cube of fixed resolution
    voxel = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    filled = voxelized.matrix.astype(np.uint8)

    # place filled voxels into the fixed voxel
    offset = (np.array(voxel.shape) - np.array(filled.shape)) // 2
    voxel[
        offset[0]:offset[0]+filled.shape[0],
        offset[1]:offset[1]+filled.shape[1],
        offset[2]:offset[2]+filled.shape[2]
    ] = filled

    return voxel

def get_mesh_from_voxel(voxel):
    """
    Run marching cubes on voxel grid to produce a mesh.
    """
    resolution = voxel.shape[0]
    voxel_size = 2 / resolution
    vgrid = trimesh.voxel.VoxelGrid(
        encoding=voxel.astype(bool),
        transform=np.eye(4) 
    )
    mesh = vgrid.as_boxes()
    mesh.apply_scale(voxel_size)
    mesh.apply_translation([-1, -1, -1])
    return mesh

def get_normals(mesh, points):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    points = np.array(points)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / (norms + 1e-10)
    
    vertex_normals = np.zeros_like(vertices)
    
    for i, face in enumerate(faces):
        face_normal = face_normals[i]
        vertex_normals[face[0]] += face_normal
        vertex_normals[face[1]] += face_normal
        vertex_normals[face[2]] += face_normal
    
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norms + 1e-10)
    
    kdtree = cKDTree(vertices)
    _, closest_vertex_idx = kdtree.query(points)
    
    return vertex_normals[closest_vertex_idx]

def get_mesh_from_pc(vertices, normals = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # Estimate normals
    if normals is None:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,  # neighborhood radius
            max_nn=30   # max neighbors
            ))
        pcd.orient_normals_consistent_tangent_plane(30)
    
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    mesh.compute_vertex_normals()
    return mesh

def triangle_box_intersection(triangle, box_mins, box_maxs):
    n_boxes = len(box_mins)
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    box_centers = (box_mins + box_maxs) * 0.5  # (N, 3)
    box_halves = (box_maxs - box_mins) * 0.5   # (N, 3)
    # Translate triangle vertices to box centers
    v0_translated = v0[None, :] - box_centers  # (N, 3)
    v1_translated = v1[None, :] - box_centers  # (N, 3)
    v2_translated = v2[None, :] - box_centers  # (N, 3)
    # Triangle edges
    e0 = v1 - v0  # (3,)
    e1 = v2 - v1  # (3,)
    e2 = v0 - v2  # (3,)
    
    # Test 1: Box axes (x, y, z)
    for axis in range(3):
        tri_coords = np.stack([v0_translated[:, axis], 
                              v1_translated[:, axis], 
                              v2_translated[:, axis]], axis=1)  # (N, 3)
        tri_mins = tri_coords.min(axis=1)  # (N,)
        tri_maxs = tri_coords.max(axis=1)  # (N,)
        
        # If triangle projection doesn't overlap box projection on this axis
        no_overlap = (tri_maxs < -box_halves[:, axis]) | (tri_mins > box_halves[:, axis])
        if np.any(no_overlap):
            intersections = np.ones(n_boxes, dtype=bool)
            intersections[no_overlap] = False
            if not np.any(intersections):
                return np.zeros(n_boxes, dtype=bool)
            
    # Test 2: Triangle normal
    normal = np.cross(e0, e1)  # (3,)
    if np.linalg.norm(normal) > 1e-10:
        normal = normal / np.linalg.norm(normal)
        d = np.dot(v0_translated, normal)  # (N,)
        r = np.sum(box_halves * np.abs(normal)[None, :], axis=1)  # (N,)
        
        no_overlap = np.abs(d) > r
        if np.any(no_overlap):
            intersections = np.ones(n_boxes, dtype=bool)
            intersections[no_overlap] = False
            if not np.any(intersections):
                return np.zeros(n_boxes, dtype=bool)
    
    # Test 3: Cross products (simplified - most important ones)
    edges = [e0, e1, e2]
    box_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    for edge in edges:
        for box_axis in box_axes:
            axis = np.cross(edge, box_axis)
            if np.linalg.norm(axis) < 1e-10:
                continue
            
            # Project triangle vertices onto axis
            p0 = np.dot(v0_translated, axis)  # (N,)
            p1 = np.dot(v1_translated, axis)  # (N,)
            p2 = np.dot(v2_translated, axis)  # (N,)
            
            tri_mins = np.minimum(np.minimum(p0, p1), p2)  # (N,)
            tri_maxs = np.maximum(np.maximum(p0, p1), p2)  # (N,)
            
            # Project boxes onto axis
            r = np.sum(box_halves * np.abs(axis)[None, :], axis=1)  # (N,)
            
            no_overlap = (tri_maxs < -r) | (tri_mins > r)
            if np.any(no_overlap):
                intersections = np.ones(n_boxes, dtype=bool)
                intersections[no_overlap] = False
                if not np.any(intersections):
                    return np.zeros(n_boxes, dtype=bool)
    
    return np.ones(n_boxes, dtype=bool)

def get_uniform_vertices_normals(mesh, resolution):
    triangles = mesh.triangles
    voxel_size = 2.0 / resolution
    surface_voxels = set()
    tri_mins = np.floor((triangles.min(axis=1) + 1.0) / voxel_size).astype(int)
    tri_maxs = np.ceil((triangles.max(axis=1) + 1.0) / voxel_size).astype(int)
    tri_mins = np.maximum(tri_mins, 0)
    tri_maxs = np.minimum(tri_maxs, resolution - 1)
    
    for tri_idx, (tri_min, tri_max) in enumerate(zip(tri_mins, tri_maxs)):
        triangle = triangles[tri_idx]
        ranges = [np.arange(tri_min[i], tri_max[i] + 1) for i in range(3)]
        if any(len(r) == 0 for r in ranges):
            continue
        ix, iy, iz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')
        candidate_voxels = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)
        voxel_mins = candidate_voxels * voxel_size - 1.0
        voxel_maxs = voxel_mins + voxel_size
        intersecting = triangle_box_intersection(triangle, voxel_mins, voxel_maxs)
        intersecting_voxels = candidate_voxels[intersecting]
        surface_voxels.update(map(tuple, intersecting_voxels))
    
    voxels = np.array(list(surface_voxels)) if surface_voxels else np.empty((0, 3), dtype=int)
    world_points = (voxels + 0.5) * voxel_size - 1.0
    normals = get_normals(mesh, world_points)
    return world_points, normals

def get_sharp_vertices_normals(mesh, sharpness_threshold):
    adj_edges = mesh.face_adjacency_edges     # (M, 2) vertex indices per edge
    adj_faces = mesh.face_adjacency           # (M, 2) face indices
    face_normals = mesh.face_normals
    
    sharp_edges = []
    for (f1, f2), (v1, v2) in zip(adj_faces, adj_edges):
        n1, n2 = face_normals[f1], face_normals[f2]
        angle = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
        if angle > sharpness_threshold:
            sharp_edges.append([v1, v2])

    sharp_edges = np.array(sharp_edges)
    if sharp_edges.shape[0] == 0:
        print("⚠️ No sharp edges found!")
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Unique sharp vertices
    vertices_index = np.unique(sharp_edges.flatten())
    vertices = mesh.vertices[vertices_index]
    normals = get_normals(mesh, vertices)
    return vertices, normals

if __name__ == "__main__":
    input_dir = "data/voromesh_processed"
    output_dir = "data/voromesh_reconstructed"
    sdf_data_dir = "data/voromesh_data/sdf"
    voxel_data_dir = "data/voromesh_data/voxel"
    pc_data_dir = "data/voromesh_data/pc"
    output_dir_voxel = "data/voromesh_reconstructed_voxel"
    output_dir_pc = "data/voromesh_reconstructed_pc"
    output_dir_voxel = None # coment it if you dont need a dir
    output_dir_pc = None # coment it if you dont need a dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sdf_data_dir, exist_ok=True)
    os.makedirs(voxel_data_dir, exist_ok=True)
    os.makedirs(pc_data_dir, exist_ok=True)
    
    if output_dir_voxel is not None:
        os.makedirs(output_dir_voxel, exist_ok=True)
    if output_dir_pc is not None:
        os.makedirs(output_dir_pc, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith((".obj", ".glb")):
            continue
        try:
            ## load data
            data_file_name = file.split(".")[0] + ".pt"
            obj = trimesh.load(os.path.join(input_dir, file))
            obj_trimesh, obj_o3d = get_mesh(obj)
            
            # sdf
            sdf, padded_bounds = get_sdf_from_mesh(obj_o3d, config.data.resolution)
            mesh = get_mesh_from_sdf(sdf, padded_bounds)
            o3d.io.write_triangle_mesh(os.path.join(output_dir, file), mesh)
            sdf = torch.Tensor(sdf)
            sdf = sdf.unsqueeze(0)
            torch.save(sdf, os.path.join(sdf_data_dir, data_file_name))
            
            # pc
            pc_uniform, normals_uniform = get_uniform_vertices_normals(obj_trimesh, config.data.resolution)
            pc_sharp, normals_sharp = get_sharp_vertices_normals(obj_trimesh, config.data.sharp_edges_threshold)
            pc = np.vstack([pc_uniform, pc_sharp])
            normals = np.vstack([normals_uniform, normals_sharp])
            
            if output_dir_pc is not None:
                mesh = get_mesh_from_pc(pc, normals)
                o3d.io.write_triangle_mesh(os.path.join(output_dir_pc, file), mesh)
                
            pc = torch.Tensor(pc)
            torch.save(pc, os.path.join(pc_data_dir, data_file_name))
            
            ## voxel
            # voxel = get_voxel_from_mesh(obj_trimesh, config.data.resolution)
            
            # if output_dir_voxel is not None:
            #     mesh = get_mesh_from_voxel(voxel)
            #     mesh.export(os.path.join(output_dir_voxel, file))
            
            # voxel = torch.Tensor(voxel)
            # voxel = voxel.unsqueeze(0)
            # torch.save(voxel, os.path.join(voxel_data_dir, data_file_name))
        except Exception as e:
            print(f"Error processing {file}: {e}")
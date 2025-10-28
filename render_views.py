import os
import glob
import numpy as np
import trimesh
import pyrender
import open3d as o3d
import cv2
import json

# -------------------- Parameters --------------------
assets_base = "assets"
output_base = "renders"
num_views = 10
elevation = 15  # degrees
width, height = 800, 600
yfov = np.pi / 3.0
bg_color = (1.0, 1.0, 1.0, 1.0)
light_intensity = 3.0

# -------------------- Look-at function --------------------
def look_at(eye, center, up=[0,1,0]):
    eye = np.array(eye, dtype=float)
    center = np.array(center, dtype=float)
    up = np.array(up, dtype=float)
    z = eye - center
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    mat = np.eye(4, dtype=float)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat

# -------------------- Process single object --------------------
def process_object(obj_path, output_dir):
    print(f"\nProcessing: {obj_path}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "silhouettes"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plys"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    # Load mesh
    mesh = trimesh.load(obj_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        meshes = list(mesh.geometry.values())
    else:
        combined = mesh
        meshes = [mesh]

    # Center at origin
    center_offset = combined.bounds.mean(axis=0)
    combined.apply_translation(-center_offset)
    for m in meshes:
        m.apply_translation(-center_offset)
    
    # Grey color
    for m in meshes:
        if hasattr(m.visual, 'vertex_colors'):
            m.visual.vertex_colors = [128, 128, 128, 255]
        else:
            m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=[128, 128, 128, 255])
    
    # Camera distance
    radius = float(np.linalg.norm(combined.extents) / 2.0)
    cam_radius = radius * 2.0
    center = np.array([0.0, 0.0, 0.0])

    # Setup renderer
    scene = pyrender.Scene(bg_color=bg_color)
    for m in meshes:
        pm = pyrender.Mesh.from_trimesh(m, smooth=True)
        scene.add(pm, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    renderer = pyrender.OffscreenRenderer(width, height)

    # Render views
    for view_idx in range(num_views):
        azimuth = view_idx * (360 / num_views)
        rad_az = np.deg2rad(azimuth)
        rad_el = np.deg2rad(elevation)

        cam_x = cam_radius * np.cos(rad_el) * np.sin(rad_az)
        cam_y = cam_radius * np.sin(rad_el)
        cam_z = cam_radius * np.cos(rad_el) * np.cos(rad_az)

        cam_pose = look_at([cam_x, cam_y, cam_z], center)

        # Update camera/light
        for node in list(scene.nodes):
            if node.camera or node.light:
                scene.remove_node(node)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        # Render
        color, depth = renderer.render(scene)

        # Save image
        img_name = f"view_{view_idx:03d}.png"
        img_path = os.path.join(output_dir, "images", img_name)
        save_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) if color.shape[2] == 3 else cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(img_path, save_color)

        # Save silhouette
        sil = (depth > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, "silhouettes", f"view_{view_idx:03d}_silhouette.png"), sil)

        # Generate PLY from depth
        H, W = depth.shape
        fx = fy = W / (2 * np.tan(yfov/2))
        cx, cy = W/2, H/2
        
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        mask = depth > 0
        
        xs_valid = xs[mask]
        ys_valid = ys[mask]
        zs_valid = depth[mask]
        
        # Camera space coordinates
        X_cam = (xs_valid - cx) * zs_valid / fx
        Y_cam = -(ys_valid - cy) * zs_valid / fy
        Z_cam = -zs_valid
        
        # Transform to world space
        points_cam = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(X_cam)], axis=-1)
        points_world = (points_cam @ cam_pose.T)[:, :3]
        
        # Get colors
        colors = color[mask] / 255.0
        
        # Save PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.io.write_point_cloud(os.path.join(output_dir, "plys", f"view_{view_idx:03d}.ply"), pcd)

        # Save metadata
        meta = {
            "image": img_name,
            "silhouette": f"view_{view_idx:03d}_silhouette.png",
            "ply": f"view_{view_idx:03d}.ply",
            "camera_pose": cam_pose.tolist(),
            "cam_location": [float(cam_x), float(cam_y), float(cam_z)],
            "azimuth": float(azimuth),
            "elevation": elevation
        }
        with open(os.path.join(output_dir, "meta", f"view_{view_idx:03d}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  View {view_idx}/{num_views} done (azimuth={azimuth:.1f}°)")

    renderer.delete()
    print(f"✓ Completed {obj_path}")

# -------------------- Main --------------------
if __name__ == "__main__":
    obj_files = glob.glob(os.path.join(assets_base, "*/mesh.obj"))
    
    if not obj_files:
        print(f"No mesh.obj files found in {assets_base}/*/")
        exit(1)
    
    print(f"Found {len(obj_files)} objects to process")
    
    for obj_file in obj_files:
        obj_name = os.path.basename(os.path.dirname(obj_file))
        output_dir = os.path.join(output_base, obj_name)
        
        try:
            process_object(obj_file, output_dir)
        except Exception as e:
            print(f"✗ Error processing {obj_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ ALL DONE! Processed {len(obj_files)} objects")
    print(f"Output: {output_base}/[object_name]/")
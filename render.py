import os
import json
import shutil
import numpy as np
import trimesh
import pyrender
import cv2

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

def render_and_save_views(mesh_path,
                          out_dir,
                          num_views=8,
                          width=800,
                          height=600,
                          yfov=np.pi/3.0,
                          bg_color=(1.0, 1.0, 1.0, 1.0),
                          light_intensity=3.0,
                          save_depth=True,
                          save_silhouette=True):
    """Render multiple views of mesh_path and save images + per-view metadata JSON into out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(mesh_path, process=True)
    # Handle Scene vs single Trimesh
    if isinstance(mesh, trimesh.Scene):
        # convert each geometry to pyrender.Mesh and combine for sampling bounds
        meshes = [pyrender.Mesh.from_trimesh(m, smooth=True) for m in mesh.geometry.values()]
        combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    else:
        meshes = [pyrender.Mesh.from_trimesh(mesh, smooth=True)]
        combined = mesh

    center = combined.bounds.mean(axis=0)
    radius = float(np.linalg.norm(combined.extents) / 2.0)

    scene = pyrender.Scene(bg_color=bg_color)
    # add mesh(es) once
    for m in meshes:
        scene.add(m, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=float(yfov))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=float(light_intensity))

    renderer = pyrender.OffscreenRenderer(int(width), int(height))

    saved = []
    angles = np.linspace(0.0, 360.0, num_views, endpoint=False)
    for i, angle in enumerate(angles):
        rad = np.deg2rad(angle)
        cam_x = float(center[0] + radius * 2.0 * np.sin(rad))
        cam_y = float(center[1] + radius * 0.5)
        cam_z = float(center[2] + radius * 2.0 * np.cos(rad))
        camera_pose = look_at([cam_x, cam_y, cam_z], center)

        # remove previous camera & light nodes (if any)
        for node in list(scene.nodes):
            if node.camera is not None or node.light is not None:
                scene.remove_node(node)

        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)

        # render
        color, depth = renderer.render(scene)  # color: HxWx3 or HxWx4, depth: HxW float32 (0 for background)

        # convert color to BGR/A format for cv2
        if color.shape[2] == 4:
            # RGBA -> BGRA
            save_color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
        else:
            # RGB -> BGR
            save_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        img_name = f"view_{i:02d}.png"
        img_path = os.path.join(out_dir, img_name)
        cv2.imwrite(img_path, save_color)

        # depth
        depth_npy = None
        depth_vis_path = None
        if save_depth:
            depth_npy = f"view_{i:02d}_depth.npy"
            depth_npy_path = os.path.join(out_dir, depth_npy)
            np.save(depth_npy_path, depth.astype(np.float32))

            # normalized 16-bit PNG visualization (use only for viewing)
            mask = depth > 0
            if np.any(mask):
                dvals = depth[mask]
                dmin = float(dvals.min())
                dmax = float(dvals.max())
                if dmax > dmin:
                    depth_norm = (depth - dmin) / (dmax - dmin)
                else:
                    depth_norm = np.zeros_like(depth, dtype=np.float32)
                depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)
                depth_vis_path = os.path.join(out_dir, f"view_{i:02d}_depth_vis.png")
                cv2.imwrite(depth_vis_path, depth_uint16)
            else:
                # no depth -> save an empty black visualization
                depth_vis_path = os.path.join(out_dir, f"view_{i:02d}_depth_vis.png")
                cv2.imwrite(depth_vis_path, np.zeros((height, width), dtype=np.uint8))

        # silhouette (binary mask from depth > 0)
        sil_path = None
        if save_silhouette:
            sil = (depth > 0).astype(np.uint8) * 255
            sil_path = os.path.join(out_dir, f"view_{i:02d}_silhouette.png")
            cv2.imwrite(sil_path, sil)

        # intrinsics
        half_h = 0.5 * float(height)
        fy = half_h / (np.tan(0.5 * float(yfov)) + 1e-12)
        fx = float(fy)
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        K = [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0]
        ]

        meta = {
            "image": img_name,
            "depth_npy": depth_npy,
            "depth_vis": os.path.basename(depth_vis_path) if depth_vis_path else None,
            "silhouette": os.path.basename(sil_path) if sil_path else None,
            "width": int(width),
            "height": int(height),
            "yfov": float(yfov),
            "K": K,
            "camera_pose": camera_pose.tolist(),        # camera -> world (4x4)
            "world2cam": np.linalg.inv(camera_pose).tolist(),  # world -> camera
            "cam_location": [cam_x, cam_y, cam_z],
            "radius": float(radius),
            "angle_degrees": float(angle)
        }

        meta_path = os.path.join(out_dir, f"view_{i:02d}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        saved.append({
            "img": img_path,
            "meta": meta_path,
            "depth_npy": os.path.join(out_dir, depth_npy) if depth_npy else None,
            "depth_vis": depth_vis_path,
            "silhouette": sil_path
        })

        print(f"Saved view {i:02d} -> {img_path}, meta -> {meta_path}")

    renderer.delete()
    return saved


# -------------------------
# Example integration with your dataset layout
# -------------------------
if __name__ == "__main__":
    data_dir = "data/model"   # same as your layout
    output_root = "data/renders_with_camera"  # new output folder

    os.makedirs(output_root, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        out_class_path = os.path.join(output_root, class_name)
        os.makedirs(out_class_path, exist_ok=True)

        for subclass_name in os.listdir(class_path):
            subclass_path = os.path.join(class_path, subclass_name)
            if not os.path.isdir(subclass_path):
                continue

            obj_path = os.path.join(subclass_path, "model.obj")
            if not os.path.exists(obj_path):
                print(f"Skipping {subclass_path}, no model.obj found")
                continue

            out_subclass_path = os.path.join(out_class_path, subclass_name)
            os.makedirs(out_subclass_path, exist_ok=True)

            # copy original OBJ for reference
            shutil.copy(obj_path, os.path.join(out_subclass_path, "model.obj"))

            print(f"Rendering {class_name}/{subclass_name} -> {out_subclass_path}")
            try:
                render_and_save_views(obj_path, out_subclass_path, num_views=8, width=800, height=600)
            except Exception as e:
                print(f"ERROR rendering {obj_path}: {e}")

    print("All done.")

import os
import shutil
import numpy as np
import trimesh
import pyrender
import cv2

# --- Reuse the look_at and render_multiple_views functions ---
def look_at(eye, center, up=[0,1,0]):
    eye = np.array(eye)
    center = np.array(center)
    up = np.array(up)
    z = eye - center
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat

def render_multiple_views(mesh_path, num_views=8, width=800, height=600, bg_color=(1,1,1,1)):
    mesh = trimesh.load(mesh_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        meshes = [pyrender.Mesh.from_trimesh(m, smooth=True) for m in mesh.geometry.values()]
        combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    else:
        meshes = [pyrender.Mesh.from_trimesh(mesh, smooth=True)]
        combined = mesh

    center = combined.bounds.mean(axis=0)
    radius = np.linalg.norm(combined.extents)/2

    scene = pyrender.Scene(bg_color=bg_color)
    for m in meshes:
        scene.add(m, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    r = pyrender.OffscreenRenderer(width, height)
    images = []

    for angle in np.linspace(0, 360, num_views, endpoint=False):
        rad = np.deg2rad(angle)
        cam_x = center[0] + radius * 2 * np.sin(rad)
        cam_y = center[1] + radius * 0.5
        cam_z = center[2] + radius * 2 * np.cos(rad)
        camera_pose = look_at([cam_x, cam_y, cam_z], center)

        # Remove previous camera & light nodes
        for node in list(scene.nodes):
            if node.camera is not None or node.light is not None:
                scene.remove_node(node)

        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)

        img, _ = r.render(scene)
        images.append(cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

    r.delete()
    return images

# --- Main pipeline ---
data_dir = "data/model"
output_dir = "renders"

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    out_class_path = os.path.join(output_dir, class_name)
    os.makedirs(out_class_path, exist_ok=True)

    for subclass_name in os.listdir(class_path):
        subclass_path = os.path.join(class_path, subclass_name)
        if not os.path.isdir(subclass_path):
            continue

        obj_path = os.path.join(subclass_path, "model.obj")
        if not os.path.exists(obj_path):
            print(f"Skipping {subclass_path}, no model.obj found")
            continue

        # Make subclass folder in output
        out_subclass_path = os.path.join(out_class_path, subclass_name)
        os.makedirs(out_subclass_path, exist_ok=True)

        # Copy OBJ
        shutil.copy(obj_path, os.path.join(out_subclass_path, "model.obj"))

        # Render 8 views
        print(f"Rendering {class_name}/{subclass_name} ...")
        views = render_multiple_views(obj_path, num_views=8)
        for i, img in enumerate(views):
            cv2.imwrite(os.path.join(out_subclass_path, f"view_{i}.png"), img)

print("Done!")

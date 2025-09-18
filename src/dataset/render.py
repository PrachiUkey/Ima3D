import numpy as np
import trimesh
import pyrender
import cv2

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

def render_multiple_views(mesh_path, num_views=5, width=800, height=600, bg_color=(1,1,1,1)):
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

if __name__ == "__main__":
    imgs = render_multiple_views("data/model/bed/bed_0/model.obj", num_views=8)
    for i, im in enumerate(imgs):
        cv2.imwrite(f"view_{i}.png", im)


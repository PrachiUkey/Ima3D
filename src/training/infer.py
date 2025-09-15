# src/training/infer_debug.py
import os
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from plyfile import PlyData, PlyElement
import trimesh

from src.models.image_encoder import ImageEncoder
from src.models.mesh_encoder import MeshEncoder

# -------------------------
# Preprocess image (toggle ImageNet norm)
# -------------------------
def preprocess_image(image_path, img_size=(256,256), use_imagenet_norm=True, device='cpu'):
    tf = [T.Resize(img_size), T.ToTensor()]
    if use_imagenet_norm:
        tf.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]))
    transform = T.Compose(tf)
    img = Image.open(image_path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return t

# -------------------------
# Save Nx3 points to PLY (ascii)
# -------------------------
def save_points_as_ply(points_np, out_path):
    # points_np: (N,3) numpy float32
    vertex = np.array([tuple(p) for p in points_np],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(out_path)
    print(f"Saved: {out_path}")

# -------------------------
# Load/prepare mesh point cloud robustly
# -------------------------
def sample_points_from_obj(obj_path, num_points=10000, sampling="surface"):
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        # concatenate all geometries
        if len(mesh.geometry) == 0:
            raise ValueError(f"Empty scene: {obj_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"Loaded empty mesh: {obj_path}")

    if sampling == "surface":
        # trimesh.sample can return fewer points if model tiny; handle that
        pts = mesh.sample(num_points)
        if pts.shape[0] < num_points:
            # fallback: pad by repeating
            reps = int(np.ceil(num_points / max(1, pts.shape[0])))
            pts = np.tile(pts, (reps, 1))[:num_points]
    else:
        verts = mesh.vertices
        if verts.shape[0] == 0:
            raise ValueError(f"No vertices: {obj_path}")
        idx = np.random.choice(len(verts), size=min(num_points, len(verts)), replace=True)
        pts = verts[idx]

    return np.asarray(pts, dtype=np.float32)

# -------------------------
# Main debug inference
# -------------------------
def run_debug_inference(
    image_path,
    checkpoint_path,
    mesh_dir="data/model",
    num_points=10000,
    latent_dim=256,
    device=None,
    top_k=5,
    restrict_to_same_class=True,
    use_imagenet_norm=True,
    sampling="surface"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ensure forward-slash paths on Windows
    image_path = str(image_path).replace("\\", "/")
    checkpoint_path = str(checkpoint_path).replace("\\", "/")
    mesh_dir = str(mesh_dir).replace("\\", "/")

    # load models + checkpoint
    image_encoder = ImageEncoder(embed_dim=latent_dim).to(device)
    mesh_encoder = MeshEncoder(embed_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    image_encoder.load_state_dict(ckpt['image_encoder'])
    mesh_encoder.load_state_dict(ckpt['mesh_encoder'])
    image_encoder.eval(); mesh_encoder.eval()

    # preprocess image
    img_tensor = preprocess_image(image_path, img_size=(256,256),
                                  use_imagenet_norm=use_imagenet_norm, device=device)
    with torch.no_grad():
        img_embed = image_encoder(img_tensor)  # (1, D)

    # extract image class from path (data/img/<class>/<file>)
    image_class = None
    parts = image_path.replace("\\","/").split("/")
    if len(parts) >= 3 and parts[-2] != "":
        image_class = parts[-2].lower()
    print("Image class (from path):", image_class)

    # build list of mesh paths and classes (sorted for determinism)
    mesh_candidates = []
    for cls in sorted(os.listdir(mesh_dir)):
        cls_folder = os.path.join(mesh_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        for sub in sorted(os.listdir(cls_folder)):
            obj_path = os.path.join(cls_folder, sub, "model.obj")
            if os.path.exists(obj_path):
                mesh_candidates.append((obj_path.replace("\\","/"), cls.lower()))

    if len(mesh_candidates) == 0:
        raise RuntimeError(f"No mesh .obj found under {mesh_dir}")

    print(f"Found {len(mesh_candidates)} mesh candidates across {len(set([c for _,c in mesh_candidates]))} classes")

    # optionally restrict to only meshes of same class
    if restrict_to_same_class and image_class is not None:
        filtered = [(p,c) for (p,c) in mesh_candidates if c == image_class]
        if len(filtered) == 0:
            print(f"Warning: no meshes found for image class '{image_class}'. Will search across all classes.")
        else:
            mesh_candidates = filtered
            print(f"Restricted search to class '{image_class}', {len(mesh_candidates)} candidates")

    # iterate and compute similarity, keep top_k
    top_list = []  # list of tuples (sim, class, path, points_np)
    for idx, (obj_path, cls) in enumerate(mesh_candidates):
        try:
            pts_np = sample_points_from_obj(obj_path, num_points=num_points, sampling=sampling)  # (N,3) numpy
            pts_t = torch.from_numpy(pts_np).unsqueeze(0).to(device)  # (1,N,3)
            with torch.no_grad():
                mesh_embed = mesh_encoder(pts_t)  # (1,D)
                sim = float(F.cosine_similarity(img_embed, mesh_embed).item())
        except Exception as e:
            print(f"Skipped {obj_path} due to error: {e}")
            traceback.print_exc()
            continue

        # store in top list, maintain length <= top_k
        top_list.append((sim, cls, obj_path, pts_np))
        # keep only top_k by sim
        top_list = sorted(top_list, key=lambda x: x[0], reverse=True)[:top_k]

        # optional progress print
        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1}/{len(mesh_candidates)} meshes; current best sim: {top_list[0][0]:.4f}")

    if len(top_list) == 0:
        raise RuntimeError("No valid mesh candidates processed.")

    # print top_k summary
    print("\nTop matches:")
    for rank, (sim, cls, obj_path, pts_np) in enumerate(top_list, start=1):
        print(f"#{rank}: sim={sim:.4f} class={cls} path={obj_path}")

    # save top_k PLYs
    os.makedirs("inference_debug_ply", exist_ok=True)
    for rank, (sim, cls, obj_path, pts_np) in enumerate(top_list, start=1):
        out_name = f"inference_debug_ply/top{rank}_sim{sim:.4f}_{cls}.ply"
        save_points_as_ply(pts_np, out_name)

    print("Saved top-k PLYs in ./inference_debug_ply")
    return top_list

# -------------------------
# Run (edit these paths/flags)
# -------------------------
if __name__ == "__main__":
    # Edit these variables directly
    image_path = os.path.join("data", "img", "chair", "0071.png")  # set image you want
    checkpoint_path = os.path.join("checkpoints", "embedding_epoch11.pth")  # correct checkpoint
    mesh_dir = os.path.join("data", "model")
    device = None  # None -> auto pick
    num_points = 10000      # set same as training ideally (or smaller for speed)
    latent_dim = 256
    top_k = 5
    restrict_to_same_class = True   # try True first (faster, more logical)
    use_imagenet_norm = True        # toggle if training used unnormalized images
    sampling = "surface"            # same as training

    run_debug_inference(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        mesh_dir=mesh_dir,
        num_points=num_points,
        latent_dim=latent_dim,
        device=device,
        top_k=top_k,
        restrict_to_same_class=restrict_to_same_class,
        use_imagenet_norm=use_imagenet_norm,
        sampling=sampling
    )

import os
import json
import shutil
import random
from pathlib import Path

# -------------------- Configuration --------------------
renders_dir = "renders"  # Where rendered data is
output_dir = "dataset"   # Where processed dataset goes
train_ratio = 0.8        # 80% train, 20% test
random_seed = 42

random.seed(random_seed)

# -------------------- Prepare Dataset --------------------
def prepare_dataset():
    print("Preparing dataset...")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    for split in [train_dir, test_dir]:
        os.makedirs(os.path.join(split, "images"), exist_ok=True)
        os.makedirs(os.path.join(split, "plys"), exist_ok=True)
        os.makedirs(os.path.join(split, "silhouettes"), exist_ok=True)
    
    # Collect all samples
    all_samples = []
    object_dirs = [d for d in Path(renders_dir).iterdir() if d.is_dir()]
    
    print(f"Found {len(object_dirs)} objects")
    
    for obj_dir in object_dirs:
        obj_name = obj_dir.name
        images_dir = obj_dir / "images"
        plys_dir = obj_dir / "plys"
        sil_dir = obj_dir / "silhouettes"
        meta_dir = obj_dir / "meta"
        
        if not images_dir.exists():
            continue
        
        # Get all views
        image_files = sorted(images_dir.glob("view_*.png"))
        
        for img_file in image_files:
            view_id = img_file.stem  # e.g., "view_000"
            
            ply_file = plys_dir / f"{view_id}.ply"
            sil_file = sil_dir / f"{view_id}_silhouette.png"
            meta_file = meta_dir / f"{view_id}_meta.json"
            
            if ply_file.exists() and sil_file.exists():
                all_samples.append({
                    "object": obj_name,
                    "view_id": view_id,
                    "image": str(img_file),
                    "ply": str(ply_file),
                    "silhouette": str(sil_file),
                    "meta": str(meta_file)
                })
    
    print(f"Total samples: {len(all_samples)}")
    
    # Shuffle and split by object (not by view to avoid data leakage)
    objects = {}
    for sample in all_samples:
        obj = sample["object"]
        if obj not in objects:
            objects[obj] = []
        objects[obj].append(sample)
    
    object_names = list(objects.keys())
    random.shuffle(object_names)
    
    split_idx = int(len(object_names) * train_ratio)
    train_objects = object_names[:split_idx]
    test_objects = object_names[split_idx:]
    
    train_samples = []
    test_samples = []
    
    for obj in train_objects:
        train_samples.extend(objects[obj])
    for obj in test_objects:
        test_samples.extend(objects[obj])
    
    print(f"Train objects: {len(train_objects)}, samples: {len(train_samples)}")
    print(f"Test objects: {len(test_objects)}, samples: {len(test_samples)}")
    
    # Copy files and create manifests
    def copy_samples(samples, split_dir, split_name):
        manifest = []
        
        for idx, sample in enumerate(samples):
            # Create unique filenames
            img_dst = os.path.join(split_dir, "images", f"{idx:05d}.png")
            ply_dst = os.path.join(split_dir, "plys", f"{idx:05d}.ply")
            sil_dst = os.path.join(split_dir, "silhouettes", f"{idx:05d}.png")
            
            # Copy files
            shutil.copy2(sample["image"], img_dst)
            shutil.copy2(sample["ply"], ply_dst)
            shutil.copy2(sample["silhouette"], sil_dst)
            
            # Add to manifest
            manifest.append({
                "id": idx,
                "object": sample["object"],
                "view_id": sample["view_id"],
                "image": f"images/{idx:05d}.png",
                "ply": f"plys/{idx:05d}.ply",
                "silhouette": f"silhouettes/{idx:05d}.png"
            })
        
        # Save manifest
        manifest_path = os.path.join(split_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ {split_name} manifest saved: {len(manifest)} samples")
    
    copy_samples(train_samples, train_dir, "Train")
    copy_samples(test_samples, test_dir, "Test")
    
    # Create dataset info
    info = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_objects": train_objects,
        "test_objects": test_objects,
        "train_ratio": train_ratio,
        "random_seed": random_seed
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"Output directory: {output_dir}/")
    print(f"  - train/: {len(train_samples)} samples")
    print(f"  - test/: {len(test_samples)} samples")

if __name__ == "__main__":
    prepare_dataset()
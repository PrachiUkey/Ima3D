import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256,256)):
        self.samples = []
        self.img_size = img_size

        # loop over classes
        for cls in os.listdir(img_dir):
            cls_img_dir = os.path.join(img_dir, cls)
            cls_mask_dir = os.path.join(mask_dir, cls)
            if not os.path.isdir(cls_img_dir) or not os.path.isdir(cls_mask_dir):
                continue

            for img_fname in os.listdir(cls_img_dir):
                img_path = os.path.join(cls_img_dir, img_fname)
                # try to find matching mask with any extension
                base_name = os.path.splitext(img_fname)[0]
                mask_path = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = os.path.join(cls_mask_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break

                if mask_path is not None:
                    self.samples.append({
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "class": cls.lower()
                    })
                else:
                    print(f"Warning: Mask not found for {img_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["img_path"]).convert("RGB")
        img = img.resize(self.img_size)

        # Load mask
        mask = Image.open(sample["mask_path"]).convert("L")
        mask = mask.resize(self.img_size)

        # Convert to tensor
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1) / 255.0
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.0

        return img, mask, sample["class"]

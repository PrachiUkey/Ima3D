import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, classes, img_size=(256,256), transform=None):
        """
        Dataset for images and masks
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.classes = classes
        self.img_size = img_size
        self.transform = transform

        self.samples = []
        for cls in classes:
            img_cls_dir = os.path.join(img_dir, cls)
            mask_cls_dir = os.path.join(mask_dir, cls)
            img_files = sorted(os.listdir(img_cls_dir))
            for img_file in img_files:
                img_id = os.path.splitext(img_file)[0]
                mask_file = os.path.join(mask_cls_dir, f"{img_id}.png")
                self.samples.append({
                    "img": os.path.join(img_cls_dir, img_file),
                    "mask": mask_file,
                    "class": cls
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load image & mask
        img = Image.open(sample["img"]).convert("RGB").resize(self.img_size)
        mask = Image.open(sample["mask"]).convert("L").resize(self.img_size)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        else:
            img = np.array(img).astype(np.float32)/255.0
            mask = np.array(mask).astype(np.float32)/255.0
            img = torch.from_numpy(img).permute(2,0,1)  # C,H,W
            mask = torch.from_numpy(mask).unsqueeze(0)  # 1,H,W
        
        return img, mask, sample["class"]

# Example usage
if __name__ == "__main__":
    dataset = ImageDataset(
        img_dir="data/img",
        mask_dir="data/mask",
        classes=["bed","chair"]
    )
    img, mask, cls = dataset[0]
    print(img.shape, mask.shape, cls)

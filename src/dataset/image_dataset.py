import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=(256, 256), subclass_level=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.subclass_level = subclass_level

        # Collect samples
        self.samples = []
        for class_name in os.listdir(img_dir):
            class_path = os.path.join(img_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, fname)

                    if subclass_level:
                        # Use filename (without extension) as subclass label
                        subclass_name = os.path.splitext(fname)[0].lower()
                        label = f"{class_name.lower()}_{subclass_name}"
                    else:
                        label = class_name.lower()

                    self.samples.append({
                        "img": img_path,
                        "mask": None,
                        "class": label
                    })

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img"]).convert("RGB")
        img = self.transform(img)
        return img, sample["mask"], sample["class"]

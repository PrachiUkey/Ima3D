import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with coarse class folders (e.g. bed, chair)
            transform: Torchvision transforms for images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_images()

    def _load_images(self):
        image_paths, labels = [], []
        for coarse_class in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, coarse_class)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(class_path, fname))
                    labels.append(coarse_class)  # only coarse class
        print(f"[ImageDataset] Loaded {len(image_paths)} images from {self.root_dir}")
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]  # coarse label
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

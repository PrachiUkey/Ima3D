import os
import numpy as np
import matplotlib.pyplot as plt
from .image_dataset import ImageDataset  # make sure this points to your dataset file

def visualize_object_focus(dataset, save_dir="visualized_object_focus", top_n=10):
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(top_n, len(dataset))):
        img, mask, cls = dataset[idx]

        # Convert tensors to numpy
        img_np = img.permute(1,2,0).numpy()  # H,W,C
        mask_np = mask.squeeze(0).numpy()   # H,W

        # Create object-focused image: mask applied
        object_focus = img_np * mask_np[..., None]  # multiply mask to each channel

        # Save original object-focused image
        img_save_path = os.path.join(save_dir, f"{cls}_{idx}_object_focus.png")
        plt.imsave(img_save_path, (object_focus * 255).astype('uint8'))

        # Save mask separately
        mask_save_path = os.path.join(save_dir, f"{cls}_{idx}_mask.png")
        mask_rgb = np.stack([mask_np]*3, axis=-1)  # make 3-channel
        plt.imsave(mask_save_path, (mask_rgb * 255).astype('uint8'))

        # Save overlay (for easy inspection)
        overlay = 0.5 * img_np + 0.5 * mask_rgb
        overlay_save_path = os.path.join(save_dir, f"{cls}_{idx}_overlay.png")
        plt.imsave(overlay_save_path, (overlay * 255).astype('uint8'))

        print(f"Saved object-focused visualization for {cls}_{idx}")

if __name__ == "__main__":
    dataset = ImageDataset(
        img_dir="data/img",
        mask_dir="data/mask",
        classes=["bed","chair"]
    )
    visualize_object_focus(dataset, save_dir="outputs/visualized_object_focus", top_n=10)

import os
from PIL import Image
from src.training.train_embedding import train_embedding
from src.training.inference_encoder import infer_nearest_pointcloud, save_pointcloud_as_ply

def inference_callback(image_encoder, mesh_encoder, mesh_dataset, device, epoch=None):
    test_img_dir = "data/img"
    class_folder = os.listdir(test_img_dir)[0]
    test_image_path = os.path.join(test_img_dir, class_folder, os.listdir(os.path.join(test_img_dir, class_folder))[0])

    nearest_pc = infer_nearest_pointcloud(test_image_path, mesh_dataset, image_encoder, mesh_encoder, device=device)

    os.makedirs("inference_ply", exist_ok=True)
    ply_path = os.path.join("inference_ply", f"epoch_{epoch+1}_nearest.ply") if epoch is not None else "inference_ply/nearest.ply"
    save_pointcloud_as_ply(nearest_pc, ply_path)

    os.makedirs("inference_img", exist_ok=True)
    img_save_path = os.path.join("inference_img", f"epoch_{epoch+1}_input.png") if epoch is not None else "inference_img/input.png"
    img = Image.open(test_image_path)
    img.save(img_save_path)

    print(f"Inference done for epoch {epoch+1}. Point cloud shape: {nearest_pc.shape}")

if __name__ == "__main__":
    train_embedding(
        img_dir="data/img",
        mask_dir="data/mask",
        mesh_dir="data/model",
        embed_dim=256,
        batch_size=2,
        epochs=20,
        lr=1e-4,
        device=None,
        inference_callback=inference_callback
    )

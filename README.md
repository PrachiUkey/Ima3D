project_root/
│── configs/
│   └── config.yaml             # All hyperparams, paths, training configs
│
│── data/
│   ├── images/                 # Real-world images (per class)
│   │   ├── bed/
│   │   │   ├── img_001.jpg
│   │   │   └── ...
│   │   └── chair/
│   │       ├── img_001.jpg
│   │       └── ...
│   │
│   ├── masks/                  # Binary masks aligned with images
│   │   ├── bed/
│   │   │   ├── img_001_mask.png
│   │   │   └── ...
│   │   └── chair/
│   │       ├── img_001_mask.png
│   │       └── ...
│   │
│   ├── models/                 # CAD models (.obj) per class
│   │   ├── bed/
│   │   │   ├── bed1.obj
│   │   │   ├── bed2.obj
│   │   │   └── ...
│   │   └── chair/
│   │       ├── chair1.obj
│   │       └── ...
│   │
│   └── splits/                 # Train/test splits
│       ├── train.txt           # List of (image_path, mask_path, class)
│       ├── val.txt
│       └── test.txt
│
│── src/
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── image_dataset.py     # Loads real-world images + masks
│   │   ├── cad_dataset.py       # Loads OBJ → point clouds
│   │   └── retrieval_dataset.py # Paired dataset (image, cad, label)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_encoder.py     # CNN/ViT encoder
│   │   ├── cad_encoder.py       # PointNet/DGCNN encoder
│   │   └── retrieval_model.py   # Combines both encoders
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py            # Triplet, contrastive, etc.
│   │   ├── trainer.py           # Training loop
│   │   └── evaluator.py         # Retrieval evaluation
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── mesh_utils.py        # OBJ → point cloud, normalization
│   │   ├── image_utils.py       # Masking, augmentations
│   │   └── logger.py            # Logging, checkpoints
│   │
│   └── main.py                  # Entry point (train/test/infer)
│
│── outputs/
│   ├── checkpoints/             # Saved models
│   ├── embeddings/              # Precomputed embeddings for CADs
│   └── logs/                    # Training logs
│
│── requirements.txt
│── README.md

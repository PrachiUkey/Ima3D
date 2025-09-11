Step 1: Utilities (low-level functions)

We need helpers to load/prepare raw data before we touch datasets or models.


src/utils/image_utils.py → load image, apply mask, augment

👉 Once we finish these, we can actually see the data in correct format (image tensors, point clouds).

Step 2: Datasets

src/datasets/image_dataset.py → gives (image, class_label)

src/datasets/cad_dataset.py → gives (point_cloud, class_label)

src/datasets/retrieval_dataset.py → pairs them for triplet/contrastive

👉 This makes sure our Dataloader is ready and we can batch the inputs.

Step 3: Models

src/models/image_encoder.py → ResNet / ViT backbone → embedding

src/models/cad_encoder.py → PointNet / DGCNN → embedding

src/models/retrieval_model.py → wraps both for training

Step 4: Training logic

src/training/losses.py → triplet loss, contrastive

src/training/trainer.py → one epoch training loop

src/training/evaluator.py → retrieval metrics

Step 5: Entry point

src/main.py → python main.py --mode train/test

✅ This order ensures you always have something runnable & testable before moving forward.
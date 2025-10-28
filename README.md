# Image-to-PointCloud Reconstruction Pipeline

Complete pipeline for training a transformer-based model to reconstruct 3D point clouds from single images.

## 📁 Project Structure
```
project/
├── assets/                          # Input 3D models
│   ├── bed_0/mesh.obj
│   ├── bookcase_21/mesh.obj
│   └── ...
├── renders/                         # Generated renders (from step 1)
│   ├── bed_0/
│   │   ├── images/
│   │   ├── silhouettes/
│   │   ├── plys/
│   │   └── meta/
│   └── ...
├── dataset/                         # Prepared dataset (from step 2)
│   ├── train/
│   │   ├── images/
│   │   ├── plys/
│   │   ├── silhouettes/
│   │   └── manifest.json
│   ├── test/
│   └── dataset_info.json
├── checkpoints/                     # Trained models
├── logs/                           # TensorBoard logs
├── render_views.py                 # Step 1: Multi-view rendering
├── prepare_data.py                 # Step 2: Train/test split
├── dataset.py                      # PyTorch dataset
├── model.py                        # Model architecture
├── train.py                        # Step 3: Training
├── inference.py                    # Step 4: Inference/evaluation
└── requirements.txt
```

## 🚀 Quick Start

### Step 0: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1: Render Multi-view Data

Generate 10 views per object with images, silhouettes, and point clouds:
```bash
python render_views.py
```

**Output:**
- `renders/[object_name]/images/` - 10 RGB images
- `renders/[object_name]/silhouettes/` - 10 binary masks
- `renders/[object_name]/plys/` - 10 view-based point clouds (ground truth)
- `renders/[object_name]/meta/` - Camera parameters

### Step 2: Prepare Dataset

Split data into train/test sets (80/20 split by object):
```bash
python prepare_data.py
```

**Output:**
- `dataset/train/` - Training samples
- `dataset/test/` - Test samples
- `dataset/dataset_info.json` - Dataset statistics

### Step 3: Train Model

Train the transformer-based reconstruction model:
```bash
python train.py
```

**Training features:**
- ResNet50 image encoder
- Transformer decoder with learnable point queries
- Chamfer Distance loss
- AdamW optimizer with cosine annealing
- TensorBoard logging
- Automatic checkpoint saving

**Monitor training:**
```bash
tensorboard --logdir logs/
```

### Step 4: Inference & Evaluation

**Evaluate on test set:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth --mode evaluate
```

**Predict single image:**
```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --mode single \
                    --image path/to/image.png \
                    --output prediction.ply
```

## 🏗️ Model Architecture
```
Input Image (3, 224, 224)
    ↓
ResNet50 Encoder
    ↓
Feature Maps (512, H', W')
    ↓
Positional Encoding
    ↓
Flatten → (H'×W', 512)
    ↓
Transformer Decoder ← Learnable Point Queries (2048, 512)
    ↓
Point Prediction Head
    ↓
Output Points (2048, 3)
```

**Key Components:**
- **Image Encoder**: Pre-trained ResNet50
- **Point Queries**: 2048 learnable embeddings
- **Transformer**: 6 layers, 8 attention heads
- **Loss**: Bidirectional Chamfer Distance

## 📊 Configuration

Edit parameters in `train.py`:
```python
class Config:
    # Model
    num_points = 2048      # Number of output points
    hidden_dim = 512       # Transformer hidden dimension
    num_layers = 6         # Transformer decoder layers
    num_heads = 8          # Attention heads
    
    # Training
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    
    # Paths
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    checkpoint_dir = "checkpoints"
```

## 📈 Expected Results

- **Chamfer Distance**: Lower is better (typically 0.001 - 0.01)
- **Training time**: ~2-4 hours on single GPU (depends on dataset size)
- **Inference**: ~50ms per image on GPU

## 🔧 Customization

### Change number of views
Edit `render_views.py`:
```python
num_views = 10  # Change to 8, 12, etc.
```

### Adjust point cloud density
Edit `dataset.py`:
```python
num_points = 4096  # Increase for more detail
```

### Fine-tune on specific objects
```python
# In train.py, load pretrained checkpoint
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
```

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `render_views.py` | Multi-view rendering with pyrender |
| `prepare_data.py` | Data splitting and organization |
| `dataset.py` | PyTorch Dataset class for loading |
| `model.py` | Transformer model + Chamfer loss |
| `train.py` | Training loop with checkpointing |
| `inference.py` | Prediction and evaluation |

## 🐛 Troubleshooting

**Issue: CUDA out of memory**
- Reduce `batch_size` in `train.py`
- Reduce `num_points` in dataset

**Issue: Poor reconstruction quality**
- Train longer (increase `num_epochs`)
- Increase `num_views` for more training data
- Adjust learning rate

**Issue: PLY files don't align**
- Check camera coordinate systems
- Verify point cloud normalization in dataset

## 📚 References

- Transformer architecture inspired by DETR
- Point cloud processing based on PointNet++
- Chamfer Distance for shape matching

## ⚡ GPU Acceleration

For faster training, use mixed precision:
```python
# In train.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    pred = model(images)
    loss = criterion(pred, gt)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🎯 Next Steps

1. Add color prediction (RGB values per point)
2. Multi-view fusion (combine multiple input views)
3. Experiment with different backbones (ViT, EfficientNet)
4. Add normal estimation
5. Mesh reconstruction from point clouds

## 📋 Usage Summary
```bash
# Complete workflow
python render_views.py      # Generate 10 views per object
python prepare_data.py      # Split train/test
python train.py             # Train model
tensorboard --logdir logs/  # Monitor training

# Evaluate
python inference.py --checkpoint checkpoints/best_model.pth --mode evaluate

# Predict
python inference.py --checkpoint checkpoints/best_model.pth \
                    --mode single --image test.png --output pred.ply
```

---

**Happy Training! 🚀**
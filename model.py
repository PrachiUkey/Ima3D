import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np  

class Image2PointCloud(nn.Module):
    """
    Coordinate-based transformer model.
    Automatically adapts to any number of points (2048, 4096, 8192, 10240).
    """
    
    def __init__(self, num_points=4096, hidden_dim=320, num_layers=5, num_heads=5):
        super().__init__()
        
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        
        print(f"Initializing model:")
        print(f"  Num points: {num_points}")

        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        
        # Image encoder (ResNet18)
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        
        # Coordinate-based queries (FIXED, not learned)
        self.register_buffer('coordinate_queries', self._create_coordinate_queries())
        
        # Project 3D coordinates to hidden_dim
        self.query_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Point prediction head
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Tanh()
        )
    
    def _create_coordinate_queries(self):
        """Create 3D grid coordinates as queries."""
        grid_size = int(np.ceil(self.num_points ** (1/3)))
        
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        z = torch.linspace(-1, 1, grid_size)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        
        # Exactly num_points
        if coords.shape[0] > self.num_points:
            indices = torch.linspace(0, coords.shape[0]-1, self.num_points).long()
            coords = coords[indices]
        elif coords.shape[0] < self.num_points:
            extra = self.num_points - coords.shape[0]
            random_coords = torch.rand(extra, 3) * 2 - 1
            coords = torch.cat([coords, random_coords], dim=0)
        
        print(f"  ✓ Created {self.num_points} coordinate queries")
        return coords
    
    def forward(self, images):
        batch_size = images.size(0)
        
        # Image features
        features = self.backbone(images)
        features = self.feature_proj(features)
        features = self.pos_encoding(features)
        
        B, C, H, W = features.shape
        features = features.flatten(2).permute(0, 2, 1)
        
        # Coordinate queries
        coord_queries = self.query_proj(self.coordinate_queries)
        queries = coord_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer
        decoded = self.transformer_decoder(queries, features)
        
        # Predict points
        points = self.point_head(decoded)
        
        return points


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)
    
    def forward(self, tensor):
        B, C, H, W = tensor.shape
        pe = self.pe.expand(B, C, H, W)
        return tensor + pe


class ChamferLoss(nn.Module):
    def forward(self, pred_points, gt_points):
        pred_expanded = pred_points.unsqueeze(2)
        gt_expanded = gt_points.unsqueeze(1)
        distances = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)
        
        min_dist_pred_to_gt = torch.min(distances, dim=2)[0]
        forward_loss = torch.mean(min_dist_pred_to_gt)
        
        min_dist_gt_to_pred = torch.min(distances, dim=1)[0]
        backward_loss = torch.mean(min_dist_gt_to_pred)
        
        return forward_loss + backward_loss


class SpreadLoss(nn.Module):
    """Encourages points to spread out."""
    
    def forward(self, pred_points, min_distance=0.03):
        B, N, _ = pred_points.shape
        
        pred_expanded_1 = pred_points.unsqueeze(2)
        pred_expanded_2 = pred_points.unsqueeze(1)
        
        distances = torch.sqrt(
            torch.sum((pred_expanded_1 - pred_expanded_2) ** 2, dim=-1) + 1e-8
        )
        
        mask = torch.eye(N, device=pred_points.device).unsqueeze(0).bool()
        distances = distances.masked_fill(mask, float('inf'))
        
        min_distances = torch.min(distances, dim=2)[0]
        spread_loss = torch.mean(F.relu(min_distance - min_distances))
        
        return spread_loss


class CombinedLoss(nn.Module):
    def __init__(self, chamfer_weight=1.0, spread_weight=0.01):
        super().__init__()
        self.chamfer_loss = ChamferLoss()
        self.spread_loss = SpreadLoss()
        self.chamfer_weight = chamfer_weight
        self.spread_weight = spread_weight
    
    def forward(self, pred_points, gt_points):
        chamfer = self.chamfer_loss(pred_points, gt_points)
        spread = self.spread_loss(pred_points)
        total_loss = self.chamfer_weight * chamfer + self.spread_weight * spread
        return total_loss, chamfer, spread


    def encode_sequence(pc):
        k = 10 #variable frequency bands
        freqs = (((2 ** torch.arange(k))/k) * torch.pi)
        pc_exp = pc.unsqueeze(-1) * freqs
        sin_enc = torch.sin(pc_exp)
        cos_enc = torch.cos(pc_exp)
        enc = torch.stack([sin_enc, cos_enc], dim=-1)
        enc = enc.view(pc.shape[0], -1) 
        pc = torch.cat([pc, enc], -1)
        return pc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with different point counts
    for num_points in [2048, 4096, 8192]:
        print(f"\n{'='*60}")
        print(f"Testing with {num_points} points")
        print(f"{'='*60}")
        
        model = Image2PointCloud(num_points=num_points).to(device)
        images = torch.randn(2, 3, 224, 224).to(device)
        points = model(images)
        
        print(f"Output shape: {points.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
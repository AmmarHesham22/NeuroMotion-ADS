import torch
import torch.nn as nn
from model.backbone import MultimodalBackbone

class ProjectorHead(nn.Module):
    """
    Non-linear projection for Contrastive Learning (InfoNCE).
    Maps from 2*d_model down to latent_dim (256).
    """
    def __init__(self, in_dim=512, latent_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, latent_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class RegressionHead(nn.Module):
    """
    Supervised head for ADOS prediction.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.linear(z).squeeze(-1)

class NeuroMotionModel(nn.Module):
    """
    The full NeuroMotion-ADS model combining backbone and heads.
    """
    def __init__(self, config):
        super().__init__()
        d_model = config['model']['d_model']
        fused_dim = 2 * d_model
        latent_dim = config['model']['latent_dim']
        
        self.backbone = MultimodalBackbone(
            d_model=d_model,
            num_joints=config['data']['num_joints'],
            gaze_dim=config['data']['gaze_dim'],
            st_gcn_layers=config['model']['st_gcn_layers'],
            transformer_layers=config['model']['transformer_layers'],
            n_heads=config['model']['n_heads'],
            dropout=config['model']['dropout']
        )
        
        self.projector = ProjectorHead(in_dim=fused_dim, latent_dim=latent_dim)
        self.regressor = RegressionHead(latent_dim=latent_dim)

    def forward(self, skeleton, gaze, edge_index):
        # Extract global representation
        z_global, attn_weights = self.backbone(skeleton, gaze, edge_index)
        
        # Map to latent manifold
        z_latent = self.projector(z_global)
        
        # Predict ADOS
        ados_pred = self.regressor(z_latent)
        
        return {
            "z": z_latent,
            "ados_pred": ados_pred,
            "attn_weights": attn_weights
        }

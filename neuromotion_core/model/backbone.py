import torch
import torch.nn as nn
import math
from model.modules.st_gcn import STGCNEncoder
from model.modules.tcn_mlp import TCNMLPEncoder
from model.modules.cross_attention import CrossModalAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]

class MultimodalBackbone(nn.Module):
    """
    Assembles ST-GCN, TCN, Fusion, and Transformer.
    """
    def __init__(
        self, 
        d_model=256, 
        num_joints=17, 
        gaze_dim=4, 
        st_gcn_layers=4, 
        transformer_layers=6, 
        n_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        # Encoders
        self.skel_encoder = STGCNEncoder(in_channels=3, d_model=d_model, num_layers=st_gcn_layers)
        self.gaze_encoder = TCNMLPEncoder(gaze_dim=gaze_dim, d_model=d_model)
        
        # Fusion
        self.fusion = CrossModalAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        
        # Transformer
        # H_fused is [B, T, 2*d_model]
        fused_dim = 2 * d_model
        self.pos_encoder = PositionalEncoding(d_model=fused_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim, 
            nhead=n_heads, 
            dim_feedforward=fused_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
    def forward(self, skeleton, gaze, edge_index):
        """
        skeleton: [B, 3, T, V]
        gaze: [B, T, D]
        edge_index: [2, E]
        """
        # Encode
        H_skel = self.skel_encoder(skeleton, edge_index) # [B, T, d_model]
        H_gaze = self.gaze_encoder(gaze)                 # [B, T, d_model]
        
        # Fuse
        H_fused, attn_weights = self.fusion(H_gaze, H_skel) # [B, T, 2*d_model]
        
        # Temporal Dynamics
        H_fused = self.pos_encoder(H_fused)
        H_temporal = self.transformer(H_fused) # [B, T, 2*d_model]
        
        # Global Max Pooling
        # [B, T, 2*d_model] -> [B, 2*d_model]
        z_global = torch.max(H_temporal, dim=1)[0]
        
        return z_global, attn_weights

import torch
import torch.nn as nn
from einops import rearrange

class TCNMLPEncoder(nn.Module):
    """
    Encoder for Gaze Vectors using MLP and Temporal Convolutional Network.
    Input: [B, T, D_gaze]
    Output: [B, T, d_model]
    """
    def __init__(self, gaze_dim=4, d_model=256, tcn_channels=[64, 128, 256]):
        super().__init__()
        
        # MLP Projection
        self.mlp = nn.Sequential(
            nn.Linear(gaze_dim, tcn_channels[0]),
            nn.ReLU(),
            nn.Linear(tcn_channels[0], tcn_channels[0])
        )
        
        # TCN layers
        layers = []
        in_ch = tcn_channels[0]
        for out_ch in tcn_channels[1:] + [d_model]:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
            
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, D_gaze]
        
        # Project each vector
        x = self.mlp(x) # [B, T, C0]
        
        # Prepare for 1D convolution: [B, C, T]
        x = rearrange(x, 'b t c -> b c t')
        
        # Apply TCN
        x = self.tcn(x) # [B, d_model, T]
        
        # Restore sequence format: [B, T, d_model]
        x = rearrange(x, 'b c t -> b t c')
        
        return x

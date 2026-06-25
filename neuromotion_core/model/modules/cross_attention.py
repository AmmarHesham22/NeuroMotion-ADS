import torch
import torch.nn as nn
import math

class CrossModalAttention(nn.Module):
    """
    Fuses Gaze (Query) and Skeleton (Key, Value) features.
    Input:
        H_gaze: [B, T, d_model]
        H_skel: [B, T, d_model]
    Output:
        H_fused: [B, T, 2 * d_model]
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # We can use standard MultiheadAttention
        # PyTorch MultiheadAttention expects [T, B, d_model] if batch_first=False
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, H_gaze, H_skel):
        # H_gaze (Q), H_skel (K, V)
        # B, T, D = H_gaze.shape
        
        # Cross Attention
        # query, key, value
        attn_out, attn_weights = self.mha(H_gaze, H_skel, H_skel)
        
        # Residual connection + norm on the gaze stream
        fused_gaze = self.norm(H_gaze + attn_out)
        
        # The design says "H_fused \in R^{B \times T \times 2d_model}"
        # We can concatenate the original or attended skeleton features
        H_fused = torch.cat([fused_gaze, H_skel], dim=-1) # [B, T, 2*d_model]
        
        return H_fused, attn_weights

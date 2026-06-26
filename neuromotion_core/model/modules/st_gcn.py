import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from einops import rearrange
from torch_geometric.utils import to_dense_adj
 
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Spatial Graph Convolution
        self.gcn = GCNConv(in_channels, out_channels)
        # Temporal Convolution
        self.tcn = nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        # x: [B, C, T, V]
        B, C, T, V = x.shape
        
        # Reshape for GCN: process all batches and frames as separate graphs
        # or process batch as a large graph. PyG typically expects [N, C] where N=B*T*V.
        # Alternatively, we can loop or reshape. Let's reshape to [B*T*V, C] if batched.
        # Actually, for standard ST-GCN over [B, C, T, V], we often apply GCN spatially per frame.
        
        # Rearrange to [(B T), V, C]
        x = rearrange(x, 'b c t v -> (b t) v c')
        
        # We need edge_index for a single graph of size V.
        # To apply GCNConv on batched separate graphs, we can just loop or use dense batching.
        # For simplicity and given standard PyG dense, we iterate or use batched graphs.
        # A simpler way without PyG DataLoader is to loop over (b t) or use standard PyTorch GCN.
        # Since directives say "PyTorch Geometric for the ST-GCN", let's use GCNConv.
        
        # Create a batched edge_index for (B*T) graphs
        # But wait, GCNConv supports [N, in_channels].
        # out_gcn = torch.zeros((B*T, V, self.gcn.out_channels), device=x.device)
        # for i in range(B*T):
        #     out_gcn[i] = self.gcn(x[i], edge_index)
        # استبدل حلقة الـ for loop بهذا الكود (Vectorized):
        # أولاً: نحول الـ edge_index إلى Dense Adjacency Matrix (مرة واحدة في الـ __init__ أو هنا)
        adj = to_dense_adj(edge_index, max_num_nodes=V).squeeze(0) # [V, V]

        # ثانياً: نطبق الـ GCN يدوياً كعمليات مصفوفات سريعة جداً
        # x shape is [(B T), V, C_in]
        # W weight shape from self.gcn.lin.weight
        x_transformed = self.gcn.lin(x) # [(B T), V, C_out]
        out_gcn = torch.matmul(adj, x_transformed) # [(B T), V, C_out]
        x = out_gcn # [(B T), V, C_out]
        x = rearrange(x, '(b t) v c -> b c t v', b=B, t=T)
        
        # Temporal Conv: operate over T dimension
        # Fold V into B: [B*V, C, T]
        x_tcn = rearrange(x, 'b c t v -> (b v) c t')
        x_tcn = self.tcn(x_tcn)
        x_tcn = self.bn(x_tcn)
        x_tcn = self.relu(x_tcn)
        
        # Restore shape
        x_out = rearrange(x_tcn, '(b v) c t -> b c t v', b=B, v=V)
        return x_out

class STGCNEncoder(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for Skeleton Encoding.
    Input: [B, 3, T, V]
    Output: [B, T, d_model]
    """
    def __init__(self, in_channels=3, d_model=256, num_layers=4):
        super().__init__()
        
        # A simple stack of ST-GCN layers
        layers = []
        channels = [in_channels] + [d_model // 2, d_model] + [d_model]*(num_layers-2)
        for i in range(num_layers):
            layers.append(STGCNLayer(channels[i], channels[i+1]))
            
        self.layers = nn.ModuleList(layers)
        
        # Global Spatial Pooling to map [B, C, T, V] to [B, C, T]
        self.pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x, edge_index):
        # x: [B, 3, T, V]
        for layer in self.layers:
            x = layer(x, edge_index)
            
        # Global Spatial Pooling
        # x is [B, d_model, T, V] -> [B, d_model, T, 1]
        x = self.pool(x).squeeze(-1) # [B, d_model, T]
        
        # Rearrange to [B, T, d_model]
        x = rearrange(x, 'b d t -> b t d')
        return x

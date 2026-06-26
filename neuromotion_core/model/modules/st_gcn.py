import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_adj
from einops import rearrange

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # استخدام DenseGCNConv لدعم المعالجة بالباتشات مباشرة وبأقصى سرعة رياضية
        self.gcn = DenseGCNConv(in_channels, out_channels)
        # Temporal Convolution
        self.tcn = nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, adj_batched):
        # x: [B, C, T, V]
        B, C, T, V = x.shape
        
        # تحويل الأبعاد لتناسب DenseGCNConv التي تتوقع [Batch, Nodes, Features]
        # نقوم بدمج الباتش مع الزمن: [(B*T), V, C]
        x_reshaped = rearrange(x, 'b c t v -> (b t) v c')
        
        # تطبيق الـ Graph Convolution بشكل صحيح مع كل خصائص PyG (Degree Normalization & Message Passing)
        out_gcn = self.gcn(x_reshaped, adj_batched) # [(B*T), V, C_out]
        
        # إرجاع الأبعاد لأصلها
        x_gcn = rearrange(out_gcn, '(b t) v c -> b c t v', b=B, t=T)
        
        # المعالجة الزمنية Temporal Convolution
        x_tcn = rearrange(x_gcn, 'b c t v -> (b v) c t')
        x_tcn = self.tcn(x_tcn)
        x_tcn = self.bn(x_tcn)
        x_tcn = self.relu(x_tcn)
        
        # الشكل النهائي
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
        
        layers = []
        channels = [in_channels] + [d_model // 2, d_model] + [d_model]*(num_layers-2)
        for i in range(num_layers):
            layers.append(STGCNLayer(channels[i], channels[i+1]))
            
        self.layers = nn.ModuleList(layers)
        
        # Global Spatial Pooling to map [B, C, T, V] to [B, C, T]
        self.pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x, edge_index):
        # x: [B, 3, T, V]
        B, C, T, V = x.shape
        
        # تحويل edge_index إلى Dense Adjacency Matrix مرة واحدة هنا لتوفير الحسابات
        adj = to_dense_adj(edge_index, max_num_nodes=V).squeeze(0) # [V, V]
        
        # توسيع مصفوفة التجاور لتغطي كل الجرافات (Batch * Time)
        adj_batched = adj.unsqueeze(0).expand(B * T, -1, -1) # [(B*T), V, V]
        
        for layer in self.layers:
            x = layer(x, adj_batched)
            
        # Global Spatial Pooling
        # x is [B, d_model, T, V] -> [B, d_model, T, 1]
        x = self.pool(x).squeeze(-1) # [B, d_model, T]
        
        # Rearrange to [B, T, d_model]
        x = rearrange(x, 'b d t -> b t d')
        return x
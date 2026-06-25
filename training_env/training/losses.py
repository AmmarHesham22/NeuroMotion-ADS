import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Numerically stable InfoNCE Loss for Self-Supervised Learning.
    Based on SimCLR formulation.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        # Normalize representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Batch size
        B = z1.size(0)
        
        # Concatenate z1 and z2: [2B, D]
        features = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix: [2B, 2B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels: 
        # i matches with i + B
        # i + B matches with i
        labels = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z1.device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss

class MaskedJointLoss(nn.Module):
    """
    An optional auxiliary loss for reconstructing masked joints.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_joints, target_joints, mask):
        # Compute MSE only on masked locations
        return self.mse(pred_joints[mask], target_joints[mask])

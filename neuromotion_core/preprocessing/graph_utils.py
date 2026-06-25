import torch
import numpy as np

def get_coco_17_edges():
    """
    Returns the edge connections for the COCO 17-joint format.
    0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear, 
    5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 
    9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip, 
    13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle
    """
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6),                  # Neck to shoulders (approx)
        (5, 7), (7, 9),                  # Left arm
        (6, 8), (8, 10),                 # Right arm
        (5, 11), (6, 12),                # Torso
        (11, 13), (13, 15),              # Left leg
        (12, 14), (14, 16),              # Right leg
        (11, 12)                         # Pelvis
    ]
    return edges

def build_adjacency_matrix(num_joints=17, self_loops=True):
    """
    Builds the adjacency matrix A for the GCN.
    Returns edge_index format required by PyTorch Geometric.
    """
    edges = get_coco_17_edges()
    
    # Make it undirected
    undirected_edges = []
    for src, dst in edges:
        undirected_edges.append((src, dst))
        undirected_edges.append((dst, src))
        
    if self_loops:
        for i in range(num_joints):
            undirected_edges.append((i, i))
            
    # Convert to edge_index [2, num_edges]
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    return edge_index

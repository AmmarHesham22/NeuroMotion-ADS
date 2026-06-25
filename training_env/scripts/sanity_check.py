import sys
import os
import yaml
import torch

# Add the project root to sys.path so we can import modules

from dataset.builder import NeuroMotionDataset
from training.trainer import NeuroMotionLightningModule
from neuromotion_core.preprocessing.graph_utils import build_adjacency_matrix
from inference.anomaly_scorer import AnomalyScorer

def run_sanity_check():
    print("=== Starting NeuroMotion-ADS Sanity Check ===")
    
    # 1. Load config
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"1. Configuration loaded. Batch size: {config['training']['batch_size']}")
    
    # 2. Initialize Model
    model = NeuroMotionLightningModule(config)
    print("2. NeuroMotionLightningModule initialized.")
    
    # 3. Create a small mock dataset (SSL mode)
    mock_files = ["mock1.json", "mock2.json", "mock3.json", "mock4.json"]
    dataset = NeuroMotionDataset(file_paths=mock_files, dataset_type="MIXED", mode="ssl", window_size=config['data']['window_size'])
    print(f"3. Mock Dataset created with {len(dataset)} items.")
    
    # 4. Fetch a batch
    # We will manually collate a batch for testing
    batch_v1_skel, batch_v1_gaze = [], []
    batch_v2_skel, batch_v2_gaze = [], []
    
    for i in range(2): # batch_size = 2 for quick check
        item = dataset[i]
        batch_v1_skel.append(item["v1"]["skeleton"])
        batch_v1_gaze.append(item["v1"]["gaze"])
        batch_v2_skel.append(item["v2"]["skeleton"])
        batch_v2_gaze.append(item["v2"]["gaze"])
        
    v1_skel = torch.stack(batch_v1_skel) # [B, 3, T, V]
    v1_gaze = torch.stack(batch_v1_gaze) # [B, T, D]
    
    v2_skel = torch.stack(batch_v2_skel)
    v2_gaze = torch.stack(batch_v2_gaze)
    
    # Require gradients on inputs for gradient flow check
    v1_skel.requires_grad_(True)
    v1_gaze.requires_grad_(True)
    
    print("\n=== Tensor Shape Verification ===")
    print(f"Input Skeleton (v1): {v1_skel.shape}")
    print(f"Input Gaze (v1):     {v1_gaze.shape}")
    
    # Explicitly print shapes at key boundaries by passing through sub-modules
    edge_index = model.edge_index
    
    # A. STGCN Encoder Output
    stgcn_out = model.model.backbone.skel_encoder(v1_skel, edge_index)
    print(f"STGCNEncoder Output: {stgcn_out.shape} -> Expected: [B, T, d_model]")
    
    # B. TCNMLP Encoder Output
    tcn_out = model.model.backbone.gaze_encoder(v1_gaze)
    print(f"TCNMLPEncoder Output: {tcn_out.shape} -> Expected: [B, T, d_model]")
    
    # C. Cross-Attention & Transformer Backbone Output
    z_global, attn = model.model.backbone(v1_skel, v1_gaze, edge_index)
    print(f"Backbone Global Out: {z_global.shape} -> Expected: [B, 2*d_model]")
    
    # D. Final Model Outputs
    out_dict = model.model(v1_skel, v1_gaze, edge_index)
    z_latent = out_dict["z"]
    ados_pred = out_dict["ados_pred"]
    print(f"Final Embeddings (z): {z_latent.shape} -> Expected: [B, latent_dim]")
    print(f"ADOS Score Output:    {ados_pred.shape} -> Expected: [B]")
    
    # Mock anomaly scorer testing
    scorer = AnomalyScorer()
    # fake baseline
    fake_baseline = torch.randn(100, 256).numpy()
    scorer.fit_baseline(fake_baseline)
    anomaly_score = scorer.score_clip(z_latent[0].detach().numpy().reshape(1, -1))
    print(f"Anomaly Score (mock): {anomaly_score:.4f} (type: float)")

    print("\n=== Gradient Flow Verification ===")
    # 5. Run Backward Pass
    # Get v2 outputs
    out_dict_v2 = model.model(v2_skel, v2_gaze, edge_index)
    
    # Calculate InfoNCE loss
    loss = model.infonce_loss(z_latent, out_dict_v2["z"])
    print(f"Initial InfoNCE Loss: {loss.item():.4f}")
    
    print("Calling loss.backward()...")
    loss.backward()
    
    if v1_skel.grad is not None and v1_gaze.grad is not None:
        skel_grad_sum = v1_skel.grad.abs().sum().item()
        gaze_grad_sum = v1_gaze.grad.abs().sum().item()
        
        if skel_grad_sum > 0 and gaze_grad_sum > 0:
            print("SUCCESS: Gradients successfully flowed back to both raw inputs.")
            print(f"  Skeleton Input Grad Sum: {skel_grad_sum:.4f}")
            print(f"  Gaze Input Grad Sum:     {gaze_grad_sum:.4f}")
        else:
            print("WARNING: Gradients computed but sum to 0. Check network connections.")
    else:
        print("ERROR: Gradients failed to reach inputs.")

if __name__ == "__main__":
    run_sanity_check()

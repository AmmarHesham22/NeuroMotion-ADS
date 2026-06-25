import os
import sys
import glob
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset.builder import NeuroMotionDataset
from training.trainer import NeuroMotionLightningModule

def find_latest_checkpoint(ckpt_dir: str) -> str:
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
    # Sort by modification time to get the latest
    return max(ckpts, key=os.path.getmtime)

def main():
    print("Starting Manifold Visualization...")
    
    # 1. Load Config
    config_path = os.path.join('config', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Find Checkpoint
    ckpt_dir = "checkpoints"
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    
    if ckpt_path is None:
        print(f"Error: No checkpoints found in {ckpt_dir}/ directory. Please ensure training has saved at least one checkpoint.")
        return
        
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load model
    model = NeuroMotionLightningModule.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    
    # 3. Load Data
    processed_dir = os.path.join("data", "processed")
    all_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    
    if not all_files:
        print(f"Error: No processed tensor files found in {processed_dir}. Run build_dataset.py first.")
        return
        
    # We load the entire dataset or a large batch to have enough points for t-SNE
    dataset = NeuroMotionDataset(file_paths=all_files, dataset_type="MIXED", mode="supervised", window_size=config['data']['window_size'])
    # Since we might have very few files locally for testing, we use batch_size = len(dataset) or a large number
    batch_size = min(len(dataset), 256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Extract Embeddings
    print("Extracting embeddings...")
    z_vectors = []
    predicted_ados = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Mode 'supervised' yields skel, gaze, target
            skel, gaze, _ = batch
            
            # Forward pass through the base model
            # edge_index is constructed dynamically in the lightning module, so we grab it from there
            z, ados_pred, _ = model.model(skel, gaze, model.edge_index)
            
            z_vectors.append(z)
            predicted_ados.append(ados_pred)
            
    # Concatenate all batches
    z_vectors = torch.cat(z_vectors, dim=0).numpy()
    predicted_ados = torch.cat(predicted_ados, dim=0).squeeze().numpy()
    
    print(f"Extracted {z_vectors.shape[0]} embeddings of dimension {z_vectors.shape[1]}.")
    
    # 5. Dimensionality Reduction (t-SNE)
    # If we have fewer than 2 points, t-SNE will fail. We need a minimum number of points.
    n_samples = z_vectors.shape[0]
    if n_samples < 2:
        print("Error: Need at least 2 samples to run t-SNE visualization.")
        return
        
    perplexity = min(30, n_samples - 1)
    print(f"Running t-SNE with perplexity={perplexity}...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    z_2d = tsne.fit_transform(z_vectors)
    
    # 6. Visualization
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with seaborn
    scatter = sns.scatterplot(
        x=z_2d[:, 0], 
        y=z_2d[:, 1], 
        hue=predicted_ados, 
        palette="viridis", 
        s=100, 
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )
    
    plt.title("NeuroMotion-ADS Latent Manifold (t-SNE)", fontsize=16, pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    # Customize legend/colorbar
    norm = plt.Normalize(predicted_ados.min(), predicted_ados.max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    # Remove the default seaborn legend
    scatter.get_legend().remove()
    # Add colorbar
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Predicted ADOS Severity', rotation=270, labelpad=15, fontsize=12)
    
    # Save the plot
    output_path = os.path.join("checkpoints", "manifold_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved manifold visualization to {output_path}")

if __name__ == "__main__":
    main()

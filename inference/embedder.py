import torch
import numpy as np

class Embedder:
    """
    Inference tool to extract the 256D continuous behavioral embedding.
    """
    def __init__(self, model_checkpoint, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        from training.trainer import NeuroMotionLightningModule
        self.device = device
        self.model = NeuroMotionLightningModule.load_from_checkpoint(model_checkpoint, config=config)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_clip(self, skeleton_tensor: torch.Tensor, gaze_tensor: torch.Tensor):
        """
        Extracts embedding for a 10s sequence.
        skeleton_tensor: [1, 3, T, V]
        gaze_tensor: [1, T, D]
        """
        skeleton_tensor = skeleton_tensor.to(self.device)
        gaze_tensor = gaze_tensor.to(self.device)
        
        output = self.model(skeleton_tensor, gaze_tensor)
        
        z = output["z"]
        ados_pred = output["ados_pred"]
        attn = output["attn_weights"]
        
        return {
            "embedding": z.cpu().numpy(),
            "ados_pred": ados_pred.cpu().numpy(),
            "attention_weights": attn.cpu().numpy()
        }

import os
import yaml
import torch

from neuromotion_core.model.heads import NeuroMotionModel
from neuromotion_core.preprocessing.graph_utils import build_adjacency_matrix
# ... (باقي الكود زي ما هو بدون تعديل)

def load_inference_model(checkpoint_path: str, config_path: str = None, device: str = "cpu"):
    """
    Loads a PyTorch Lightning .ckpt file into a pure PyTorch inference state.
    """
    if config_path is None:
        # Default assumption relative to the project root
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../neuromotion_core/config/default_config.yaml'))
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 1. Instantiate the core architecture
    model = NeuroMotionModel(config)
    
    # Generate the static edge_index graph required for inference
    edge_index = build_adjacency_matrix(config['data']['num_joints'], self_loops=True)
    edge_index = edge_index.to(device)
    
    # 2. Load the state dictionary
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 3. Strip the 'model.' prefix from Lightning checkpoints
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            cleaned_state_dict[key[6:]] = value
        else:
            cleaned_state_dict[key] = value
            
    # Load weights (strict=False to allow ignoring lightning-specific buffers)
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    # 4. Move to target device and set evaluation mode
    model = model.to(device)
    model.eval()
    
    return model, edge_index

class NeuroMotionInferenceSession:
    """
    Utility wrapper to ensure safe inference contexts.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model, self.edge_index = load_inference_model(checkpoint_path, device=device)
        
    @torch.no_grad()
    def predict(self, skeleton_chunk: torch.Tensor, gaze_chunk: torch.Tensor):
        skeleton_chunk = skeleton_chunk.to(self.device)
        gaze_chunk = gaze_chunk.to(self.device)
        
        # 1. الموديل بيرجع Dictionary
        outputs = self.model(skeleton_chunk, gaze_chunk, self.edge_index)
        
        # 2. استخراج القيم من الـ Dictionary باستخدام المفاتيح
        z_latent = outputs["z"]
        ados_pred = outputs["ados_pred"]
        
        return {
            "z": z_latent.cpu().numpy(),
            "ados": ados_pred.cpu().numpy(),
            # 3. الموديل مبيرجعش Anomaly بشكل مباشر، فبنحط قيمة افتراضية عشان الـ API يكمل
            "anomaly": [0.0]
        }

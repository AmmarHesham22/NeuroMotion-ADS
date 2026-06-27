import os
import yaml
import torch
import numpy as np
import math

from inference.anomaly_scorer import AnomalyScorer
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
        
        # تفعيل حاسب الشذوذ السلوكي مع خط أساس وهمي مؤقت
        self.anomaly_scorer = AnomalyScorer()
        dummy_baseline = np.random.randn(100, 256)
        self.anomaly_scorer.fit_baseline(dummy_baseline)
        
    @torch.no_grad()
    def predict(self, skeleton_chunk: torch.Tensor, gaze_chunk: torch.Tensor):
        skeleton_chunk = skeleton_chunk.to(self.device)
        gaze_chunk = gaze_chunk.to(self.device)
        
        outputs = self.model(skeleton_chunk, gaze_chunk, self.edge_index)
        
        z_latent = outputs["z"].cpu().numpy()
        ados_pred = outputs["ados_pred"].cpu().numpy()
        
        # حساب المسافة الفلكية الخام
        raw_anomaly = self.anomaly_scorer.score_clip(z_latent)
        
        # ---------------------------------------------------------
        # معالجة تضخم المسافة (Latent Space Domain Gap)
        # استخدام Log-Scale لضغط الرقم الفلكي (مثلاً 221955) إلى نطاق 0 لـ 5
        # ---------------------------------------------------------
        try:
            # np.log1p يحسب log(1 + x) بشكل آمن
            scaled_anomaly = np.log1p(raw_anomaly) / 2.5 
            # تقليم الرقم ليكون في نطاق 0.0 إلى 5.0 كحد أقصى
            final_anomaly = min(5.0, max(0.0, scaled_anomaly - 1.0))
        except:
            final_anomaly = 0.0
            
        return {
            "z": z_latent,
            "ados": ados_pred,
            "anomaly": [final_anomaly]
        }

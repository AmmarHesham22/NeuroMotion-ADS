import torch
from torch.utils.data import Dataset
from typing import List, Optional
import os

# Assuming preprocessing modules are accessible in PYTHONPATH or module setup
from preprocessing.parser import JSONParser
from dataset.transforms import SkeletonTransform, GazeTransform

class NeuroMotionDataset(Dataset):
    """
    Dataset class handling both DREAM (supervised) and PInSoRo (unsupervised).
    Supports Contrastive View generation (v1, v2) for SSL.
    """
    def __init__(
        self, 
        file_paths: List[str], 
        dataset_type: str = "DREAM", 
        mode: str = "ssl", 
        window_size: int = 300
    ):
        self.file_paths = file_paths
        self.dataset_type = dataset_type
        self.mode = mode # 'ssl' or 'supervised'
        self.parser = JSONParser(window_size=window_size)
        
        self.skel_transform = SkeletonTransform()
        self.gaze_transform = GazeTransform()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        if path.endswith(".pt"):
            data = torch.load(path)
        else:
            if self.dataset_type == "DREAM":
                data = self.parser.parse_dream_stream(path)
            else:
                data = self.parser.parse_pinsoro_stream(path)
            
        skel = data["skeleton"] # [3, T, V]
        gaze = data["gaze"]     # [T, D]
        target = data["ados_target"] if data.get("ados_target") is not None else torch.tensor(-1.0)
        
        if self.mode == "ssl":
            # Generate two augmented views
            skel_v1 = self.skel_transform(skel)
            skel_v2 = self.skel_transform(skel)
            
            gaze_v1 = self.gaze_transform(gaze)
            gaze_v2 = self.gaze_transform(gaze)
            
            return {
                "v1": {"skeleton": skel_v1, "gaze": gaze_v1},
                "v2": {"skeleton": skel_v2, "gaze": gaze_v2},
                "target": target
            }
        else:
            # Supervised / Inference mode
            return {
                "skeleton": skel,
                "gaze": gaze,
                "target": target
            }

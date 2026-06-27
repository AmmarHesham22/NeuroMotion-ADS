import json
import os
import numpy as np

class CoordinateScaler:
    """
    Applies Domain Adaptation: Maps MediaPipe [0, 1] coords to DREAM Space
    """
    def __init__(self):
        self.is_fitted = False
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        
    def save_scaler(self, filepath: str):
        if not self.is_fitted:
            print("Warning: Scaler is not fitted. Saving empty scaler.")
            
        data = {
            "is_fitted": self.is_fitted,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_scaler(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"Warning: Scaler file {filepath} not found.")
            self.is_fitted = False
            return
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.is_fitted = data.get("is_fitted", False)
            
            if data.get("mean") is not None:
                self.mean = np.array(data["mean"], dtype=np.float32)
            if data.get("std") is not None:
                self.std = np.array(data["std"], dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.is_fitted = False
            
    def transform(self, skeleton_chunk: np.ndarray) -> np.ndarray:
        """
        Reverse Mapping: Inflates MediaPipe normalized coords to match DREAM's raw scale.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            return skeleton_chunk
            
        transformed = skeleton_chunk.copy()
        
        # إحداثيات MediaPipe تتراوح بين 0 و 1 (متوسط تقريبي 0.5 وانحراف 0.25)
        mp_mean = 0.5
        mp_std = 0.25
        
        # 1. تكبير إحداثيات X (القناة صفر)
        transformed[0] = ((transformed[0] - mp_mean) / mp_std) * self.std[0] + self.mean[0]
        
        # 2. تكبير إحداثيات Y (القناة 1)
        transformed[1] = ((transformed[1] - mp_mean) / mp_std) * self.std[1] + self.mean[1]
        
        # القناة 2 (الثقة - Confidence) تظل كما هي
        
        return transformed
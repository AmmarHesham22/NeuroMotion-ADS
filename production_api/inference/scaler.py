import json
import os
import numpy as np

class CoordinateScaler:
    """
    Spatial consistency anchor to ensure live video coordinates match 
    the training dataset distribution.
    """
    def __init__(self):
        self.is_fitted = False
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        
    def fit(self, data: np.ndarray, method: str = "zscore"):
        """
        Fits the scaler to historical training data.
        data shape is arbitrary, typically [N, Channels, Frames, Vertices]
        Calculations are typically done per channel.
        """
        # Example logic if we ever fit at runtime/offline
        pass
        
    def save_scaler(self, filepath: str):
        """
        Stores calibration metrics as JSON.
        """
        if not self.is_fitted:
            print("Warning: Scaler is not fitted. Saving empty scaler.")
            
        data = {
            "is_fitted": self.is_fitted,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "min_val": self.min_val.tolist() if self.min_val is not None else None,
            "max_val": self.max_val.tolist() if self.max_val is not None else None
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_scaler(self, filepath: str):
        """
        Loads calibration metrics.
        """
        if not os.path.exists(filepath):
            print(f"Warning: Scaler file {filepath} not found. Defaulting to identity passthrough.")
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
            if data.get("min_val") is not None:
                self.min_val = np.array(data["min_val"], dtype=np.float32)
            if data.get("max_val") is not None:
                self.max_val = np.array(data["max_val"], dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading scaler from {filepath}: {e}. Defaulting to identity passthrough.")
            self.is_fitted = False
            
    def transform(self, skeleton_chunk: np.ndarray) -> np.ndarray:
        """
        Applies loaded metrics to the live [3, 300, 17] tensor.
        Fail-safe fallback: identity transformation if not fitted.
        """
        if not self.is_fitted:
            # Passthrough transformation
            return skeleton_chunk
            
        # Example Z-score scaling applying standard normalization per channel
        # Assumes self.mean and self.std are shapes [3, 1, 1] to broadcast properly
        transformed = skeleton_chunk.copy()
        
        if self.mean is not None and self.std is not None:
            # Add eps to prevent division by zero
            eps = 1e-8
            transformed = (transformed - self.mean) / (self.std + eps)
            
        return transformed

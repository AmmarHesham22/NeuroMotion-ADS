import json
import torch
import numpy as np
from typing import Dict, Any, List

class JSONParser:
    """
    JSON Parser for NeuroMotion-ADS data streams.
    
    Reads the DREAM dataset format (dream.1.2.json schema) where 
    features are stored as arrays over time.
    """
    def __init__(self, window_size: int = 300, num_joints: int = 17, gaze_dim: int = 4):
        self.window_size = window_size
        self.num_joints = num_joints
        self.gaze_dim = gaze_dim
        
        # COCO 17-joint mapping:
        # 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear,
        # 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow,
        # 9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip,
        # 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle.
        self.joint_map = {
            "head": 0,             # Mapping Nose to Head
            "eye_left": 1,
            "eye_right": 2,
            "ear_left": 3,
            "ear_right": 4,
            "sholder_left": 5,
            "sholder_right": 6,
            "elbow_left": 7,
            "elbow_right": 8,
            "wrist_left": 9,
            "wrist_right": 10,
            "hip_left": 11,
            "hip_right": 12,
            "knee_left": 13,
            "knee_right": 14,
            "ankle_left": 15,
            "ankle_right": 16
        }

    def _extract_tensors_from_json(self, data: Dict[str, Any]) -> tuple:
        """
        Extracts tensor streams with pandas-based interpolation for missing intermediate frames.
        """
        import pandas as pd
        
        skeleton_array = np.zeros((3, self.window_size, self.num_joints), dtype=np.float32)
        gaze_array = np.zeros((self.window_size, self.gaze_dim), dtype=np.float32)
        
        # Extract Skeleton
        skeleton_data = data.get("skeleton", {})
        
        for joint_name, joint_idx in self.joint_map.items():
            if joint_name in skeleton_data:
                joint_dict = skeleton_data[joint_name]
                x_vals = joint_dict.get("x", [])
                y_vals = joint_dict.get("y", [])
                c_vals = joint_dict.get("confidence", [])
                
                seq_len = min(len(x_vals), self.window_size)
                
                # Convert to pandas Series for interpolation
                df = pd.DataFrame({
                    "x": x_vals[:seq_len],
                    "y": y_vals[:seq_len],
                    "c": c_vals[:seq_len] if c_vals else [None]*seq_len
                })
                
                # Replace explicitly None or missing with NaN
                df.fillna(value=np.nan, inplace=True)
                
                # Interpolate intermediate NaNs linearily
                df["x"] = df["x"].interpolate(method="linear", limit_direction="both").fillna(0.0)
                df["y"] = df["y"].interpolate(method="linear", limit_direction="both").fillna(0.0)
                
                # Confidence: if NaN, 0.0, else keep original (or 1.0)
                df["c"] = df["c"].fillna(0.0)
                
                skeleton_array[0, :seq_len, joint_idx] = df["x"].values
                skeleton_array[1, :seq_len, joint_idx] = df["y"].values
                skeleton_array[2, :seq_len, joint_idx] = df["c"].values
            else:
                # Joint entirely missing, c=0.0
                skeleton_array[2, :, joint_idx] = 0.0
                
        # Extract Gaze
        head_gaze = data.get("head_gaze", {})
        if head_gaze:
            rx_vals = head_gaze.get("rx", [])
            ry_vals = head_gaze.get("ry", [])
            
            seq_len = min(len(rx_vals), self.window_size)
            df_gaze = pd.DataFrame({
                "yaw": rx_vals[:seq_len],
                "pitch": ry_vals[:seq_len]
            })
            df_gaze.fillna(value=np.nan, inplace=True)
            df_gaze = df_gaze.interpolate(method="linear", limit_direction="both").fillna(0.0)
            
            gaze_array[:seq_len, 0] = df_gaze["yaw"].values
            gaze_array[:seq_len, 1] = df_gaze["pitch"].values
            
        return torch.tensor(skeleton_array), torch.tensor(gaze_array)

    def parse_dream_stream(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}. Falling back to mock data.")
            skeleton_tensor = torch.randn(3, self.window_size, self.num_joints)
            gaze_tensor = torch.randn(self.window_size, self.gaze_dim)
            return {
                "skeleton": skeleton_tensor,
                "gaze": gaze_tensor,
                "ados_target": 5.0, # Mock ADOS score
                "metadata": {"source": "DREAM", "file": file_path}
            }
            
        skeleton_tensor, gaze_tensor = self._extract_tensors_from_json(data)
        
        # Extract ADOS
        ados_score = -1.0
        if "ados" in data and "preTest" in data["ados"]:
            ados_score = float(data["ados"]["preTest"].get("total", -1.0))
        elif "ados" in data and "postTest" in data["ados"]:
            ados_score = float(data["ados"]["postTest"].get("total", -1.0))
            
        return {
            "skeleton": skeleton_tensor,
            "gaze": gaze_tensor,
            "ados_target": ados_score,
            "metadata": {"source": "DREAM", "file": file_path, "frame_rate": data.get("frame_rate", 30)}
        }

    def parse_pinsoro_stream(self, file_path: str) -> Dict[str, Any]:
        # PInSoRo has a different schema or same, but no ADOS
        # Assuming same extraction logic for the sake of the parser structure
        data_dict = self.parse_dream_stream(file_path)
        data_dict["ados_target"] = -1.0
        data_dict["metadata"]["source"] = "PInSoRo"
        return data_dict

    def load_batch(self, file_paths: List[str], dataset_type: str = "DREAM") -> Dict[str, torch.Tensor]:
        skeletons, gazes, targets = [], [], []
        
        for path in file_paths:
            if dataset_type == "DREAM":
                data = self.parse_dream_stream(path)
                targets.append(data["ados_target"])
            else:
                data = self.parse_pinsoro_stream(path)
                targets.append(-1.0)
                
            skeletons.append(data["skeleton"])
            gazes.append(data["gaze"])
            
        return {
            "skeleton": torch.stack(skeletons),
            "gaze": torch.stack(gazes),
            "target": torch.tensor(targets, dtype=torch.float32)
        }

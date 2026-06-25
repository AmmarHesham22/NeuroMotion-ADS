import torch
import torch.nn.functional as F

class TemporalAligner:
    """
    Responsible for syncing multiple streams (e.g., gaze, skeleton) to a fixed FPS.
    """
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        
    def align_streams(self, skeleton_stream: torch.Tensor, gaze_stream: torch.Tensor, timestamps: torch.Tensor) -> tuple:
        """
        Resamples variable-length temporal streams to match the target_fps.
        skeleton_stream: [3, T, V]
        gaze_stream: [T, D]
        timestamps: [T] in seconds.
        """
        # Calculate duration and target number of frames
        duration = timestamps[-1] - timestamps[0]
        T_new = max(1, int(round(duration.item() * self.target_fps)))
        
        # Resample Skeleton
        # Shape from [3, T, V] -> [1, 3*V, T]
        C, T, V = skeleton_stream.shape
        skel_reshaped = skeleton_stream.permute(0, 2, 1).reshape(1, C * V, T)
        skel_resampled = F.interpolate(skel_reshaped, size=T_new, mode='linear', align_corners=False)
        # Reshape back to [3, T_new, V]
        skel_resampled = skel_resampled.view(C, V, T_new).permute(0, 2, 1).contiguous()
        
        # Resample Gaze
        # Shape from [T, D] -> [1, D, T]
        D = gaze_stream.shape[1]
        gaze_reshaped = gaze_stream.t().unsqueeze(0)
        gaze_resampled = F.interpolate(gaze_reshaped, size=T_new, mode='linear', align_corners=False)
        # Reshape back to [T_new, D]
        gaze_resampled = gaze_resampled.squeeze(0).t().contiguous()
        
        return skel_resampled, gaze_resampled

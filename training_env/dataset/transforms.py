import torch
import torch.nn.functional as F

class SkeletonTransform:
    """
    Augmentations for Skeletal data.
    """
    def __init__(self, jitter_scale=0.01, mask_prob=0.1):
        self.jitter_scale = jitter_scale
        self.mask_prob = mask_prob

    def spatial_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds Gaussian noise to joint coordinates.
        x: [3, T, V]
        """
        noise = torch.randn_like(x) * self.jitter_scale
        return x + noise

    def node_dropping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly masks out joints (nodes).
        x: [3, T, V]
        """
        mask = torch.rand(x.shape[2]) > self.mask_prob
        x_aug = x.clone()
        x_aug[:, :, ~mask] = 0.0 # Zero out dropped nodes
        return x_aug
        
    def temporal_masking(self, x: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
        """
        Masks out contiguous frames.
        x: [3, T, V]
        """
        T = x.shape[1]
        mask_len = int(T * mask_ratio)
        start_idx = torch.randint(0, T - mask_len, (1,)).item()
        x_aug = x.clone()
        x_aug[:, start_idx:start_idx+mask_len, :] = 0.0
        return x_aug

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_jitter(x)
        x = self.node_dropping(x)
        x = self.temporal_masking(x)
        return x

class GazeTransform:
    """
    Augmentations for Gaze data.
    """
    def __init__(self, noise_scale=0.05):
        self.noise_scale = noise_scale
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, D]
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise

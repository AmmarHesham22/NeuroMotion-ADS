import torch
from torch.utils.data import Sampler
from typing import List

class BalancedBatchSampler(Sampler):
    """
    Samples batches to maintain a specific ratio of DREAM vs PInSoRo data
    during joint optimization.
    """
    def __init__(self, dream_indices: List[int], pinsoro_indices: List[int], batch_size: int, dream_ratio: float = 0.5):
        self.dream_indices = dream_indices
        self.pinsoro_indices = pinsoro_indices
        self.batch_size = batch_size
        self.dream_ratio = dream_ratio
        
        self.num_dream_per_batch = int(batch_size * dream_ratio)
        self.num_pinsoro_per_batch = batch_size - self.num_dream_per_batch
        
        # Determine total batches based on the maximum number of iterations needed
        # to ensure at least one dataset completes an epoch.
        if len(self.dream_indices) > 0 and self.num_dream_per_batch > 0:
            dream_batches = len(self.dream_indices) // self.num_dream_per_batch
        else:
            dream_batches = 0
            
        if len(self.pinsoro_indices) > 0 and self.num_pinsoro_per_batch > 0:
            pinsoro_batches = len(self.pinsoro_indices) // self.num_pinsoro_per_batch
        else:
            pinsoro_batches = 0
            
        self.total_batches = max(dream_batches, pinsoro_batches)

    def __iter__(self):
        import random
        # Shuffle indices at the start of each epoch
        dream_shuffled = self.dream_indices.copy()
        pinsoro_shuffled = self.pinsoro_indices.copy()
        random.shuffle(dream_shuffled)
        random.shuffle(pinsoro_shuffled)
        
        dream_idx = 0
        pinsoro_idx = 0
        
        for _ in range(self.total_batches):
            batch = []
            
            # Sample DREAM
            for _ in range(self.num_dream_per_batch):
                if dream_idx >= len(dream_shuffled):
                    random.shuffle(dream_shuffled)
                    dream_idx = 0
                if len(dream_shuffled) > 0:
                    batch.append(dream_shuffled[dream_idx])
                    dream_idx += 1
                    
            # Sample PInSoRo
            for _ in range(self.num_pinsoro_per_batch):
                if pinsoro_idx >= len(pinsoro_shuffled):
                    random.shuffle(pinsoro_shuffled)
                    pinsoro_idx = 0
                if len(pinsoro_shuffled) > 0:
                    batch.append(pinsoro_shuffled[pinsoro_idx])
                    pinsoro_idx += 1
                    
            yield batch
            
    def __len__(self):
        return self.total_batches

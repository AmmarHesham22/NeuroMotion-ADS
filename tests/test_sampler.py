import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.sampler import BalancedBatchSampler

def test_sampler():
    print("Testing BalancedBatchSampler with uneven list sizes...")
    
    # Very uneven lists
    dream_indices = list(range(100))      # 100 items
    pinsoro_indices = list(range(100, 105)) # 5 items
    
    batch_size = 32
    dream_ratio = 0.5 # 16 DREAM, 16 PInSoRo per batch
    
    sampler = BalancedBatchSampler(dream_indices, pinsoro_indices, batch_size, dream_ratio)
    
    print(f"Total expected batches: {len(sampler)}")
    
    batch_count = 0
    for batch in sampler:
        assert len(batch) == batch_size, f"Batch size mismatch: {len(batch)} != {batch_size}"
        
        # Verify ratio
        dream_items = [idx for idx in batch if idx < 100]
        pinsoro_items = [idx for idx in batch if idx >= 100]
        
        assert len(dream_items) == 16, f"Expected 16 DREAM items, got {len(dream_items)}"
        assert len(pinsoro_items) == 16, f"Expected 16 PInSoRo items, got {len(pinsoro_items)}"
        
        batch_count += 1
        
    assert batch_count == len(sampler), f"Iterated {batch_count} times, expected {len(sampler)}"
    print("SUCCESS: Sampler handled uneven list sizes correctly with cyclic yielding!")

if __name__ == "__main__":
    test_sampler()

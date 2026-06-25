#!/bin/bash
# run_finetune.sh
# Executes the Supervised Learning phase for ADOS prediction on DREAM dataset.

export CUDA_VISIBLE_DEVICES=0

echo "Starting NeuroMotion-ADS Fine-tuning Phase..."

cat << 'EOF' > finetune.py
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset.builder import NeuroMotionDataset
from training.trainer import NeuroMotionLightningModule

with open('config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set supervised weight
config['training']['lambda_mse'] = 1.0
config['training']['lambda_infonce'] = 0.1 # Optional: keep small SSL loss as regularizer

dream_files = ["mock_dream_1.json", "mock_dream_2.json"]

dataset = NeuroMotionDataset(file_paths=dream_files, dataset_type="DREAM", mode="supervised", window_size=config['data']['window_size'])
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Load checkpoint from pre-training ideally
model = NeuroMotionLightningModule(config)
# For finetuning, we might unfreeze specific layers or freeze backbone. 
# Here we train all end-to-end.

trainer = pl.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(model, dataloader)
EOF

python finetune.py

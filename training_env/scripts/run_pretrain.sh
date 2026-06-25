#!/bin/bash
# run_pretrain.sh
# Executes the Self-Supervised Learning (SSL) phase using InfoNCE on both DREAM and PInSoRo datasets.

export CUDA_VISIBLE_DEVICES=0,1

echo "Starting NeuroMotion-ADS Pre-training Phase..."

# Normally we would call a Python script here, e.g.
# python -m training.trainer --mode ssl --config config/default_config.yaml

# Mock implementation since we don't have a main.py wrapper yet.
cat << 'EOF' > pretrain.py
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset.builder import NeuroMotionDataset
from training.trainer import NeuroMotionLightningModule, get_callbacks
from pytorch_lightning.loggers import WandbLogger

import os
import glob

with open('config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Gather processed tensor files
processed_dir = r"E:\datasets\snd1156-1-1\data\processed"
all_files = glob.glob(os.path.join(processed_dir, "*.pt"))

if not all_files:
    print(f"No processed files found in {processed_dir}. Run scripts/build_dataset.py first.")
    exit(1)

dataset = NeuroMotionDataset(file_paths=all_files, dataset_type="MIXED", mode="ssl", window_size=config['data']['window_size'])
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)

model = NeuroMotionLightningModule(config)

# MLOps Callbacks and Logger
# Monitor InfoNCE loss during SSL pretraining
callbacks = get_callbacks(monitor_metric="train/infonce_loss", mode="min")
wandb_logger = WandbLogger(project="NeuroMotion-ADS", name="SSL-Pretrain")

trainer = pl.Trainer(
    max_epochs=config['training']['max_epochs'], 
    accelerator="auto",
    callbacks=callbacks,
    logger=wandb_logger
)
trainer.fit(model, dataloader)
EOF

python pretrain.py

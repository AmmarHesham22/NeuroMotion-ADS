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
# أضفنا num_workers=2 عشان التحميل يتم بالتوازي
dataloader = DataLoader(
    dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)

model = NeuroMotionLightningModule(config)

# MLOps Callbacks and Logger
# Monitor InfoNCE loss during SSL pretraining
import os
os.environ["WANDB_MODE"] = "offline" # Set to offline for agent execution

callbacks = get_callbacks(monitor_metric="train/infonce_loss", mode="min")
wandb_logger = WandbLogger(project="NeuroMotion-ADS", name="SSL-Pretrain")

trainer = pl.Trainer(
    max_epochs=config['training']['max_epochs'], 
    accelerator="gpu",          # إجبار الكود على استخدام الـ GPU
    devices=1,                  # استخدام GPU واحد
    precision="16-mixed",       # تفعيل سرعة Mixed Precision (سرعة مضاعفة)
    callbacks=callbacks,
    logger=wandb_logger
)
trainer.fit(model, dataloader)

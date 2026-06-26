import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset.builder import NeuroMotionDataset
from dataset.sampler import BalancedBatchSampler  # استدعاء السامبلر المفقود
from training.trainer import NeuroMotionLightningModule, get_callbacks
from pytorch_lightning.loggers import WandbLogger

import os
import glob

with open('config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# تأكد من المسار حسب بيئة Colab
processed_dir = "/content/processed"
all_files = glob.glob(os.path.join(processed_dir, "*.pt"))

if not all_files:
    print(f"No processed files found in {processed_dir}. Run scripts/build_dataset.py first.")
    exit(1)

dataset = NeuroMotionDataset(file_paths=all_files, dataset_type="MIXED", mode="ssl", window_size=config['data']['window_size'])

# 1. فصل الملفات لضمان التوزيع العادل في الباتش
dream_indices = [i for i, f in enumerate(all_files) if "DREAM" in os.path.basename(f)]
pinsoro_indices = [i for i, f in enumerate(all_files) if "PInSoRo" in os.path.basename(f)]

batch_size = config['training']['batch_size']

# 2. تشغيل السامبلر بدلاً من الـ Shuffle العشوائي
sampler = BalancedBatchSampler(
    dream_indices=dream_indices, 
    pinsoro_indices=pinsoro_indices, 
    batch_size=batch_size, 
    dream_ratio=0.5
)

# 3. تعديل الـ DataLoader ليستخدم الـ sampler ولحل مشكلة كراش الكولاب
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler, # نستخدم batch_sampler ولا نستخدم batch_size أو shuffle هنا
    num_workers=2,         # تم التقليل من 4 إلى 2 لتخفيف الضغط على RAM الكولاب
    pin_memory=True
)

model = NeuroMotionLightningModule(config)

os.environ["WANDB_MODE"] = "offline"

callbacks = get_callbacks(monitor_metric="train/infonce_loss", mode="min")
wandb_logger = WandbLogger(project="NeuroMotion-ADS", name="SSL-Pretrain")

trainer = pl.Trainer(
    max_epochs=config['training']['max_epochs'], 
    accelerator="gpu",          
    devices=1,                  
    precision="16-mixed",       
    callbacks=callbacks,
    logger=wandb_logger
)

trainer.fit(model, dataloader)
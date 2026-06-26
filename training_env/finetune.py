import os
import glob
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset.builder import NeuroMotionDataset
from training.trainer import NeuroMotionLightningModule

def main():
    print("Starting NeuroMotion-ADS Fine-tuning Phase (ADOS Prediction)...")
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. تفعيل حساب خسارة التوقع (MSE Loss) وتقليل خسارة الـ SSL
    config['training']['lambda_mse'] = 1.0
    config['training']['lambda_infonce'] = 0.1 

    # 2. تحديد مسار الداتا الخاصة بـ DREAM فقط (لأن PInSoRo لا يحتوي على تقييم ADOS)
    processed_dir = "/content/processed"
    dream_files = glob.glob(os.path.join(processed_dir, "DREAM_*.pt"))
    
    if not dream_files:
        print(f"Error: No DREAM .pt files found in {processed_dir}")
        return

    print(f"Found {len(dream_files)} DREAM files for supervised fine-tuning.")

    # 3. إعداد الـ Dataset في وضع 'supervised'
    dataset = NeuroMotionDataset(
        file_paths=dream_files, 
        dataset_type="DREAM", 
        mode="supervised", 
        window_size=config['data']['window_size']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )

    # 4. تحميل أفضل أوزان من مرحلة الـ Pre-training
    # تأكد من وضع مسار آخر ملف ckpt نتج عن pretrain.py
    # سنفترض أنه موجود في مجلد checkpoints
    ckpt_list = glob.glob("checkpoints/neuromotion-*.ckpt")
    if not ckpt_list:
        print("Warning: No Pre-trained checkpoint found! Training from scratch...")
        model = NeuroMotionLightningModule(config)
    else:
        best_ckpt = max(ckpt_list, key=os.path.getctime)
        print(f"Loading Pre-trained weights from {best_ckpt}")
        model = NeuroMotionLightningModule.load_from_checkpoint(best_ckpt, config=config)
        # تحديث الإعدادات داخل الموديل لتفعيل الـ MSE
        model.lambda_mse = 1.0
        model.lambda_infonce = 0.1

    # 5. إعداد المراقبة والحفظ
    os.environ["WANDB_MODE"] = "offline"
    wandb_logger = WandbLogger(project="NeuroMotion-ADS", name="Supervised-Finetune")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/mse_loss", # المراقبة الآن على خسارة التوقع بدلاً من InfoNCE
        dirpath="finetune_checkpoints",
        filename="neuromotion-finetuned-{epoch:02d}-{val/mse_loss:.4f}",
        save_top_k=2,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator="gpu", 
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    
    # بدء التدريب الدقيق
    trainer.fit(model, dataloader)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
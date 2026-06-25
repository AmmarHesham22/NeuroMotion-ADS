import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.heads import NeuroMotionModel
from training.losses import InfoNCELoss
from preprocessing.graph_utils import build_adjacency_matrix
from training.metrics import compute_pearson_r, compute_r2

class NeuroMotionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for joint optimization of the NeuroMotion-ADS model.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Core Model
        self.model = NeuroMotionModel(config)
        
        # Losses
        self.infonce_loss = InfoNCELoss(temperature=config['training']['ssl_temperature'])
        self.mse_loss = nn.MSELoss()
        
        # Loss weights
        self.lambda_infonce = config['training']['lambda_infonce']
        self.lambda_mse = config['training']['lambda_mse']
        self.lambda_reg = config['training']['lambda_reg']
        
        # Static edge index for the graph
        edge_index = build_adjacency_matrix(config['data']['num_joints'], self_loops=True)
        self.register_buffer("edge_index", edge_index)

    def forward(self, skeleton, gaze):
        return self.model(skeleton, gaze, self.edge_index)

    def training_step(self, batch, batch_idx):
        loss = 0.0
        
        # Determine mode based on batch contents
        if "v1" in batch and "v2" in batch:
            # SSL Mode
            v1, v2 = batch["v1"], batch["v2"]
            
            out1 = self(v1["skeleton"], v1["gaze"])
            out2 = self(v2["skeleton"], v2["gaze"])
            
            l_infonce = self.infonce_loss(out1["z"], out2["z"])
            loss += self.lambda_infonce * l_infonce
            self.log("train/infonce_loss", l_infonce, prog_bar=True)
            
        if "skeleton" in batch and self.lambda_mse > 0.0:
            # Supervised / Fine-tuning Mode
            out = self(batch["skeleton"], batch["gaze"])
            
            target = batch["target"]
            # Filter out unsupervised (-1.0) samples
            valid_idx = target >= 0
            
            if valid_idx.any():
                pred = out["ados_pred"][valid_idx]
                target_valid = target[valid_idx]
                
                l_mse = self.mse_loss(pred, target_valid)
                loss += self.lambda_mse * l_mse
                self.log("train/mse_loss", l_mse, prog_bar=True)
                
        # L2 Regularization
        l2_reg = sum(p.norm(2) for p in self.parameters())
        loss += self.lambda_reg * l2_reg
        self.log("train/l2_reg", l2_reg)
        
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if "skeleton" in batch:
            out = self(batch["skeleton"], batch["gaze"])
            target = batch["target"]
            
            valid_idx = target >= 0
            if valid_idx.any():
                pred = out["ados_pred"][valid_idx]
                target_valid = target[valid_idx]
                
                val_mse = self.mse_loss(pred, target_valid)
                self.log("val/mse_loss", val_mse, prog_bar=True)
                
                r_val = compute_pearson_r(target_valid, pred)
                r2_val = compute_r2(target_valid, pred)
                
                self.log("val/pearson_r", r_val, prog_bar=True)
                self.log("val/r2", r2_val, prog_bar=True)
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['training']['max_epochs'])
        return [optimizer], [scheduler]

def get_callbacks(monitor_metric="val/mse_loss", mode="min"):
    """
    Production-grade callbacks for model checkpointing and early stopping.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath="checkpoints",
        filename="neuromotion-{epoch:02d}-{" + monitor_metric.replace('/', '_') + ":.4f}",
        save_top_k=3,
        mode=mode,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode=mode
    )
    
    return [checkpoint_callback, early_stop_callback]

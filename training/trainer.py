import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from data_utils.brats_dataset import stack_modalities, VOL_SIZE
from models.pe_modules import make_coord_channels


REGION_NAMES = ["WT", "TC", "ET"]


def _prepare_batch(batch, device, pe_type):
    x  = stack_modalities(batch).to(device)          # (B, 4, H, W, D)
    y  = batch["seg"].to(device).float()              # (B, 3, H, W, D) — float for loss
    pc = batch.get("patch_center")
    cc = batch.get("coord_channels")
    if pc is not None:
        pc = pc.to(device)
    if cc is not None:
        cc = cc.to(device)
    return x, y, pc, cc


class Trainer:
    def __init__(self, model: nn.Module, cfg: dict, experiment_name: str,
                 device: torch.device):
        self.model = model.to(device)
        self.cfg   = cfg
        self.name  = experiment_name
        self.device = device
        self.pe_type = cfg.get("pe", {}).get("type", "none")

        train_cfg = cfg.get("training", {})
        self.max_epochs   = train_cfg.get("max_epochs", 300)
        self.patience     = train_cfg.get("early_stopping_patience", 50)
        self.grad_clip    = train_cfg.get("grad_clip", 1.0)
        self.use_amp      = train_cfg.get("mixed_precision", True)
        self.save_every   = cfg.get("logging", {}).get("save_every_n_epochs", 50)

        self.ckpt_dir   = Path(cfg.get("logging", {}).get("checkpoint_dir", "checkpoints"))
        self.tb_dir     = Path(cfg.get("logging", {}).get("tensorboard_dir", "experiments/logs"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        loss_cfg = train_cfg
        self.criterion = DiceCELoss(
            sigmoid=True,
            batch=True,
            squared_pred=True,
            lambda_dice=loss_cfg.get("loss_dice_weight", 0.5),
            lambda_ce=1.0 - loss_cfg.get("loss_dice_weight", 0.5),
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )
        self.scaler = GradScaler("cpu" if not torch.cuda.is_available() else "cuda",
                                  enabled=self.use_amp)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir / experiment_name))

        self.best_val_dice   = -1.0
        self.no_improve_cnt  = 0
        self.start_epoch     = 0

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_val_dice  = ckpt.get("best_val_dice", -1.0)
        self.start_epoch    = ckpt["epoch"] + 1
        print(f"Resumed from {path} at epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch: int, tag: str):
        path = self.ckpt_dir / f"{self.name}_{tag}.pth"
        torch.save({
            "epoch":         epoch,
            "state_dict":    self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "cfg":           self.cfg,
        }, path)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            x, y, pc, cc = _prepare_batch(batch, self.device, self.pe_type)
            self.optimizer.zero_grad(set_to_none=True)
            _device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(_device_type, enabled=self.use_amp):
                logits = self.model(x, patch_center=pc, coord_channels=cc)
                loss   = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        metric = DiceMetric(include_background=True, reduction="mean_batch",
                            get_not_nans=False)
        for batch in loader:
            x, y, pc, cc = _prepare_batch(batch, self.device, self.pe_type)
            # Val volumes are not randomly cropped, so PE metadata is absent.
            # Use volume-centre as a fallback; good enough for early stopping.
            if self.pe_type in ("film", "concat") and pc is None:
                B = x.shape[0]
                pc = torch.full((B, 3), 0.5, dtype=torch.float32, device=self.device)
            if self.pe_type == "concat" and cc is None:
                cc = make_coord_channels(pc, x.shape[2:], VOL_SIZE).to(self.device)
            _device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(_device_type, enabled=self.use_amp):
                logits = self.model(x, patch_center=pc, coord_channels=cc)
            pred = (logits.sigmoid() > 0.5).long()
            metric(y_pred=pred, y=y.long())
        scores = metric.aggregate().tolist()
        metric.reset()
        return {r: scores[i] for i, r in enumerate(REGION_NAMES)}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*60}")
        print(f"Experiment: {self.name}")
        print(f"Max epochs: {self.max_epochs}  |  Early stop patience: {self.patience}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.max_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_dice   = self._val_epoch(val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0

            mean_dice = sum(val_dice.values()) / len(val_dice)
            print(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"Loss {train_loss:.4f} | "
                f"Val WT {val_dice['WT']:.3f} TC {val_dice['TC']:.3f} "
                f"ET {val_dice['ET']:.3f} Mean {mean_dice:.3f} | "
                f"{elapsed:.0f}s"
            )

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Dice/val_mean", mean_dice, epoch)
            for r, v in val_dice.items():
                self.writer.add_scalar(f"Dice/val_{r}", v, epoch)

            if mean_dice > self.best_val_dice:
                self.best_val_dice = mean_dice
                self.no_improve_cnt = 0
                self._save_checkpoint(epoch, "best")
                print(f"  → New best val Dice: {mean_dice:.4f}")
            else:
                self.no_improve_cnt += 1

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, f"epoch{epoch:03d}")

            if self.no_improve_cnt >= self.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs).")
                break

        self.writer.close()
        print(f"\nTraining complete. Best val Dice: {self.best_val_dice:.4f}")
        print(f"Best checkpoint: {self.ckpt_dir / (self.name + '_best.pth')}")

"""
Trainer — handles the staged training loop.

Stage 1 : GSplat supervision + EMA distillation  (no geometry)
Stage 2 : + reprojection + multi-view losses
Stage 3 : end-to-end, L_feat turned off
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm

from models.ddnet import DDNet
from training.losses import LossManager


class Trainer:
    """
    Args:
        cfg        : full OmegaConf config
        model      : DDNet instance
        train_loader, val_loader : DataLoaders
        device     : 'cuda' or 'cpu'
        log_dir    : tensorboard log directory
    """

    def __init__(self, cfg, model, train_loader, val_loader, device="cuda", log_dir="runs/"):
        self.cfg          = cfg
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.writer       = SummaryWriter(log_dir)
        self.loss_mgr     = LossManager(cfg.training, model.encoder.out_channels).to(device)
        self.global_step  = 0

    def _make_optimizer(self, lr: float) -> optim.Optimizer:
        return optim.AdamW(
            list(self.model.parameters()) + list(self.loss_mgr.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

    def _run_epoch(self, stage_cfg, optimizer, run_geometry: bool):
        """One epoch of training."""
        self.model.train()
        active = list(stage_cfg.active_losses)
        total_loss = 0.0

        for batch in tqdm(self.train_loader, leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            pred = self.model(
                batch["img_a"], batch["img_b"],
                batch["K_a"],  batch["K_b"],
                run_geometry=run_geometry,
            )

            losses = self.loss_mgr(pred, batch, active)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
            )
            optimizer.step()
            self.model.update_ema()          # EMA update after every step

            # logging
            for name, val in losses.items():
                self.writer.add_scalar(f"train/{name}", val.item(), self.global_step)
            total_loss += losses["total"].item()
            self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self, stage_cfg, run_geometry: bool) -> float:
        self.model.eval()
        active = list(stage_cfg.active_losses)
        total = 0.0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = self.model(
                batch["img_a"], batch["img_b"],
                batch["K_a"],  batch["K_b"],
                run_geometry=run_geometry,
            )
            losses = self.loss_mgr(pred, batch, active)
            total += losses["total"].item()
        return total / len(self.val_loader)

    def train(self):
        """Run all three training stages in sequence."""
        stages = [
            ("stage1", self.cfg.training.stage1, False),
            ("stage2", self.cfg.training.stage2, True),
            ("stage3", self.cfg.training.stage3, True),
        ]

        for stage_name, stage_cfg, run_geometry in stages:
            print(f"\n{'='*50}\n  {stage_name.upper()}\n{'='*50}")
            optimizer = self._make_optimizer(stage_cfg.lr)

            for epoch in range(stage_cfg.epochs):
                train_loss = self._run_epoch(stage_cfg, optimizer, run_geometry)
                val_loss   = self._validate(stage_cfg, run_geometry)

                self.writer.add_scalar(f"{stage_name}/train_loss_epoch", train_loss, epoch)
                self.writer.add_scalar(f"{stage_name}/val_loss_epoch",   val_loss,   epoch)
                print(f"  epoch {epoch+1:3d}/{stage_cfg.epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")

            # save checkpoint after each stage
            torch.save({
                "model":       self.model.state_dict(),
                "loss_mgr":    self.loss_mgr.state_dict(),
                "stage":       stage_name,
                "global_step": self.global_step,
            }, f"checkpoint_{stage_name}.pt")
            print(f"  saved checkpoint_{stage_name}.pt")

        self.writer.close()

"""
Trainer — staged training loop with full logging.

Writes:
  runs/<name>/training_log.json   — full loss history (every step + epoch)
  runs/<name>/training_log.csv    — epoch-level CSV for quick inspection
  checkpoint_stage1.pt etc.       — model weights + full training state
"""

import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.ddnet  import DDNet
from training.losses         import LossManager
from utils.logger  import TrainingLogger


class Trainer:
    """
    Args:
        cfg          : full OmegaConf config
        model        : DDNet instance
        train_loader : DataLoader
        val_loader   : DataLoader
        device       : 'cuda' or 'cpu'
        run_name     : experiment name (used for log directory)
    """

    def __init__(
        self,
        cfg,
        model: DDNet,
        train_loader,
        val_loader,
        device: str = "cuda",
        run_name: str = "exp",
    ):
        self.cfg          = cfg
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        self.log_dir  = f"runs/{run_name}"
        self.writer   = SummaryWriter(self.log_dir)
        self.logger   = TrainingLogger(self.log_dir)
        self.loss_mgr = LossManager(
            cfg.training, model.encoder.out_channels
        ).to(device)

        self.global_step = 0

    # ── Optimizer ─────────────────────────────────────────────

    def _make_optimizer(self, lr: float) -> optim.Optimizer:
        params = (list(self.model.parameters()) +
                  list(self.loss_mgr.parameters()))
        return optim.AdamW(params, lr=lr, weight_decay=1e-4)

    # ── Training epoch ────────────────────────────────────────

    def _run_epoch(
        self,
        stage_cfg,
        optimizer: optim.Optimizer,
        run_geometry: bool,
        stage_name: str,
    ) -> dict:
        """
        One training epoch.
        Returns dict of mean loss values for the epoch.
        """
        self.model.train()
        active = list(stage_cfg.active_losses)

        epoch_totals = {}
        n_batches    = 0

        for batch in tqdm(self.train_loader, leave=False):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            pred   = self.model(
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
            self.model.update_ema()

            # ── log every step ──
            step_losses = {k: v.item() for k, v in losses.items()}
            self.logger.log_step(self.global_step, stage_name, step_losses)

            # tensorboard
            for k, v in step_losses.items():
                self.writer.add_scalar(f"step/{k}", v, self.global_step)

            # accumulate for epoch mean
            for k, v in step_losses.items():
                epoch_totals[k] = epoch_totals.get(k, 0.0) + v
            n_batches    += 1
            self.global_step += 1

        return {k: v / n_batches for k, v in epoch_totals.items()}

    # ── Validation epoch ──────────────────────────────────────

    @torch.no_grad()
    def _validate(self, stage_cfg, run_geometry: bool) -> dict:
        self.model.eval()
        active = list(stage_cfg.active_losses)

        totals    = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            pred   = self.model(
                batch["img_a"], batch["img_b"],
                batch["K_a"],  batch["K_b"],
                run_geometry=run_geometry,
            )
            losses = self.loss_mgr(pred, batch, active)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ── Full training ─────────────────────────────────────────

    def train(self):
        """Run all three training stages."""
        stages = [
            ("stage1", self.cfg.training.stage1, False),
            ("stage2", self.cfg.training.stage2, True),
            ("stage3", self.cfg.training.stage3, True),
        ]

        for stage_name, stage_cfg, run_geometry in stages:
            print(f"\n{'='*52}\n  {stage_name.upper()}"
                  f"  ({stage_cfg.epochs} epochs)\n{'='*52}")
            optimizer = self._make_optimizer(stage_cfg.lr)

            for epoch in range(stage_cfg.epochs):
                t0 = time.time()

                train_losses = self._run_epoch(
                    stage_cfg, optimizer, run_geometry, stage_name
                )
                val_losses   = self._validate(stage_cfg, run_geometry)

                epoch_time = time.time() - t0

                # ── log epoch ──
                self.logger.log_epoch(
                    epoch       = epoch + 1,
                    stage       = stage_name,
                    train_loss  = train_losses["total"],
                    val_loss    = val_losses["total"],
                    epoch_time  = epoch_time,
                    extra       = {
                        f"train_{k}": v
                        for k, v in train_losses.items()
                        if k != "total"
                    },
                )

                # tensorboard epoch-level
                self.writer.add_scalars(
                    f"{stage_name}/loss",
                    {"train": train_losses["total"],
                     "val":   val_losses["total"]},
                    epoch,
                )
                for k, v in train_losses.items():
                    self.writer.add_scalar(
                        f"{stage_name}/train_{k}", v, epoch
                    )

            # ── checkpoint after each stage ──
            ckpt_path = f"checkpoint_{stage_name}.pt"
            torch.save({
                "model":       self.model.state_dict(),
                "loss_mgr":    self.loss_mgr.state_dict(),
                "stage":       stage_name,
                "global_step": self.global_step,
                "cfg":         self.cfg,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

        # ── final summary ──
        summary = self.logger.summary()
        print(f"\nTraining complete.")
        print(f"  Total epochs : {summary['total_epochs']}")
        print(f"  Total steps  : {summary['total_steps']}")
        print(f"  Best val loss: {summary['best_val_loss']:.4f} "
              f"(epoch {summary['best_epoch']})")
        print(f"\nLog files:")
        print(f"  {self.log_dir}/training_log.json")
        print(f"  {self.log_dir}/training_log.csv")
        print(f"\nGenerate plots:")
        print(f"  python -m utils.plotting --log {self.log_dir}/training_log.json")

        self.writer.close()

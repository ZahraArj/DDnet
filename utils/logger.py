"""
Training logger.

Saves every loss value to:
  - training_log.json   (full history, append-only, never lost)
  - training_log.csv    (same data, easy to open in Excel/pandas)

Also tracks best validation loss and training time per epoch.

Usage inside Trainer:
    logger = TrainingLogger(log_dir="runs/exp1")
    logger.log_step(step, stage, losses_dict)       # called every batch
    logger.log_epoch(epoch, stage, train_loss, val_loss, epoch_time)
    logger.save()                                   # writes JSON + CSV
"""

import json
import csv
import time
import os
from pathlib import Path


class TrainingLogger:
    """
    Logs all training metrics to JSON and CSV files.

    Args:
        log_dir : directory to write log files into
    """

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.log_dir / "training_log.json"
        self.csv_path  = self.log_dir / "training_log.csv"

        # In-memory history
        self.step_log  = []   # every batch: {step, stage, loss_name: value, ...}
        self.epoch_log = []   # every epoch: {epoch, stage, train_loss, val_loss, time_s}

        # Best val loss tracking
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

        # Load existing log if resuming
        if self.json_path.exists():
            with open(self.json_path) as f:
                data = json.load(f)
                self.step_log  = data.get("steps", [])
                self.epoch_log = data.get("epochs", [])
            print(f"[Logger] Resumed from {self.json_path} "
                  f"({len(self.epoch_log)} epochs already logged)")

    def log_step(self, step: int, stage: str, losses: dict):
        """
        Log all loss values for one training step.

        Args:
            step   : global step counter
            stage  : 'stage1', 'stage2', 'stage3'
            losses : dict of {loss_name: float_value}
        """
        entry = {"step": step, "stage": stage}
        for k, v in losses.items():
            entry[k] = round(float(v), 6)
        self.step_log.append(entry)

    def log_epoch(
        self,
        epoch: int,
        stage: str,
        train_loss: float,
        val_loss: float,
        epoch_time: float,
        extra: dict = None,
    ):
        """
        Log per-epoch summary.

        Args:
            epoch      : epoch index within the stage
            stage      : stage name
            train_loss : mean training loss for the epoch
            val_loss   : mean validation loss for the epoch
            epoch_time : wall-clock seconds for the epoch
            extra      : optional dict of extra metrics (e.g. depth AbsRel)
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch    = epoch

        entry = {
            "epoch":      epoch,
            "stage":      stage,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "time_s":     round(epoch_time, 2),
            "is_best":    is_best,
        }
        if extra:
            for k, v in extra.items():
                entry[k] = round(float(v), 6)

        self.epoch_log.append(entry)
        self.save()   # write to disk after every epoch

        status = " ← best" if is_best else ""
        print(f"  [{stage}] epoch {epoch:3d}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"time={epoch_time:.1f}s{status}")

    def save(self):
        """Write full log to JSON and CSV."""
        # JSON
        with open(self.json_path, "w") as f:
            json.dump({"steps": self.step_log, "epochs": self.epoch_log},
                      f, indent=2)

        # CSV (epoch-level only — step log can be huge)
        if self.epoch_log:
            fieldnames = list(self.epoch_log[0].keys())
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.epoch_log)

    def summary(self) -> dict:
        """Return a summary dict (printed at end of training)."""
        return {
            "total_epochs":    len(self.epoch_log),
            "total_steps":     len(self.step_log),
            "best_val_loss":   self.best_val_loss,
            "best_epoch":      self.best_epoch,
        }

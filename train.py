"""
Entry point.

Usage:
    python train.py                          # uses configs/default.yaml
    python train.py model.vit_arch=vit_small_patch16_224
    python train.py training.batch_size=4
"""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from models.ddnet      import DDNet
from training.trainer  import Trainer
from data.dataset      import DummyDataset   # swap for your real dataset


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset ──
    # Replace DummyDataset with your real dataset subclass
    train_ds = DummyDataset(length=256,
                            img_h=cfg.data.img_height,
                            img_w=cfg.data.img_width,
                            desc_dim=cfg.model.desc_dim)
    val_ds   = DummyDataset(length=64,
                            img_h=cfg.data.img_height,
                            img_w=cfg.data.img_width,
                            desc_dim=cfg.model.desc_dim)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.training.batch_size,
                              shuffle=True,
                              num_workers=cfg.training.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,
                              batch_size=cfg.training.batch_size,
                              shuffle=False,
                              num_workers=cfg.training.num_workers)

    # ── Model ──
    model = DDNet(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

    # ── Train ──
    trainer = Trainer(cfg, model, train_loader, val_loader, device=device)
    trainer.train()


if __name__ == "__main__":
    main()

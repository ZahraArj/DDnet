"""
Miscellaneous utilities.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> dict:
    """Return trainable and total parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_checkpoint(path: str, model, optimizer, epoch: int, loss: float, cfg=None):
    torch.save({
        "epoch":     epoch,
        "loss":      loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg":       cfg,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("loss", None)

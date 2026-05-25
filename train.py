"""
train.py  —  entry point with real data

Usage:
    # point to a folder of .npz files
    python train.py data.train_dir=data/train data.val_dir=data/val

    # or point to text-file lists
    python train.py data.train_list=data/train.txt data.val_list=data/val.txt

    # override descriptor dim if yours is 32 not 128
    python train.py model.desc_dim=32
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from models.ddnet      import DDNet
from training.trainer  import Trainer
from data.dataset      import NPZPairDataset, DummyDataset


@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(OmegaConf.to_yaml(cfg))
    print(f'Device: {device}')

    # ── choose data source ──────────────────────────────────
    # Option A: directory of .npz files
    if OmegaConf.select(cfg, 'data.train_dir'):
        train_ds = NPZPairDataset(
            source   = cfg.data.train_dir,
            img_h    = cfg.data.img_height,
            img_w    = cfg.data.img_width,
            desc_dim = cfg.model.desc_dim,
            split    = 'train',
        )
        val_ds = NPZPairDataset(
            source   = cfg.data.val_dir,
            img_h    = cfg.data.img_height,
            img_w    = cfg.data.img_width,
            desc_dim = cfg.model.desc_dim,
            split    = 'val',
        )

    # Option B: text-file lists
    elif OmegaConf.select(cfg, 'data.train_list'):
        train_ds = NPZPairDataset(
            source   = cfg.data.train_list,
            img_h    = cfg.data.img_height,
            img_w    = cfg.data.img_width,
            desc_dim = cfg.model.desc_dim,
            split    = 'train',
        )
        val_ds = NPZPairDataset(
            source   = cfg.data.val_list,
            img_h    = cfg.data.img_height,
            img_w    = cfg.data.img_width,
            desc_dim = cfg.model.desc_dim,
            split    = 'val',
        )

    # Option C: dummy data (fallback for testing)
    else:
        print('WARNING: no data source specified — using DummyDataset')
        train_ds = DummyDataset(length=256, img_h=cfg.data.img_height,
                                img_w=cfg.data.img_width,
                                desc_dim=cfg.model.desc_dim)
        val_ds   = DummyDataset(length=64,  img_h=cfg.data.img_height,
                                img_w=cfg.data.img_width,
                                desc_dim=cfg.model.desc_dim)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = True,
        num_workers = cfg.training.num_workers,
        pin_memory  = (device == 'cuda'),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = False,
        num_workers = cfg.training.num_workers,
        pin_memory  = (device == 'cuda'),
    )

    print(f'Train pairs : {len(train_ds)}')
    print(f'Val   pairs : {len(val_ds)}')

    model = DDNet(cfg)
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n/1e6:.1f}M')

    trainer = Trainer(cfg, model, train_loader, val_loader,
                      device=device, run_name='ddnet_real')
    trainer.train()


if __name__ == '__main__':
    main()

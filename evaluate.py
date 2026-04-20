"""
Evaluation script.

Usage:
    python evaluate.py --checkpoint checkpoint_stage3.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from models.ddnet        import DDNet
from data.dataset        import DummyDataset
from utils.geometry_utils import pose_error, scale_invariant_depth_error


def evaluate(checkpoint_path: str, device: str = "cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["cfg"]

    model = DDNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = DummyDataset(length=128,
                           img_h=cfg.data.img_height,
                           img_w=cfg.data.img_width,
                           desc_dim=cfg.model.desc_dim)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    depth_metrics = {"silog": [], "abs_rel": [], "rmse": []}
    pose_metrics  = {"rot_deg": [], "trans_deg": []}

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model(batch["img_a"], batch["img_b"],
                         batch["K_a"],  batch["K_b"],
                         run_geometry=True)

            # depth
            dm = scale_invariant_depth_error(pred["Za"], batch["Z_gt_a"], batch.get("mask_a"))
            for k in depth_metrics:
                depth_metrics[k].append(dm[k].item())

    # aggregate
    print("\n── Depth evaluation ──")
    for k, vals in depth_metrics.items():
        print(f"  {k:10s}: {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.device)

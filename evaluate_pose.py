"""
Pose and reprojection evaluation script.

Usage:
    # evaluate on val set + save all figures
    python evaluate_pose.py --checkpoint checkpoint_stage3.pt --out results/

    # quick test with DummyDataset (no real data needed)
    python evaluate_pose.py --checkpoint checkpoint_stage3.pt --dummy --out results/
"""

import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pathlib import Path

from models.ddnet        import DDNet
from data.dataset        import DummyDataset
from utils.eval_viz      import ReproViz, PoseEval


def evaluate(checkpoint_path: str, out_dir: str, dummy: bool,
             device: str, n_reproj_batches: int = 2):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── load model ──
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg")
    if cfg is None:
        raise ValueError("Checkpoint has no cfg. Re-train with updated trainer.py.")

    model = DDNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {checkpoint_path}  (stage: {ckpt.get('stage','?')})")

    # ── dataset ──
    # DummyDataset generates random R_gt and t_gt so pose eval works
    # Replace with your real dataset subclass for meaningful numbers
    ds = DummyDataset(
        length=128,
        img_h=cfg.data.img_height,
        img_w=cfg.data.img_width,
        desc_dim=cfg.model.desc_dim,
        with_pose_gt=True,      # see note below
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    # ── 1. Reprojection visualisation ──
    print("\n── Reprojection visualisation ──")
    viz    = ReproViz(model, device)
    sample = next(iter(loader))
    fig    = viz.visualise(sample, n_pts=200, max_pairs=4,
                           save_path=str(out / "reprojection.png"))
    print(f"Saved reprojection.png")

    # ── 2. Pose accuracy ──
    print("\n── Pose accuracy ──")
    pe = PoseEval(model, device)
    pe.run(loader)
    pe.print_summary()
    pe.plot(save_path=str(out / "pose_accuracy.png"))
    print(f"Saved pose_accuracy.png")

    # save summary JSON
    import json
    with open(out / "pose_summary.json", "w") as f:
        json.dump(pe.summary(), f, indent=2)
    print(f"Saved pose_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out",        default="results")
    parser.add_argument("--dummy",      action="store_true")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.out, args.dummy, args.device)

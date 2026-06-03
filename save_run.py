"""
save_run.py — save a completed training run into a named folder

Copies:
  - checkpoints (checkpoint_stage*.pt)
  - training log (training_log.json, training_log.csv)
  - config (configs/default.yaml)
  - plots (runs/<run>/plots/)
  - evaluation results (results/)

Then runs all evaluations inside the saved folder.

Usage:
    python save_run.py --name my_first_real_run
    python save_run.py --name desc32_480x640_50epochs --eval
    python save_run.py --name exp_001 --eval --checkpoint stage3
"""

import argparse
import os
import shutil
import json
from pathlib import Path
from datetime import datetime


# ── Defaults ──────────────────────────────────────────────────
DEFAULT_RUN_DIR   = "runs/ddnet_real"
DEFAULT_CKPT_DIR  = "runs/ddnet_real"   # where trainer saves checkpoints
DEFAULT_SAVED_DIR = "saved_runs"         # where named runs are stored


def copy_run(name: str,
             run_dir: str  = DEFAULT_RUN_DIR,
             ckpt_dir: str = DEFAULT_CKPT_DIR,
             saved_dir: str = DEFAULT_SAVED_DIR) -> Path:
    """
    Copy all artefacts from the current training run into
    saved_runs/<name>/  and return the destination path.
    """
    src_run  = Path(run_dir)
    src_ckpt = Path(ckpt_dir)
    dst      = Path(saved_dir) / name

    if dst.exists():
        ans = input(f"\n'{dst}' already exists. Overwrite? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            raise SystemExit(0)
        shutil.rmtree(dst)

    dst.mkdir(parents=True)
    print(f"\nSaving run '{name}' to {dst}/\n")

    # ── training log ──────────────────────────────────────────
    log_dir = dst / "logs"
    log_dir.mkdir()
    for fname in ["training_log.json", "training_log.csv"]:
        src = src_run / fname
        if src.exists():
            shutil.copy2(src, log_dir / fname)
            print(f"  copied {fname}")
        else:
            print(f"  WARNING: {src} not found — skipping")

    # ── checkpoints ───────────────────────────────────────────
    ckpt_dst = dst / "checkpoints"
    ckpt_dst.mkdir()
    found = list(src_ckpt.glob("checkpoint_stage*.pt"))
    if not found:
        # also check inside run dir
        found = list(src_run.glob("checkpoint_stage*.pt"))
    if found:
        for f in sorted(found):
            shutil.copy2(f, ckpt_dst / f.name)
            size_mb = f.stat().st_size / 1e6
            print(f"  copied {f.name}  ({size_mb:.0f} MB)")
    else:
        print("  WARNING: no checkpoint_stage*.pt files found")

    # ── config ────────────────────────────────────────────────
    cfg_src = Path("configs/default.yaml")
    if cfg_src.exists():
        cfg_dst = dst / "config"
        cfg_dst.mkdir()
        shutil.copy2(cfg_src, cfg_dst / "default.yaml")
        print(f"  copied configs/default.yaml")

    # ── plots (if already generated) ──────────────────────────
    plots_src = src_run / "plots"
    if plots_src.exists():
        shutil.copytree(plots_src, dst / "plots")
        n = len(list((dst / "plots").glob("*.png")))
        print(f"  copied plots/  ({n} images)")

    # ── previous results (if any) ─────────────────────────────
    results_src = Path("results")
    if results_src.exists():
        shutil.copytree(results_src, dst / "results_old",
                        dirs_exist_ok=True)
        print(f"  copied results/  → results_old/")

    # ── write metadata ────────────────────────────────────────
    meta = {
        "name":       name,
        "saved_at":   datetime.now().isoformat(),
        "source_run": str(src_run.resolve()),
        "checkpoints": [f.name for f in sorted(ckpt_dst.glob("*.pt"))],
    }

    # read best val loss from log if available
    log_json = log_dir / "training_log.json"
    if log_json.exists():
        with open(log_json) as f:
            data = json.load(f)
        epochs = data.get("epochs", [])
        if epochs:
            best = min(epochs, key=lambda e: e["val_loss"])
            meta["best_val_loss"] = best["val_loss"]
            meta["best_epoch"]    = best["epoch"]
            meta["total_epochs"]  = len(epochs)
            meta["stages"]        = list(dict.fromkeys(
                e["stage"] for e in epochs))

    with open(dst / "run_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote run_info.json")

    return dst


def run_evaluations(dst: Path, checkpoint: str = "stage3"):
    """
    Run all evaluation scripts and save results inside dst/results/
    """
    results_dir = dst / "results"
    results_dir.mkdir(exist_ok=True)

    ckpt_path = dst / "checkpoints" / f"checkpoint_{checkpoint}.pt"
    if not ckpt_path.exists():
        # try other stages
        for stage in ["stage3", "stage2", "stage1"]:
            p = dst / "checkpoints" / f"checkpoint_{stage}.pt"
            if p.exists():
                ckpt_path = p
                print(f"  using checkpoint_{stage}.pt")
                break
        else:
            print("  ERROR: no checkpoint found — skipping evaluations")
            return

    print(f"\nRunning evaluations with {ckpt_path.name}...")
    print(f"Results will be saved to {results_dir}/\n")

    # ── 1. training plots ──────────────────────────────────────
    log_json = dst / "logs" / "training_log.json"
    if log_json.exists():
        plots_dir = dst / "plots"
        print("  generating training plots...")
        ret = os.system(
            f"python eval_metrics.py "
            f"--log {log_json} "
            f"--out {plots_dir}"
        )
        if ret == 0:
            print(f"  saved to {plots_dir}/")
        else:
            print("  WARNING: eval_metrics.py failed")

    # ── 2. depth evaluation ────────────────────────────────────
    print("\n  running depth evaluation...")
    depth_log = results_dir / "depth_metrics.txt"
    ret = os.system(
        f"python evaluate.py "
        f"--checkpoint {ckpt_path} "
        f"2>&1 | tee {depth_log}"
    )
    if ret == 0:
        print(f"  saved to {depth_log}")

    # ── 3. pose evaluation + reprojection visualisation ────────
    print("\n  running pose evaluation...")
    ret = os.system(
        f"python evaluate_pose.py "
        f"--checkpoint {ckpt_path} "
        f"--out {results_dir}"
    )
    if ret == 0:
        print(f"  saved to {results_dir}/")

    # ── summary ────────────────────────────────────────────────
    print(f"\nEvaluation complete. Results in {results_dir}/")
    print("  Files:")
    for f in sorted(results_dir.iterdir()):
        size = f.stat().st_size / 1e3
        print(f"    {f.name}  ({size:.0f} KB)")


def print_summary(dst: Path):
    info_path = dst / "run_info.json"
    if not info_path.exists():
        return
    with open(info_path) as f:
        info = json.load(f)

    print(f"\n── Run summary: {info['name']} ──────────────────────")
    print(f"  Saved at      : {info['saved_at']}")
    if "total_epochs" in info:
        print(f"  Total epochs  : {info['total_epochs']}")
    if "stages" in info:
        print(f"  Stages        : {info['stages']}")
    if "best_val_loss" in info:
        print(f"  Best val loss : {info['best_val_loss']:.4f}"
              f"  (epoch {info['best_epoch']})")
    print(f"  Checkpoints   : {info.get('checkpoints', [])}")
    print(f"  Location      : {dst}/")
    print(f"\nTo run evaluations later:")
    print(f"  python save_run.py --name {info['name']} "
          f"--eval --skip_copy")


def main():
    parser = argparse.ArgumentParser(
        description='Save a completed DD-Net training run to a named folder')

    parser.add_argument('--name',       required=True,
                        help='Name for this run  (e.g. desc32_50epochs)')
    parser.add_argument('--eval',       action='store_true',
                        help='Run all evaluations after saving')
    parser.add_argument('--skip_copy',  action='store_true',
                        help='Skip copying files — only run evaluations '
                             'on an already-saved run')
    parser.add_argument('--checkpoint', default='stage3',
                        choices=['stage1', 'stage2', 'stage3'],
                        help='Which checkpoint to use for evaluation')
    parser.add_argument('--run_dir',    default=DEFAULT_RUN_DIR,
                        help=f'Training run directory  '
                             f'(default: {DEFAULT_RUN_DIR})')
    parser.add_argument('--saved_dir',  default=DEFAULT_SAVED_DIR,
                        help=f'Where to store saved runs  '
                             f'(default: {DEFAULT_SAVED_DIR})')
    args = parser.parse_args()

    dst = Path(args.saved_dir) / args.name

    if not args.skip_copy:
        dst = copy_run(
            name       = args.name,
            run_dir    = args.run_dir,
            ckpt_dir   = args.run_dir,
            saved_dir  = args.saved_dir,
        )
    else:
        if not dst.exists():
            print(f"ERROR: {dst} does not exist. "
                  f"Run without --skip_copy first.")
            raise SystemExit(1)
        print(f"Skipping copy — using existing {dst}/")

    if args.eval:
        run_evaluations(dst, checkpoint=args.checkpoint)

    print_summary(dst)


if __name__ == '__main__':
    main()

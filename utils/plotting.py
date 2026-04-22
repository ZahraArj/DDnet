"""
Training analysis and plotting.

Reads training_log.json and produces a set of publication-quality plots:

  1. Total loss curves (train + val per stage)
  2. Individual loss components over training
  3. Loss breakdown bar chart (final epoch contribution per term)
  4. Training time per epoch
  5. Stage transition markers
  6. Gradient norm (if logged)

Usage:
    python utils/plotting.py --log runs/exp1/training_log.json
    python utils/plotting.py --log runs/exp1/training_log.json --out figures/
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Style ───────────────────────────────────────────────────────
STYLE = {
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    
    "grid.linestyle":     "--",
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "legend.fontsize":    8,
    "legend.framealpha":  0.8,
    "lines.linewidth":    1.8,
}
plt.rcParams.update(STYLE)

# Color palette — one per loss term
LOSS_COLORS = {
    "total":       "#111827",
    "train_loss":  "#111827",
    "val_loss":    "#6366F1",
    "depth_sup":   "#059669",
    "desc_sup":    "#10B981",
    "feat":        "#7C3AED",
    "reproj":      "#D97706",
    "depth_mv":    "#F59E0B",
    "desc_mv":     "#EF4444",
    "smooth":      "#94A3B8",
    "time_s":      "#3B82F6",
}
STAGE_COLORS = {"stage1": "#DBEAFE", "stage2": "#D1FAE5", "stage3": "#FEF3C7"}
STAGE_LABELS = {"stage1": "Stage 1\n(supervision + EMA)",
                "stage2": "Stage 2\n(+ geometry)",
                "stage3": "Stage 3\n(end-to-end)"}


def load_log(path: str) -> tuple:
    with open(path) as f:
        data = json.load(f)
    steps  = data.get("steps",  [])
    epochs = data.get("epochs", [])
    return steps, epochs


def smooth(values, window=5):
    """Simple moving average for step-level plots."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same").tolist()


def get_stage_boundaries(epochs):
    """Return list of (start_epoch, end_epoch, stage_name) tuples."""
    boundaries = []
    if not epochs:
        return boundaries
    current_stage = epochs[0]["stage"]
    start = 0
    for i, e in enumerate(epochs):
        if e["stage"] != current_stage:
            boundaries.append((start, i - 1, current_stage))
            current_stage = e["stage"]
            start = i
    boundaries.append((start, len(epochs) - 1, current_stage))
    return boundaries


def add_stage_bands(ax, boundaries, epochs):
    """Shade background by training stage."""
    for s, e, stage in boundaries:
        ax.axvspan(s, e, alpha=0.12,
                   color=STAGE_COLORS.get(stage, "#F3F4F6"),
                   zorder=0)
        mid = (s + e) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97,
                STAGE_LABELS.get(stage, stage),
                ha="center", va="top", fontsize=7,
                color="#6B7280", style="italic")


# ════════════════════════════════════════════════════════════════
# Plot 1 — Total loss: train vs val
# ════════════════════════════════════════════════════════════════

def plot_total_loss(epochs, out_dir):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))

    xs          = [e["epoch"]      for e in epochs]
    train_vals  = [e["train_loss"] for e in epochs]
    val_vals    = [e["val_loss"]   for e in epochs]

    ax.plot(xs, train_vals, color=LOSS_COLORS["train_loss"],
            label="Train loss", linewidth=2)
    ax.plot(xs, val_vals,   color=LOSS_COLORS["val_loss"],
            label="Val loss",   linewidth=2, linestyle="--")

    # mark best epoch
    best = min(epochs, key=lambda e: e["val_loss"])
    ax.scatter([best["epoch"]], [best["val_loss"]],
               color="#EF4444", s=60, zorder=5,
               label=f"Best val = {best['val_loss']:.4f} (epoch {best['epoch']})")

    boundaries = get_stage_boundaries(epochs)
    for s, e, stage in boundaries:
        ax.axvline(s, color="#D1D5DB", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total loss — train vs validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "01_total_loss.png", dpi=150)
    plt.close(fig)
    print(f"  saved 01_total_loss.png")


# ════════════════════════════════════════════════════════════════
# Plot 2 — Individual loss components (epoch-level, from step log)
# ════════════════════════════════════════════════════════════════

def plot_loss_components(steps, epochs, out_dir):
    if not steps:
        return

    # Aggregate step log by epoch using epoch boundaries
    # Build a mapping: step → epoch (approximate via total steps)
    all_loss_names = set()
    for s in steps:
        all_loss_names.update(k for k in s.keys()
                              if k not in ("step", "stage", "total"))

    if not all_loss_names:
        return

    # Group by stage for colour banding
    # Build per-step arrays
    step_nums = [s["step"] for s in steps]

    n_losses = len(all_loss_names)
    ncols = 3
    nrows = (n_losses + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, name in enumerate(sorted(all_loss_names)):
        ax = axes[idx]
        vals = [s.get(name, None) for s in steps]
        # filter None
        xs_plot = [x for x, v in zip(step_nums, vals) if v is not None]
        ys_plot = [v for v in vals if v is not None]

        if not ys_plot:
            ax.set_visible(False)
            continue

        # raw (faint) + smoothed
        ax.plot(xs_plot, ys_plot, alpha=0.2,
                color=LOSS_COLORS.get(name, "#6B7280"), linewidth=0.8)
        ax.plot(xs_plot, smooth(ys_plot, window=max(1, len(ys_plot)//50)),
                color=LOSS_COLORS.get(name, "#6B7280"), linewidth=1.8,
                label="smoothed")

        ax.set_title(f"L_{name}" if not name.startswith("L_") else name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

    # hide unused subplots
    for i in range(idx + 1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Individual loss components (per step)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "02_loss_components.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 02_loss_components.png")


# ════════════════════════════════════════════════════════════════
# Plot 3 — Loss breakdown (final epoch, stacked bar per stage)
# ════════════════════════════════════════════════════════════════

def plot_loss_breakdown(steps, out_dir):
    if not steps:
        return

    stages = list(dict.fromkeys(s["stage"] for s in steps))
    loss_names = sorted(set(
        k for s in steps
        for k in s.keys()
        if k not in ("step", "stage", "total")
    ))

    if not loss_names:
        return

    # Mean of last 10% of steps per stage
    stage_means = {}
    for stage in stages:
        stage_steps = [s for s in steps if s["stage"] == stage]
        tail = stage_steps[max(0, len(stage_steps) - len(stage_steps)//10):]
        means = {}
        for name in loss_names:
            vals = [s[name] for s in tail if name in s]
            means[name] = np.mean(vals) if vals else 0.0
        stage_means[stage] = means

    fig, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(stages))
    width  = 0.55
    bottom = np.zeros(len(stages))

    for name in loss_names:
        vals = [stage_means[s].get(name, 0) for s in stages]
        bars = ax.bar(x, vals, width, bottom=bottom,
                      color=LOSS_COLORS.get(name, "#9CA3AF"),
                      label=f"L_{name}", alpha=0.85)
        # value labels on tall bars
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages])
    ax.set_ylabel("Mean loss contribution")
    ax.set_title("Loss breakdown by stage (mean of final 10% of steps)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "03_loss_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  saved 03_loss_breakdown.png")


# ════════════════════════════════════════════════════════════════
# Plot 4 — Training time per epoch
# ════════════════════════════════════════════════════════════════

def plot_training_time(epochs, out_dir):
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    xs    = [e["epoch"]  for e in epochs]
    times = [e["time_s"] for e in epochs]

    bars = ax.bar(xs, times, color=LOSS_COLORS["time_s"], alpha=0.75, width=0.8)

    # stage colour bands
    boundaries = get_stage_boundaries(epochs)
    for s, e, stage in boundaries:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.10,
                   color=STAGE_COLORS.get(stage, "#F3F4F6"), zorder=0)

    # mean line
    mean_t = np.mean(times)
    ax.axhline(mean_t, color="#EF4444", linestyle="--", linewidth=1,
               label=f"Mean: {mean_t:.1f}s")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("Training time per epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "04_training_time.png", dpi=150)
    plt.close(fig)
    print(f"  saved 04_training_time.png")


# ════════════════════════════════════════════════════════════════
# Plot 5 — Train vs val gap (overfitting monitor)
# ════════════════════════════════════════════════════════════════

def plot_train_val_gap(epochs, out_dir):
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    xs   = [e["epoch"]                               for e in epochs]
    gap  = [e["val_loss"] - e["train_loss"]          for e in epochs]
    cols = ["#EF4444" if g > 0 else "#059669"        for g in gap]

    ax.bar(xs, gap, color=cols, alpha=0.7, width=0.8)
    ax.axhline(0, color="#374151", linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss − Train loss")
    ax.set_title("Generalisation gap  (positive = overfitting, negative = underfitting)")

    boundaries = get_stage_boundaries(epochs)
    for s, e, stage in boundaries:
        ax.axvline(s, color="#D1D5DB", linewidth=0.8, linestyle=":")

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "05_generalisation_gap.png", dpi=150)
    plt.close(fig)
    print(f"  saved 05_generalisation_gap.png")


# ════════════════════════════════════════════════════════════════
# Plot 6 — Summary dashboard (all in one figure)
# ════════════════════════════════════════════════════════════════

def plot_dashboard(steps, epochs, out_dir):
    if not epochs:
        return

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Total loss ──
    ax0 = fig.add_subplot(gs[0, :2])
    xs = [e["epoch"] for e in epochs]
    ax0.plot(xs, [e["train_loss"] for e in epochs],
             color=LOSS_COLORS["train_loss"], label="Train", linewidth=2)
    ax0.plot(xs, [e["val_loss"]   for e in epochs],
             color=LOSS_COLORS["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    best = min(epochs, key=lambda e: e["val_loss"])
    ax0.scatter([best["epoch"]], [best["val_loss"]],
                color="#EF4444", s=60, zorder=5)
    boundaries = get_stage_boundaries(epochs)
    for s, e, stage in boundaries:
        ax0.axvspan(s, e, alpha=0.10, color=STAGE_COLORS.get(stage,"#F3F4F6"), zorder=0)
        ax0.axvline(s, color="#D1D5DB", linewidth=0.7, linestyle=":")
    ax0.set_title("Total loss")
    ax0.set_xlabel("Epoch")
    ax0.legend()

    # ── Generalisation gap ──
    ax1 = fig.add_subplot(gs[0, 2])
    gap  = [e["val_loss"] - e["train_loss"] for e in epochs]
    cols = ["#EF4444" if g > 0 else "#059669" for g in gap]
    ax1.bar(xs, gap, color=cols, alpha=0.7, width=0.8)
    ax1.axhline(0, color="#374151", linewidth=1)
    ax1.set_title("Val − Train gap")
    ax1.set_xlabel("Epoch")

    # ── Training time ──
    ax2 = fig.add_subplot(gs[1, 0])
    times = [e["time_s"] for e in epochs]
    ax2.bar(xs, times, color=LOSS_COLORS["time_s"], alpha=0.75, width=0.8)
    ax2.axhline(np.mean(times), color="#EF4444", linestyle="--", linewidth=1)
    ax2.set_title("Time / epoch (s)")
    ax2.set_xlabel("Epoch")

    # ── Individual losses from step log ──
    loss_names = sorted(set(
        k for s in steps
        for k in s.keys()
        if k not in ("step", "stage", "total")
    ))

    subplot_positions = [gs[1,1], gs[1,2], gs[2,0], gs[2,1], gs[2,2]]
    for idx, name in enumerate(loss_names[:5]):
        ax = fig.add_subplot(subplot_positions[idx])
        vals = [s.get(name) for s in steps if s.get(name) is not None]
        snums = list(range(len(vals)))
        ax.plot(snums, vals, alpha=0.2,
                color=LOSS_COLORS.get(name,"#6B7280"), linewidth=0.7)
        ax.plot(snums, smooth(vals, max(1,len(vals)//40)),
                color=LOSS_COLORS.get(name,"#6B7280"), linewidth=1.8)
        ax.set_title(f"L_{name}")
        ax.set_xlabel("Step")

    fig.suptitle("DD-Net Training Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(Path(out_dir) / "00_dashboard.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 00_dashboard.png")


# ════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════

def generate_all_plots(log_path: str, out_dir: str = None):
    """
    Read training_log.json and generate all plots.

    Args:
        log_path : path to training_log.json
        out_dir  : output directory (default: same dir as log file)
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if out_dir is None:
        out_dir = log_path.parent / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {log_path}")
    steps, epochs = load_log(log_path)
    print(f"  {len(steps)} steps,  {len(epochs)} epochs")

    print("Generating plots...")
    plot_dashboard(steps, epochs, out_dir)
    plot_total_loss(steps, epochs, out_dir) if False else plot_total_loss(epochs, out_dir)
    plot_loss_components(steps, epochs, out_dir)
    plot_loss_breakdown(steps, out_dir)
    plot_training_time(epochs, out_dir)
    plot_train_val_gap(epochs, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
    print("Files:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True,
                        help="Path to training_log.json")
    parser.add_argument("--out", default=None,
                        help="Output directory for plots")
    args = parser.parse_args()
    generate_all_plots(args.log, args.out)

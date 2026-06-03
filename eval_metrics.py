"""
eval_metrics.py v2 — fixed stage separation in all plots

Each stage gets its own subplot so lines never overlap.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    '#FAFAFA',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'axes.labelsize':    9,
    'legend.fontsize':   8,
    'lines.linewidth':   1.8,
})

LOSS_COLORS = {
    'total':      '#111827',
    'train_loss': '#1E40AF',
    'val_loss':   '#DC2626',
    'depth_sup':  '#059669',
    'desc_sup':   '#10B981',
    'feat':       '#7C3AED',
    'reproj':     '#D97706',
    'depth_mv':   '#F59E0B',
    'desc_mv':    '#EF4444',
    'smooth':     '#94A3B8',
    'time_s':     '#3B82F6',
}

STAGE_COLORS = {
    'stage1': '#EFF6FF',
    'stage2': '#F0FDF4',
    'stage3': '#FFFBEB',
}
STAGE_BORDER = {
    'stage1': '#BFDBFE',
    'stage2': '#BBF7D0',
    'stage3': '#FDE68A',
}
STAGE_LABELS = {
    'stage1': 'Stage 1\n(Supervision + EMA)',
    'stage2': 'Stage 2\n(+ Geometry)',
    'stage3': 'Stage 3\n(End-to-end)',
}


# ── Helpers ────────────────────────────────────────────────────

def load_log(path):
    with open(path) as f:
        d = json.load(f)
    return d.get('steps', []), d.get('epochs', [])


def smooth(vals, window=15):
    if len(vals) < window:
        return np.array(vals, dtype=float)
    k = np.ones(window) / window
    return np.convolve(vals, k, mode='same')


def split_by_stage(items, key='stage'):
    """Group a list of dicts by their stage field."""
    stages = list(dict.fromkeys(s[key] for s in items))
    return stages, {st: [s for s in items if s[key] == st] for st in stages}


def loss_names(steps):
    names = set()
    for s in steps:
        names.update(k for k in s if k not in ('step', 'stage', 'total'))
    return sorted(names)


def savefig(fig, path, name):
    fig.tight_layout()
    fig.savefig(path / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')


def style_stage_ax(ax, stage, title_suffix=''):
    """Apply stage colour band and title to an axis."""
    ax.set_facecolor(STAGE_COLORS.get(stage, 'white'))
    for spine in ax.spines.values():
        spine.set_edgecolor(STAGE_BORDER.get(stage, '#E5E7EB'))
    label = STAGE_LABELS.get(stage, stage)
    ax.set_title(f'{label}\n{title_suffix}' if title_suffix else label,
                 fontsize=9)


# ════════════════════════════════════════════════════════════════
# Plot 1 — Total loss: train vs val, one subplot per stage
# ════════════════════════════════════════════════════════════════

def plot_total_loss(epochs, out):
    if not epochs:
        return
    stages, by_stage = split_by_stage(epochs)
    n = len(stages)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5),
                              sharey=False)
    if n == 1:
        axes = [axes]

    overall_best = min(epochs, key=lambda e: e['val_loss'])

    for ax, stage in zip(axes, stages):
        ep = by_stage[stage]
        xs = [e['epoch'] for e in ep]
        tr = [e['train_loss'] for e in ep]
        vl = [e['val_loss']   for e in ep]

        ax.plot(xs, tr, color=LOSS_COLORS['train_loss'],
                lw=2, label='Train', zorder=3)
        ax.plot(xs, vl, color=LOSS_COLORS['val_loss'],
                lw=2, linestyle='--', label='Val', zorder=3)

        # mark best within stage
        best = min(ep, key=lambda e: e['val_loss'])
        ax.scatter([best['epoch']], [best['val_loss']],
                   color='#EF4444', s=80, zorder=5,
                   label=f"Best val={best['val_loss']:.3f}")

        style_stage_ax(ax, stage)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total loss')
        ax.legend(fontsize=8)

    fig.suptitle('Total loss — train vs validation  (one panel per stage)',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '01_total_loss.png')


# ════════════════════════════════════════════════════════════════
# Plot 2 — Individual loss components, one row per stage
# ════════════════════════════════════════════════════════════════

def plot_loss_components(steps, out):
    if not steps:
        return
    names = loss_names(steps)
    if not names:
        return

    stages, by_stage = split_by_stage(steps)
    n_stages = len(stages)
    n_losses = len(names)

    fig, axes = plt.subplots(n_stages, n_losses,
                              figsize=(4.5 * n_losses, 3.5 * n_stages),
                              squeeze=False)

    for row, stage in enumerate(stages):
        ss = by_stage[stage]
        # local step counter starting from 0 for each stage
        for col, name in enumerate(names):
            ax   = axes[row][col]
            vals = [s[name] for s in ss if name in s]
            if not vals:
                ax.set_visible(False)
                continue
            xs   = list(range(len(vals)))
            col_ = LOSS_COLORS.get(name, '#6B7280')
            ax.plot(xs, vals, alpha=0.18, color=col_, lw=0.7)
            ax.plot(xs, smooth(vals, max(1, len(vals) // 30)),
                    color=col_, lw=2)
            style_stage_ax(ax, stage)
            ax.set_title(f'{STAGE_LABELS.get(stage, stage)}\nL_{name}',
                         fontsize=8)
            ax.set_xlabel('Step (within stage)')
            ax.set_ylabel('Loss')

    fig.suptitle('Loss components per stage  (each panel is one stage × one loss)',
                 fontsize=12, fontweight='bold', y=1.01)
    savefig(fig, out, '02_loss_components.png')


# ════════════════════════════════════════════════════════════════
# Plot 3 — Loss breakdown stacked bar per stage
# ════════════════════════════════════════════════════════════════

def plot_loss_breakdown(steps, out):
    if not steps:
        return
    names  = loss_names(steps)
    stages, by_stage = split_by_stage(steps)
    if not names or not stages:
        return

    stage_means = {}
    for st in stages:
        ss   = by_stage[st]
        tail = ss[max(0, len(ss) - max(1, len(ss) // 10)):]
        stage_means[st] = {
            n: float(np.mean([s[n] for s in tail if n in s]))
            for n in names
        }

    fig, ax = plt.subplots(figsize=(max(6, 3 * len(stages)), 5))
    x, bot  = np.arange(len(stages)), np.zeros(len(stages))

    for name in names:
        vals = [stage_means[s].get(name, 0) for s in stages]
        bars = ax.bar(x, vals, 0.55, bottom=bot,
                      color=LOSS_COLORS.get(name, '#9CA3AF'),
                      label=f'L_{name}', alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0.05 * max(bot + np.array(vals)):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
        bot += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages])
    ax.set_ylabel('Mean loss  (final 10% of stage)')
    ax.set_title('Loss breakdown by stage')
    ax.legend(loc='upper right', ncol=2)
    savefig(fig, out, '03_loss_breakdown.png')


# ════════════════════════════════════════════════════════════════
# Plot 4 — Generalisation gap per stage
# ════════════════════════════════════════════════════════════════

def plot_generalisation_gap(epochs, out):
    if not epochs:
        return
    stages, by_stage = split_by_stage(epochs)
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        ep  = by_stage[stage]
        xs  = [e['epoch'] for e in ep]
        gap = [e['val_loss'] - e['train_loss'] for e in ep]
        ax.bar(xs, gap,
               color=['#EF4444' if g > 0 else '#059669' for g in gap],
               alpha=0.75, width=0.8)
        ax.axhline(0, color='#374151', lw=1.2)
        style_stage_ax(ax, stage)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val − Train')

    fig.suptitle('Generalisation gap  (red = overfitting,  green = underfitting)',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '04_generalisation_gap.png')


# ════════════════════════════════════════════════════════════════
# Plot 5 — Training time per epoch
# ════════════════════════════════════════════════════════════════

def plot_training_time(epochs, out):
    if not epochs:
        return
    stages, by_stage = split_by_stage(epochs)
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        ep    = by_stage[stage]
        xs    = [e['epoch'] for e in ep]
        times = [e['time_s'] for e in ep]
        ax.bar(xs, times, color=LOSS_COLORS['time_s'], alpha=0.75, width=0.8)
        ax.axhline(np.mean(times), color='#EF4444', lw=1.2, linestyle='--',
                   label=f'Mean {np.mean(times):.0f}s')
        style_stage_ax(ax, stage)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Seconds')
        ax.legend(fontsize=8)

    fig.suptitle('Training time per epoch',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '05_training_time.png')


# ════════════════════════════════════════════════════════════════
# Plot 6 — Val/Train ratio per stage
# ════════════════════════════════════════════════════════════════

def plot_loss_ratio(epochs, out):
    if not epochs:
        return
    stages, by_stage = split_by_stage(epochs)
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        ep    = by_stage[stage]
        xs    = [e['epoch'] for e in ep]
        ratio = [e['train_loss'] / max(e['val_loss'], 1e-8) for e in ep]
        ax.plot(xs, ratio, color='#7C3AED', lw=2)
        ax.axhline(1.0, color='#374151', lw=1, linestyle='--')
        ax.fill_between(xs, ratio, 1.0,
                        where=[r < 1 for r in ratio],
                        alpha=0.15, color='#059669')
        ax.fill_between(xs, ratio, 1.0,
                        where=[r >= 1 for r in ratio],
                        alpha=0.15, color='#EF4444')
        style_stage_ax(ax, stage)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train / Val')

    fig.suptitle('Train/Val loss ratio  (>1 overfitting,  <1 underfitting)',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '06_loss_ratio.png')


# ════════════════════════════════════════════════════════════════
# Plot 7 — Convergence speed per stage (normalised)
# ════════════════════════════════════════════════════════════════

def plot_convergence(steps, out):
    if not steps:
        return
    names  = loss_names(steps)
    stages, by_stage = split_by_stage(steps)
    n      = len(stages)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        ss = by_stage[stage]
        for name in names:
            vals = [s[name] for s in ss if name in s]
            if not vals or vals[0] == 0:
                continue
            sm   = smooth(vals, max(1, len(vals) // 30))
            norm = sm / (sm[0] + 1e-8)
            ax.plot(range(len(norm)), norm,
                    color=LOSS_COLORS.get(name, '#6B7280'),
                    lw=1.6, label=f'L_{name}', alpha=0.85)

        ax.axhline(1.0, color='#CBD5E1', lw=0.8, linestyle=':', label='start')
        ax.axhline(0.5, color='#CBD5E1', lw=0.8, linestyle='--', label='50%')
        ax.set_ylim(bottom=0)
        style_stage_ax(ax, stage)
        ax.set_xlabel('Step (within stage)')
        ax.set_ylabel('Loss / initial')
        ax.legend(ncol=2, fontsize=7)

    fig.suptitle('Convergence speed per stage  (1.0 = starting value)',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '07_convergence_speed.png')


# ════════════════════════════════════════════════════════════════
# Plot 8 — Geometry losses (stage 2+)
# ════════════════════════════════════════════════════════════════

def plot_geometry_losses(steps, out):
    geo   = ['reproj', 'depth_mv', 'desc_mv']
    avail = [n for n in geo if any(n in s for s in steps)]
    if not avail:
        print('  skipped 08_geometry_losses.png')
        return

    stages, by_stage = split_by_stage(steps)
    geo_stages = [st for st in stages
                  if any(any(n in s for n in avail)
                         for s in by_stage[st])]

    if not geo_stages:
        print('  skipped 08_geometry_losses.png')
        return

    n_rows = len(geo_stages)
    n_cols = len(avail)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows),
                              squeeze=False)

    for row, stage in enumerate(geo_stages):
        ss = by_stage[stage]
        for col, name in enumerate(avail):
            ax   = axes[row][col]
            vals = [s[name] for s in ss if name in s]
            if not vals:
                ax.set_visible(False)
                continue
            col_ = LOSS_COLORS.get(name, '#6B7280')
            ax.plot(range(len(vals)), vals,
                    alpha=0.18, color=col_, lw=0.7)
            ax.plot(range(len(vals)),
                    smooth(vals, max(1, len(vals) // 30)),
                    color=col_, lw=2.2)
            style_stage_ax(ax, stage)
            ax.set_title(f'{STAGE_LABELS.get(stage,stage)}\nL_{name}',
                         fontsize=8)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')

    fig.suptitle('Geometry losses per stage',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '08_geometry_losses.png')


# ════════════════════════════════════════════════════════════════
# Plot 9 — EMA loss per stage
# ════════════════════════════════════════════════════════════════

def plot_ema_loss(steps, out):
    stages, by_stage = split_by_stage(steps)
    feat_stages = [st for st in stages
                   if any('feat' in s for s in by_stage[st])]
    if not feat_stages:
        print('  skipped 09_ema_loss.png')
        return

    n = len(feat_stages)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, stage in zip(axes, feat_stages):
        ss   = by_stage[stage]
        vals = [s['feat'] for s in ss if 'feat' in s]
        if not vals:
            continue
        ax.plot(range(len(vals)), vals,
                alpha=0.18, color='#7C3AED', lw=0.7)
        ax.plot(range(len(vals)),
                smooth(vals, max(1, len(vals) // 30)),
                color='#7C3AED', lw=2.2)
        style_stage_ax(ax, stage)
        ax.set_xlabel('Step')
        ax.set_ylabel('L_feat')

    fig.suptitle('EMA distillation loss (L_feat) per stage',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig(fig, out, '09_ema_loss.png')


# ════════════════════════════════════════════════════════════════
# Plot 10 — Dashboard (compact overview with stage separation)
# ════════════════════════════════════════════════════════════════

def plot_dashboard(steps, epochs, out):
    if not epochs:
        return

    stages, by_stage_ep = split_by_stage(epochs)
    n = len(stages)

    fig = plt.figure(figsize=(7 * n, 12))
    gs  = gridspec.GridSpec(3, n, figure=fig,
                             hspace=0.55, wspace=0.35)

    for col, stage in enumerate(stages):
        ep = by_stage_ep[stage]
        xs = [e['epoch'] for e in ep]
        tr = [e['train_loss'] for e in ep]
        vl = [e['val_loss']   for e in ep]

        # row 0: total loss
        ax = fig.add_subplot(gs[0, col])
        ax.plot(xs, tr, color=LOSS_COLORS['train_loss'], lw=2, label='Train')
        ax.plot(xs, vl, color=LOSS_COLORS['val_loss'],
                lw=2, linestyle='--', label='Val')
        best = min(ep, key=lambda e: e['val_loss'])
        ax.scatter([best['epoch']], [best['val_loss']],
                   color='#EF4444', s=60, zorder=5)
        style_stage_ax(ax, stage)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total loss')
        ax.legend(fontsize=7)

        # row 1: generalisation gap
        ax = fig.add_subplot(gs[1, col])
        gap  = [e['val_loss'] - e['train_loss'] for e in ep]
        ax.bar(xs, gap,
               color=['#EF4444' if g > 0 else '#059669' for g in gap],
               alpha=0.75, width=0.8)
        ax.axhline(0, color='#374151', lw=1)
        style_stage_ax(ax, stage, 'Val − Train gap')
        ax.set_xlabel('Epoch')

        # row 2: time
        ax = fig.add_subplot(gs[2, col])
        times = [e['time_s'] for e in ep]
        ax.bar(xs, times, color='#3B82F6', alpha=0.75, width=0.8)
        ax.axhline(np.mean(times), color='#EF4444', lw=1, linestyle='--')
        style_stage_ax(ax, stage, f'Time/epoch  mean={np.mean(times):.0f}s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Seconds')

    fig.suptitle('DD-Net Training Dashboard — one column per stage',
                 fontsize=14, fontweight='bold', y=1.01)
    savefig(fig, out, '00_dashboard.png')


# ════════════════════════════════════════════════════════════════
# Compare runs
# ════════════════════════════════════════════════════════════════

def plot_compare(log_paths, labels, out):
    palette = ['#1E40AF','#DC2626','#059669',
               '#D97706','#7C3AED','#0EA5E9']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        _, epochs = load_log(path)
        if not epochs:
            continue
        col = palette[i % len(palette)]
        xs  = [e['epoch'] for e in epochs]
        tr  = [e['train_loss'] for e in epochs]
        vl  = [e['val_loss']   for e in epochs]
        axes[0].plot(xs, tr, color=col, lw=2, label=f'{label} train')
        axes[0].plot(xs, vl, color=col, lw=2, linestyle='--',
                     label=f'{label} val', alpha=0.7)
        best = min(epochs, key=lambda e: e['val_loss'])
        axes[0].scatter([best['epoch']], [best['val_loss']],
                        color=col, s=60, zorder=5)
        if tr[0] > 0:
            norm = [v / tr[0] for v in tr]
            axes[1].plot(xs, norm, color=col, lw=2, label=label)

    axes[0].set_title('Loss comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].set_title('Convergence speed  (normalised to start)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss / initial')
    axes[1].axhline(0.5, color='#CBD5E1', lw=0.8, linestyle=':')
    axes[1].legend()
    fig.suptitle('Run comparison', fontsize=13, fontweight='bold')
    savefig(fig, out, 'compare_runs.png')


# ════════════════════════════════════════════════════════════════
# Text summary
# ════════════════════════════════════════════════════════════════

def print_summary(steps, epochs, label=''):
    tag = f'[{label}]  ' if label else ''
    print(f'\n{tag}── Training summary ──────────────────────')
    if not epochs:
        return
    stages, by_stage = split_by_stage(epochs)
    for stage in stages:
        ep   = by_stage[stage]
        best = min(ep, key=lambda e: e['val_loss'])
        last = ep[-1]
        times = [e['time_s'] for e in ep]
        print(f'  {stage}:')
        print(f'    epochs        : {len(ep)}')
        print(f'    best val loss : {best["val_loss"]:.4f}  (epoch {best["epoch"]})')
        print(f'    final train   : {last["train_loss"]:.4f}')
        print(f'    final val     : {last["val_loss"]:.4f}')
        print(f'    time/epoch    : {np.mean(times):.0f}s  '
              f'total {sum(times)/3600:.2f}h')

    print(f'\n  Loss term drop (start → end of each stage):')
    stages_s, by_stage_s = split_by_stage(steps)
    for stage in stages_s:
        ss    = by_stage_s[stage]
        names = loss_names(ss)
        print(f'  {stage}:')
        for name in names:
            vals = [s[name] for s in ss if name in s]
            if len(vals) >= 2:
                drop = (vals[0] - vals[-1]) / (vals[0] + 1e-8) * 100
                print(f'    L_{name:<12} '
                      f'{vals[0]:.4f} → {vals[-1]:.4f}  ({drop:+.1f}%)')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', nargs='+', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--labels', nargs='+', default=None)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(args.log[0]).parent / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    print(f'Output → {out}\n')

    if args.compare and len(args.log) > 1:
        labels = args.labels or [f'run_{i}' for i in range(len(args.log))]
        plot_compare(args.log, labels, out)
        for path, label in zip(args.log, labels):
            s, e = load_log(path)
            print_summary(s, e, label)
        return

    log_path = args.log[0]
    print(f'Reading {log_path}')
    steps, epochs = load_log(log_path)
    print(f'  {len(steps)} steps   {len(epochs)} epochs\n')
    print('Generating plots...')

    plot_dashboard(steps, epochs, out)
    plot_total_loss(epochs, out)
    plot_loss_components(steps, out)
    plot_loss_breakdown(steps, out)
    plot_generalisation_gap(epochs, out)
    plot_training_time(epochs, out)
    plot_loss_ratio(epochs, out)
    plot_convergence(steps, out)
    plot_geometry_losses(steps, out)
    plot_ema_loss(steps, out)

    print_summary(steps, epochs)
    print(f'\nAll plots saved to {out}/')


if __name__ == '__main__':
    main()

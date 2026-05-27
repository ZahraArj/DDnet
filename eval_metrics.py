"""
eval_metrics.py — training metrics visualiser for DD-Net

Reads training_log.json produced by TrainingLogger and generates
10 publication-quality plots covering all useful training metrics.

Usage:
    python eval_metrics.py --log runs/ddnet_real/training_log.json
    python eval_metrics.py --log runs/ddnet_real/training_log.json --out figures/

    # compare two runs side by side
    python eval_metrics.py --log runs/exp1/training_log.json \
                           --log runs/exp2/training_log.json \
                           --labels "baseline" "with_epipolar" \
                           --compare
"""

import argparse
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
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
    'train_loss': '#111827',
    'val_loss':   '#6366F1',
    'depth_sup':  '#059669',
    'desc_sup':   '#10B981',
    'feat':       '#7C3AED',
    'reproj':     '#D97706',
    'depth_mv':   '#F59E0B',
    'desc_mv':    '#EF4444',
    'smooth':     '#94A3B8',
    'time_s':     '#3B82F6',
}

STAGE_COLORS = {'stage1': '#DBEAFE', 'stage2': '#D1FAE5', 'stage3': '#FEF3C7'}
STAGE_LABELS = {'stage1': 'Stage 1\n(sup+EMA)',
                'stage2': 'Stage 2\n(+geo)',
                'stage3': 'Stage 3\n(e2e)'}


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


def loss_names(steps):
    names = set()
    for s in steps:
        names.update(k for k in s if k not in ('step', 'stage', 'total'))
    return sorted(names)


def stage_bounds(epochs):
    if not epochs:
        return []
    bounds, cur, start = [], epochs[0]['stage'], 0
    for i, e in enumerate(epochs):
        if e['stage'] != cur:
            bounds.append((start, i - 1, cur))
            cur, start = e['stage'], i
    bounds.append((start, len(epochs) - 1, cur))
    return bounds


def add_bands(ax, bounds):
    ylim = ax.get_ylim()
    for s, e, st in bounds:
        ax.axvspan(s, e, alpha=0.10,
                   color=STAGE_COLORS.get(st, '#F3F4F6'), zorder=0)
        ax.axvline(s, color='#CBD5E1', lw=0.7, zorder=1)
        ax.text((s + e) / 2, ylim[1] * 0.97,
                STAGE_LABELS.get(st, st),
                ha='center', va='top', fontsize=7,
                color='#6B7280', style='italic')


def savefig(fig, path, name):
    fig.tight_layout()
    fig.savefig(path / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')


# ════════════════════════════════════════════════════════════════
# Plot 1 — Total loss: train vs val
# ════════════════════════════════════════════════════════════════
def plot_total_loss(epochs, out):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = [e['epoch'] for e in epochs]
    ax.plot(xs, [e['train_loss'] for e in epochs],
            color=LOSS_COLORS['train_loss'], lw=2, label='Train')
    ax.plot(xs, [e['val_loss'] for e in epochs],
            color=LOSS_COLORS['val_loss'], lw=2,
            linestyle='--', label='Val')
    best = min(epochs, key=lambda e: e['val_loss'])
    ax.scatter([best['epoch']], [best['val_loss']], color='#EF4444',
               s=80, zorder=5,
               label=f"Best val = {best['val_loss']:.4f}  (epoch {best['epoch']})")
    add_bands(ax, stage_bounds(epochs))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total loss')
    ax.set_title('Total loss — train vs validation')
    ax.legend()
    savefig(fig, out, '01_total_loss.png')


# ════════════════════════════════════════════════════════════════
# Plot 2 — Individual loss components per step
# ════════════════════════════════════════════════════════════════
def plot_loss_components(steps, out):
    if not steps:
        return
    names = loss_names(steps)
    if not names:
        return
    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, name in enumerate(names):
        ax   = axes[i]
        vals = [s[name] for s in steps if name in s]
        sns  = [s['step'] for s in steps if name in s]
        col  = LOSS_COLORS.get(name, '#6B7280')
        ax.plot(sns, vals, alpha=0.18, color=col, lw=0.7)
        ax.plot(sns, smooth(vals, max(1, len(vals) // 40)),
                color=col, lw=2)
        ax.set_title(f'L_{name}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Individual loss components  (raw=faint, smoothed=solid)',
                 fontsize=12, y=1.01)
    savefig(fig, out, '02_loss_components.png')


# ════════════════════════════════════════════════════════════════
# Plot 3 — Loss breakdown per stage (stacked bar)
# ════════════════════════════════════════════════════════════════
def plot_loss_breakdown(steps, out):
    if not steps:
        return
    names  = loss_names(steps)
    stages = list(dict.fromkeys(s['stage'] for s in steps))
    if not names or not stages:
        return
    stage_means = {}
    for st in stages:
        ss   = [s for s in steps if s['stage'] == st]
        tail = ss[max(0, len(ss) - max(1, len(ss) // 10)):]
        stage_means[st] = {
            n: float(np.mean([s[n] for s in tail if n in s]))
            for n in names
        }
    fig, ax = plt.subplots(figsize=(8, 5))
    x, bot  = np.arange(len(stages)), np.zeros(len(stages))
    for name in names:
        vals = [stage_means[s].get(name, 0) for s in stages]
        bars = ax.bar(x, vals, 0.55, bottom=bot,
                      color=LOSS_COLORS.get(name, '#9CA3AF'),
                      label=f'L_{name}', alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
        bot += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages])
    ax.set_ylabel('Mean loss contribution')
    ax.set_title('Loss breakdown by stage\n(mean of final 10% of steps)')
    ax.legend(loc='upper right', ncol=2)
    savefig(fig, out, '03_loss_breakdown.png')


# ════════════════════════════════════════════════════════════════
# Plot 4 — Generalisation gap
# ════════════════════════════════════════════════════════════════
def plot_generalisation_gap(epochs, out):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs  = [e['epoch'] for e in epochs]
    gap = [e['val_loss'] - e['train_loss'] for e in epochs]
    ax.bar(xs, gap,
           color=['#EF4444' if g > 0 else '#059669' for g in gap],
           alpha=0.75, width=0.8)
    ax.axhline(0, color='#374151', lw=1.2)
    add_bands(ax, stage_bounds(epochs))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val − Train loss')
    ax.set_title('Generalisation gap  (red = overfitting,  green = underfitting)')
    savefig(fig, out, '04_generalisation_gap.png')


# ════════════════════════════════════════════════════════════════
# Plot 5 — Training time per epoch
# ════════════════════════════════════════════════════════════════
def plot_training_time(epochs, out):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs    = [e['epoch'] for e in epochs]
    times = [e['time_s'] for e in epochs]
    ax.bar(xs, times, color=LOSS_COLORS['time_s'], alpha=0.75, width=0.8)
    ax.axhline(np.mean(times), color='#EF4444', lw=1.2,
               linestyle='--', label=f'Mean: {np.mean(times):.1f}s')
    add_bands(ax, stage_bounds(epochs))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.set_title('Training time per epoch')
    ax.legend()
    savefig(fig, out, '05_training_time.png')


# ════════════════════════════════════════════════════════════════
# Plot 6 — Train / Val ratio (overfitting monitor)
# ════════════════════════════════════════════════════════════════
def plot_loss_ratio(epochs, out):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs    = [e['epoch'] for e in epochs]
    ratio = [e['train_loss'] / max(e['val_loss'], 1e-8) for e in epochs]
    ax.plot(xs, ratio, color='#7C3AED', lw=2)
    ax.axhline(1.0, color='#374151', lw=1, linestyle='--', label='ratio = 1')
    ax.fill_between(xs, ratio, 1.0,
                    where=[r < 1 for r in ratio],
                    alpha=0.12, color='#059669', label='underfitting')
    ax.fill_between(xs, ratio, 1.0,
                    where=[r >= 1 for r in ratio],
                    alpha=0.12, color='#EF4444', label='overfitting')
    add_bands(ax, stage_bounds(epochs))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train / Val ratio')
    ax.set_title('Train / Val loss ratio  (>1 = overfitting,  <1 = underfitting)')
    ax.legend()
    savefig(fig, out, '06_loss_ratio.png')


# ════════════════════════════════════════════════════════════════
# Plot 7 — Convergence speed (normalised)
# ════════════════════════════════════════════════════════════════
def plot_convergence(steps, out):
    if not steps:
        return
    names = loss_names(steps)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        vals = [s[name] for s in steps if name in s]
        if not vals or vals[0] == 0:
            continue
        sm   = smooth(vals, max(1, len(vals) // 40))
        norm = sm / (sm[0] + 1e-8)
        ax.plot(range(len(norm)), norm,
                color=LOSS_COLORS.get(name, '#6B7280'),
                lw=1.8, label=f'L_{name}', alpha=0.85)
    ax.axhline(1.0, color='#CBD5E1', lw=0.8, linestyle=':',
               label='starting value')
    ax.axhline(0.5, color='#CBD5E1', lw=0.8, linestyle='--',
               label='50% reduction')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss / initial loss')
    ax.set_title('Loss convergence speed\n'
                 '1.0 = starting value — lower means improved')
    ax.legend(ncol=2)
    savefig(fig, out, '07_convergence_speed.png')


# ════════════════════════════════════════════════════════════════
# Plot 8 — Geometry losses zoom (stage 2+)
# ════════════════════════════════════════════════════════════════
def plot_geometry_losses(steps, out):
    geo = ['reproj', 'depth_mv', 'desc_mv']
    available = [n for n in geo if any(n in s for s in steps)]
    if not available:
        print('  skipped 08_geometry_losses.png  (no geometry losses in log)')
        return
    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]
    for ax, name in zip(axes, available):
        vals = [s[name] for s in steps if name in s]
        sns  = [s['step'] for s in steps if name in s]
        col  = LOSS_COLORS.get(name, '#6B7280')
        ax.plot(sns, vals, alpha=0.18, color=col, lw=0.7)
        ax.plot(sns, smooth(vals, max(1, len(vals) // 40)), color=col, lw=2.2)
        first = next((i for i, v in enumerate(vals) if v > 0), None)
        if first:
            ax.axvline(sns[first], color='#F59E0B', lw=1, linestyle=':',
                       label='activated')
            ax.legend()
        ax.set_title(f'L_{name}  (geometry loss — stage 2+)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
    fig.suptitle('Geometry losses', fontsize=12, y=1.02)
    savefig(fig, out, '08_geometry_losses.png')


# ════════════════════════════════════════════════════════════════
# Plot 9 — EMA distillation loss
# ════════════════════════════════════════════════════════════════
def plot_ema_loss(steps, out):
    vals = [s['feat'] for s in steps if 'feat' in s]
    sns  = [s['step'] for s in steps if 'feat' in s]
    if not vals:
        print('  skipped 09_ema_loss.png  (no feat loss in log)')
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sns, vals, alpha=0.18, color='#7C3AED', lw=0.7)
    ax.plot(sns, smooth(vals, max(1, len(vals) // 40)),
            color='#7C3AED', lw=2.2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('L_feat — EMA self-distillation loss\n'
                 'Should decrease smoothly. '
                 'Plateau means student features have matched EMA.')
    savefig(fig, out, '09_ema_loss.png')


# ════════════════════════════════════════════════════════════════
# Plot 10 — All-in-one dashboard
# ════════════════════════════════════════════════════════════════
def plot_dashboard(steps, epochs, out):
    if not epochs:
        return
    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36)
    xs  = [e['epoch'] for e in epochs]
    bnd = stage_bounds(epochs)

    # total loss
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(xs, [e['train_loss'] for e in epochs],
            color='#111827', lw=2, label='Train')
    ax.plot(xs, [e['val_loss'] for e in epochs],
            color='#6366F1', lw=2, linestyle='--', label='Val')
    best = min(epochs, key=lambda e: e['val_loss'])
    ax.scatter([best['epoch']], [best['val_loss']],
               color='#EF4444', s=80, zorder=5)
    add_bands(ax, bnd)
    ax.set_title('Total loss')
    ax.set_xlabel('Epoch')
    ax.legend()

    # generalisation gap
    ax = fig.add_subplot(gs[0, 2])
    gap  = [e['val_loss'] - e['train_loss'] for e in epochs]
    ax.bar(xs, gap,
           color=['#EF4444' if g > 0 else '#059669' for g in gap],
           alpha=0.75, width=0.8)
    ax.axhline(0, color='#374151', lw=1)
    ax.set_title('Val − Train gap')
    ax.set_xlabel('Epoch')

    # time
    ax = fig.add_subplot(gs[1, 0])
    times = [e['time_s'] for e in epochs]
    ax.bar(xs, times, color='#3B82F6', alpha=0.75, width=0.8)
    ax.axhline(np.mean(times), color='#EF4444', lw=1, linestyle='--')
    ax.set_title('Time / epoch (s)')
    ax.set_xlabel('Epoch')

    # ratio
    ax = fig.add_subplot(gs[1, 1:])
    ratio = [e['train_loss'] / max(e['val_loss'], 1e-8) for e in epochs]
    ax.plot(xs, ratio, color='#7C3AED', lw=2)
    ax.axhline(1.0, color='#374151', lw=0.8, linestyle='--')
    ax.fill_between(xs, ratio, 1.0,
                    where=[r < 1 for r in ratio],
                    alpha=0.12, color='#059669')
    ax.fill_between(xs, ratio, 1.0,
                    where=[r >= 1 for r in ratio],
                    alpha=0.12, color='#EF4444')
    add_bands(ax, bnd)
    ax.set_title('Train/Val ratio  (>1 = overfitting)')
    ax.set_xlabel('Epoch')

    # 3 individual losses
    names  = loss_names(steps)
    posits = [gs[2, 0], gs[2, 1], gs[2, 2]]
    shown  = 0
    for name in names:
        if shown >= 3:
            break
        vals = [s[name] for s in steps if name in s]
        sns  = [s['step'] for s in steps if name in s]
        if not vals:
            continue
        ax = fig.add_subplot(posits[shown])
        ax.plot(sns, vals, alpha=0.18,
                color=LOSS_COLORS.get(name, '#6B7280'), lw=0.7)
        ax.plot(sns, smooth(vals, max(1, len(vals) // 40)),
                color=LOSS_COLORS.get(name, '#6B7280'), lw=2)
        ax.set_title(f'L_{name}')
        ax.set_xlabel('Step')
        shown += 1

    fig.suptitle('DD-Net — Training Dashboard', fontsize=14,
                 fontweight='bold', y=1.01)
    savefig(fig, out, '00_dashboard.png')


# ════════════════════════════════════════════════════════════════
# Multi-run comparison
# ════════════════════════════════════════════════════════════════
def plot_compare(log_paths, labels, out):
    palette = ['#111827', '#6366F1', '#059669',
               '#D97706', '#7C3AED', '#EF4444']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (path, label) in enumerate(zip(log_paths, labels)):
        _, epochs = load_log(path)
        if not epochs:
            continue
        col = palette[i % len(palette)]
        xs  = [e['epoch'] for e in epochs]
        tr  = [e['train_loss'] for e in epochs]
        vl  = [e['val_loss']   for e in epochs]
        axes[0].plot(xs, tr, color=col, lw=2,
                     label=f'{label} train')
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
    print('  saved compare_runs.png')


# ════════════════════════════════════════════════════════════════
# Text summary
# ════════════════════════════════════════════════════════════════
def print_summary(steps, epochs, label=''):
    tag = f'[{label}]  ' if label else ''
    print(f'\n{tag}── Training summary ──────────────────────────')
    if epochs:
        best = min(epochs, key=lambda e: e['val_loss'])
        last = epochs[-1]
        times = [e['time_s'] for e in epochs]
        print(f'  Epochs completed  : {len(epochs)}')
        print(f'  Steps completed   : {len(steps)}')
        print(f'  Best val loss     : {best["val_loss"]:.4f}  (epoch {best["epoch"]})')
        print(f'  Final train loss  : {last["train_loss"]:.4f}')
        print(f'  Final val loss    : {last["val_loss"]:.4f}')
        print(f'  Mean epoch time   : {np.mean(times):.1f}s')
        print(f'  Total train time  : {sum(times)/3600:.2f}h')
        stages = list(dict.fromkeys(e['stage'] for e in epochs))
        print(f'  Stages completed  : {stages}')
    if steps:
        names = loss_names(steps)
        print(f'  Loss terms logged : {names}')
        print(f'  Loss term drop    :')
        for name in names:
            vals = [s[name] for s in steps if name in s]
            if len(vals) >= 2:
                drop = (vals[0] - vals[-1]) / (vals[0] + 1e-8) * 100
                print(f'    L_{name:<12}  '
                      f'{vals[0]:.4f} → {vals[-1]:.4f}  '
                      f'({drop:+.1f}%)')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='DD-Net training metrics visualiser')
    parser.add_argument('--log', nargs='+', required=True,
                        help='path(s) to training_log.json')
    parser.add_argument('--out', default=None,
                        help='output dir  (default: <log_dir>/plots/)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='run labels for comparison mode')
    parser.add_argument('--compare', action='store_true',
                        help='overlay multiple runs (use with --log a b --labels x y)')
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

    # single run
    log_path = args.log[0]
    print(f'Reading {log_path}')
    steps, epochs = load_log(log_path)
    print(f'  {len(steps)} steps   {len(epochs)} epochs\n')
    print('Generating plots...')

    plot_dashboard(steps, epochs, out)          # 00
    plot_total_loss(epochs, out)                # 01
    plot_loss_components(steps, out)            # 02
    plot_loss_breakdown(steps, out)             # 03
    plot_generalisation_gap(epochs, out)        # 04
    plot_training_time(epochs, out)             # 05
    plot_loss_ratio(epochs, out)                # 06
    plot_convergence(steps, out)                # 07
    plot_geometry_losses(steps, out)            # 08
    plot_ema_loss(steps, out)                   # 09

    print_summary(steps, epochs)

    print(f'\nAll plots saved to {out}/')


if __name__ == '__main__':
    main()

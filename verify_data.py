"""
verify_data.py  —  sanity check your .npz files before training

Run from project root:
    python verify_data.py --source data/train/
    python verify_data.py --source data/train.txt --n 20
"""

import argparse
import glob
import os
import sys
import numpy as np


REQUIRED_KEYS = ['rgb_a','rgb_b','Z_gt_a','Z_gt_b',
                 'D_gt_a','D_gt_b','mask_a','mask_b','K_a','K_b']
OPTIONAL_KEYS = ['T_a','T_b']


def check_file(path, desc_dim_expected=None, verbose=False):
    issues = []
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as e:
        return [f'LOAD ERROR: {e}']

    # required keys present?
    for k in REQUIRED_KEYS:
        if k not in d:
            issues.append(f'missing key: {k}')

    if issues:
        return issues   # stop early if keys missing

    H, W = d['rgb_a'].shape[:2]

    # types and shapes
    checks = [
        ('rgb_a',  d['rgb_a'].dtype,  np.uint8,   (H, W, 3)),
        ('rgb_b',  d['rgb_b'].dtype,  np.uint8,   (H, W, 3)),
        ('Z_gt_a', d['Z_gt_a'].dtype, np.float32, (H, W)),
        ('Z_gt_b', d['Z_gt_b'].dtype, np.float32, (H, W)),
        ('mask_a', d['mask_a'].dtype, np.float32, (H, W)),
        ('mask_b', d['mask_b'].dtype, np.float32, (H, W)),
        ('K_a',    d['K_a'].dtype,    np.float32, None),
        ('K_b',    d['K_b'].dtype,    np.float32, None),
    ]
    for key, got_dtype, exp_dtype, exp_shape in checks:
        if got_dtype != exp_dtype:
            issues.append(f'{key}: dtype {got_dtype} != {exp_dtype}')
        if exp_shape and tuple(d[key].shape) != exp_shape:
            issues.append(f'{key}: shape {d[key].shape} != {exp_shape}')

    # K shape (allow [3,3] or [1,3,3])
    for k in ['K_a','K_b']:
        s = d[k].squeeze().shape
        if s != (3,3):
            issues.append(f'{k}: squeezed shape {s} != (3,3)')

    # descriptors
    Da = d['D_gt_a']
    C  = Da.shape[0]
    if len(Da.shape) != 3:
        issues.append(f'D_gt_a: expected 3D [C,H,W], got {Da.shape}')
    elif Da.shape[1] != H or Da.shape[2] != W:
        issues.append(f'D_gt_a: spatial dims {Da.shape[1:]} != ({H},{W})')
    if desc_dim_expected and C != desc_dim_expected:
        issues.append(f'D_gt_a: C={C} != expected {desc_dim_expected}')

    # depth sanity
    Z = d['Z_gt_a']
    if Z.max() > 1000:
        issues.append(f'Z_gt_a max={Z.max():.1f} — looks like mm not metres')
    if Z.min() < 0:
        issues.append(f'Z_gt_a has negative values (min={Z.min():.3f})')

    # mask sanity
    m = d['mask_a']
    if set(np.unique(m.round(2))) - {0.0, 1.0}:
        issues.append('mask_a has values other than 0 and 1')

    # pose (optional)
    for k in ['T_a','T_b']:
        if k in d:
            s = d[k].squeeze().shape
            if s not in [(3,4),(4,4)]:
                issues.append(f'{k}: squeezed shape {s}, expected (3,4) or (4,4)')

    if verbose and not issues:
        print(f'  resolution : {H} × {W}')
        print(f'  desc dim   : {C}')
        print(f'  depth range: {Z.min():.3f} – {Z.max():.3f} m')
        print(f'  valid px   : {m.mean()*100:.1f}%')
        print(f'  has poses  : {"T_a" in d}')

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True,
                        help='directory of .npz files or a .txt list')
    parser.add_argument('--n', type=int, default=None,
                        help='max files to check (default: all)')
    parser.add_argument('--desc_dim', type=int, default=None,
                        help='expected descriptor dim C')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # collect files
    if args.source.endswith('.txt'):
        with open(args.source) as f:
            files = [l.strip() for l in f if l.strip()]
    else:
        files = sorted(glob.glob(os.path.join(args.source, '*.npz')))

    if not files:
        print(f'No .npz files found in {args.source}')
        sys.exit(1)

    if args.n:
        files = files[:args.n]

    print(f'Checking {len(files)} files from {args.source}\n')

    n_ok = 0
    n_bad = 0
    desc_dims = set()

    for path in files:
        issues = check_file(path, args.desc_dim, args.verbose)
        if issues:
            n_bad += 1
            print(f'FAIL  {os.path.basename(path)}')
            for iss in issues:
                print(f'      ✗  {iss}')
        else:
            n_ok += 1
            d = np.load(path, allow_pickle=False)
            desc_dims.add(d['D_gt_a'].shape[0])
            if args.verbose:
                print(f'OK    {os.path.basename(path)}')
                check_file(path, verbose=True)

    print(f'\n{"─"*50}')
    print(f'OK  : {n_ok} / {len(files)}')
    print(f'FAIL: {n_bad} / {len(files)}')
    if desc_dims:
        print(f'Descriptor dims found: {desc_dims}')
        if len(desc_dims) > 1:
            print('WARNING: mixed descriptor dims — all files must have same C')
        else:
            print(f'\nSet in configs/default.yaml:  model.desc_dim: {list(desc_dims)[0]}')

    if n_bad > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

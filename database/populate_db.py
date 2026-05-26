"""
database/populate_db.py
Scans a folder of .npz files, reads metadata from each,
and inserts into the database.

Usage:
    python database/populate_db.py \
        --source /nas2/zahra/ddnet/Data/ \
        --dataset tum_rgbd \
        --depth_source real_sensor \
        --split train
"""

import argparse
import glob
import os
import sqlite3
import numpy as np
from datetime import datetime

DB_PATH = 'database/ddnet_metadata.db'


def extract_metadata(npz_path, dataset, depth_source, split):
    """Read one .npz file and return a metadata dict."""
    d    = np.load(npz_path, allow_pickle=True)
    H, W = d['rgb_a'].shape[:2]

    Z_a    = d['Z_gt_a'].astype(np.float32)
    Z_b    = d['Z_gt_b'].astype(np.float32)
    mask_a = d['mask_a'].astype(np.float32)
    mask_b = d['mask_b'].astype(np.float32)

    valid_a = mask_a.sum()
    valid_b = mask_b.sum()
    total   = H * W

    avg_depth_a = float(Z_a[mask_a > 0].mean()) if valid_a > 0 else 0.0
    avg_depth_b = float(Z_b[mask_b > 0].mean()) if valid_b > 0 else 0.0

    # baseline from poses if available
    baseline_m = None
    if 'T_a' in d and 'T_b' in d:
        t_a = d['T_a'].squeeze()[:3, 3]
        t_b = d['T_b'].squeeze()[:3, 3]
        baseline_m = float(np.linalg.norm(t_a - t_b))

    # parse frame indices from filename: train_pair_0321_to_0322.npz
    name = os.path.basename(npz_path).replace('.npz', '')
    parts = name.split('_')
    try:
        frame_a = int(parts[-3])
        frame_b = int(parts[-1])
    except Exception:
        frame_a = frame_b = None

    return dict(
        file_path      = os.path.abspath(npz_path),
        dataset        = dataset,
        scene          = None,          # set manually if you have scene names
        frame_a        = frame_a,
        frame_b        = frame_b,
        split          = split,
        H              = H,
        W              = W,
        desc_dim       = int(d['D_gt_a'].shape[0]),
        avg_depth_a    = avg_depth_a,
        avg_depth_b    = avg_depth_b,
        valid_px_pct_a = float(valid_a / total),
        valid_px_pct_b = float(valid_b / total),
        baseline_m     = baseline_m,
        overlap_pct    = None,
        depth_source   = depth_source,
        date_added     = datetime.now().isoformat(),
        notes          = None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',       required=True)
    parser.add_argument('--dataset',      required=True,
                        help='dataset name e.g. tum_rgbd, nerfstudio_office')
    parser.add_argument('--depth_source', default='gsplat',
                        choices=['gsplat', 'real_sensor'])
    parser.add_argument('--split',        default='train',
                        choices=['train', 'val'])
    parser.add_argument('--scene',        default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.source, '*.npz')))
    print(f'Found {len(files)} files in {args.source}')

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    inserted = 0
    skipped  = 0
    errors   = 0

    for i, path in enumerate(files):
        try:
            meta = extract_metadata(
                path, args.dataset, args.depth_source, args.split)
            if args.scene:
                meta['scene'] = args.scene

            cur.execute("""
                INSERT OR IGNORE INTO pairs
                (file_path, dataset, scene, frame_a, frame_b, split,
                 H, W, desc_dim, avg_depth_a, avg_depth_b,
                 valid_px_pct_a, valid_px_pct_b, baseline_m,
                 overlap_pct, depth_source, date_added, notes)
                VALUES
                (:file_path, :dataset, :scene, :frame_a, :frame_b, :split,
                 :H, :W, :desc_dim, :avg_depth_a, :avg_depth_b,
                 :valid_px_pct_a, :valid_px_pct_b, :baseline_m,
                 :overlap_pct, :depth_source, :date_added, :notes)
            """, meta)

            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(files)}  inserted={inserted}  '
                      f'skipped={skipped}  errors={errors}')
                conn.commit()

        except Exception as e:
            print(f'  ERROR {os.path.basename(path)}: {e}')
            errors += 1

    conn.commit()
    conn.close()
    print(f'\nDone.  inserted={inserted}  skipped={skipped}  errors={errors}')


if __name__ == '__main__':
    main()
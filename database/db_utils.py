"""
database/db_utils.py
Query helpers for the training pipeline.
"""

import sqlite3

DB_PATH = 'database/ddnet_metadata.db'


def get_file_paths(split='train', dataset=None,
                   min_valid_px=0.3, min_depth=0.1,
                   db_path=DB_PATH):
    """
    Return a list of .npz file paths filtered by quality criteria.

    Args:
        split        : 'train' or 'val'
        dataset      : filter by dataset name (None = all datasets)
        min_valid_px : minimum fraction of valid pixels (0-1)
        min_depth    : minimum average depth in metres
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    query  = """SELECT file_path FROM pairs
                WHERE split = ?
                AND valid_px_pct_a >= ?
                AND valid_px_pct_b >= ?
                AND avg_depth_a    >= ?
                AND avg_depth_b    >= ?"""
    params = [split, min_valid_px, min_valid_px, min_depth, min_depth]

    if dataset:
        query  += ' AND dataset = ?'
        params.append(dataset)

    query += ' ORDER BY id'

    rows = cur.execute(query, params).fetchall()
    conn.close()
    return [r[0] for r in rows]


def dataset_summary(db_path=DB_PATH):
    """Print a summary table of all datasets in the database."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    rows = cur.execute("""
        SELECT
            dataset,
            split,
            COUNT(*)                    as n_pairs,
            ROUND(AVG(avg_depth_a), 2)  as mean_depth,
            ROUND(AVG(valid_px_pct_a)*100, 1) as valid_pct,
            ROUND(AVG(baseline_m), 3)   as mean_baseline,
            MIN(date_added)             as first_added
        FROM pairs
        GROUP BY dataset, split
        ORDER BY dataset, split
    """).fetchall()
    conn.close()

    print(f'\n{"Dataset":<25} {"Split":<6} {"Pairs":>7} '
          f'{"Depth(m)":>9} {"Valid%":>7} {"Baseline":>9} {"Added"}')
    print('─' * 85)
    for r in rows:
        print(f'{r[0]:<25} {r[1]:<6} {r[2]:>7} '
              f'{r[3]:>9} {r[4]:>6}% {str(r[5]):>9} {r[6][:10]}')
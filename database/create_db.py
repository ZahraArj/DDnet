"""
database/create_db.py
Creates the SQLite metadata database.
Run once: python database/create_db.py
"""

import sqlite3
import os

DB_PATH = 'database/ddnet_metadata.db'
os.makedirs('database', exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS pairs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL UNIQUE,
    dataset         TEXT    NOT NULL,
    scene           TEXT,
    frame_a         INTEGER,
    frame_b         INTEGER,
    split           TEXT    DEFAULT 'train',
    H               INTEGER,
    W               INTEGER,
    desc_dim        INTEGER,
    avg_depth_a     REAL,
    avg_depth_b     REAL,
    valid_px_pct_a  REAL,
    valid_px_pct_b  REAL,
    baseline_m      REAL,
    overlap_pct     REAL,
    depth_source    TEXT    DEFAULT 'gsplat',
    date_added      TEXT    DEFAULT (datetime('now')),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_dataset ON pairs(dataset);
CREATE INDEX IF NOT EXISTS idx_split   ON pairs(split);
CREATE INDEX IF NOT EXISTS idx_scene   ON pairs(scene);
""")

conn.commit()
conn.close()
print(f'Database created at {DB_PATH}')
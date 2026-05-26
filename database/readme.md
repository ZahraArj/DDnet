# once — create the database
python database/create_db.py


# once per dataset — populate it
python populate_db.py \
    --source /home/zara/Projects_UW/Gsplat/VisCode/desc_exports/training_pairs/ \
    --dataset Desk1 \
    --depth_source gsplat \
    --split train
    
# anytime — open and query
# open the database
sqlite3 database/ddnet_metadata.db

# inside sqlite3 shell:
.headers on
.mode column

-- see all datasets
SELECT dataset, split, COUNT(*) FROM pairs GROUP BY dataset, split;

-- find low quality pairs
SELECT file_path, valid_px_pct_a, avg_depth_a
FROM pairs WHERE valid_px_pct_a < 0.3 ORDER BY valid_px_pct_a;

-- compare two datasets
SELECT dataset, AVG(avg_depth_a), AVG(baseline_m)
FROM pairs GROUP BY dataset;

-- exit
.quit

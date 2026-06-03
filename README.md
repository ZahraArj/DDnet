# DD-Net: Dense Depth and Descriptor Network

A two-view student network that jointly predicts dense depth maps, per-pixel descriptors, confidence maps, and 3D pointmaps from RGB image pairs. Trained using GSplat pseudo-GT supervision and EMA self-distillation.

---

## Architecture overview

```
img_a, img_b  →  Siamese ViT+FPN encoder  →  F_a^s, F_b^s  [B, 256, H/8, W/8]
                 EMA encoder (no grad)    →  F_a^ema, F_b^ema
F_a^s, F_b^s  →  Cross-view attention    →  F_a^cross, F_b^cross
F_a^cross     →  Depth decoder           →  Z_a, Z_b   [B, 1, H, W]
              →  Descriptor decoder      →  D_a, D_b   [B, 32, H, W]
              →  Confidence head         →  M_a, M_b   [B, 1, H, W]
              →  Pointmap head           →  X_a, X_b   [B, 3, H, W]
Z, D, M, K    →  Geometry block          →  R_ab, t_ab
```

**Trainable parameters:** ~94M (ViT-B backbone dominates at 86M)

---

## Project structure

```
DDNet/
├── models/
│   ├── ddnet.py            # top-level model
│   ├── encoder.py          # SiameseEncoder (ViT + FPN)
│   ├── cross_attention.py  # CrossViewAttention
│   ├── decoders.py         # depth, descriptor, confidence, pointmap heads
│   └── geometry.py         # backproject → soft correspondence → Procrustes
├── training/
│   ├── losses.py           # 7 loss functions
│   └── trainer.py          # staged training loop with logging
├── data/
│   └── dataset.py          # NPZPairDataset, DummyDataset, BaseStereoDataset
├── utils/
│   ├── logger.py           # TrainingLogger → JSON + CSV
│   ├── eval_viz.py         # ReproViz, PoseEval
│   └── geometry_utils.py   # pose_error, scale_invariant_depth_error
├── configs/
│   └── default.yaml        # all hyperparameters
├── train.py                # training entry point
├── evaluate.py             # depth metrics
├── evaluate_pose.py        # reprojection + pose accuracy
├── eval_metrics.py         # training loss plots (10 figures)
├── save_run.py             # package a completed run into a named folder
├── verify_data.py          # validate .npz files before training
├── inspect_model.py        # print architecture + tensor shapes (no training)
└── create_dataset_db.py    # build SQLite metadata database
```

---

## Installation

```bash
conda create -n ddnet python=3.11
conda activate ddnet

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm omegaconf hydra-core tensorboard
pip install numpy pillow tqdm matplotlib
```

---

## Data format

Each training sample is one `.npz` file containing one RGB image pair:

| Key | Shape | Type | Description |
|---|---|---|---|
| `rgb_a`, `rgb_b` | [H, W, 3] | uint8 | RGB images |
| `Z_gt_a`, `Z_gt_b` | [H, W] | float32 | Depth in **metres** (0 = invalid) |
| `D_gt_a`, `D_gt_b` | [C, H, W] | float32 | GSplat descriptor maps |
| `mask_a`, `mask_b` | [H, W] | float32 | Valid depth mask (1 = valid) |
| `K_a`, `K_b` | [3, 3] | float32 | Camera intrinsics in pixel units |
| `T_a`, `T_b` | [1, 3, 4] | float32 | Camera-to-world poses (optional, for eval) |

**Important:**
- Depth must be in **metres** (not millimetres). If `Z_gt.mean() > 50`, divide by 1000.
- Descriptor dimension C must match `cfg.model.desc_dim` (default: 32).
- Poses use Nerfstudio / GSplat camera-to-world convention: `p_world = R @ p_cam + t`.

### Verify your data before training

```bash
python verify_data.py --source /path/to/npz/folder/ --verbose --n 5
```

### Build a metadata database (optional — browse in DBeaver)

```bash
python create_dataset_db.py --source /path/to/data/ --db dataset.db --split train
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
model:
  vit_arch:        "vit_base_patch16_224"
  desc_dim:        32        # must match your descriptor dim C
  fpn_out_channels: 256

data:
  img_height:  480           # must match your rendered resolution
  img_width:   640
  train_dir:   "/path/to/train/npz/"
  val_dir:     "/path/to/val/npz/"

training:
  batch_size:      4         # stage 1 batch size
  batch_size_geo:  2         # stage 2/3 batch size (geometry is memory-heavy)
  n_corr_samples:  512       # points sampled in geometry block

  # loss weights — keep reproj/mv tiny until stage 1 converges
  lambda_depth_sup: 1.0
  lambda_desc_sup:  1.0
  lambda_feat:      0.5
  lambda_reproj:    0.0001
  lambda_depth_mv:  0.0001
  lambda_desc_mv:   0.01
  lambda_smooth:    0.1

  stage1:
    epochs: 50
    active_losses: [depth_sup, desc_sup, feat]
    lr: 1.0e-4
  stage2:
    epochs: 40
    active_losses: [depth_sup, desc_sup, feat, reproj, depth_mv, desc_mv, smooth]
    lr: 5.0e-5
  stage3:
    epochs: 30
    active_losses: [depth_sup, desc_sup, reproj, depth_mv, desc_mv, smooth]
    lr: 2.0e-5
```

---

## Training

```bash
conda activate ddnet
cd /nas2/zahra/ddnet/DDnet

# check GPU
nvidia-smi

# verify data first
python verify_data.py --source /path/to/data/ --n 5 --verbose

# train (uses configs/default.yaml)
python train.py

# or run in background with logging
nohup python train.py > logs/train.log 2>&1 &

# or in a tmux session (survives disconnection)
tmux new -s ddnet_train
python train.py
# detach: Ctrl+B then D
# reattach: tmux attach -t ddnet_train
```

Training saves to `runs/ddnet_real/` and **overwrites** on each run. To keep a run, save it first (see below).

### Monitor training

```bash
# watch logs
tail -f logs/train.log

# TensorBoard
tensorboard --logdir runs/ --port 6006

# generate training plots at any time
python eval_metrics.py --log runs/ddnet_real/training_log.json
```

### If GPU runs out of memory

```bash
# reduce batch size
python train.py training.batch_size=2 training.batch_size_geo=1

# reduce geometry samples
python train.py training.n_corr_samples=256
```

---

## Saving a completed run

When training looks good, save everything into a named folder:

```bash
# save + run all evaluations
python save_run.py --name desc32_480x640_50epochs --eval

# save only (evaluate later)
python save_run.py --name desc32_480x640_50epochs

# evaluate an already-saved run
python save_run.py --name desc32_480x640_50epochs --eval --skip_copy
```

This creates:

```
saved_runs/desc32_480x640_50epochs/
  run_info.json          # metadata: best loss, epochs, date
  config/default.yaml    # exact config used
  logs/
    training_log.json
    training_log.csv
  checkpoints/
    checkpoint_stage1.pt
    checkpoint_stage2.pt
    checkpoint_stage3.pt
  plots/                 # all 10 training plots
  results/               # evaluation outputs
    reprojection.png
    pose_accuracy.png
    pose_summary.json
    depth_metrics.txt
```

The live working folder `runs/ddnet_real/` stays in place and will be overwritten by the next training run.

---

## Evaluation

```bash
# depth metrics (silog, abs_rel, rmse)
python evaluate.py --checkpoint runs/ddnet_real/checkpoint_stage3.pt

# pose accuracy + reprojection visualisation
python evaluate_pose.py \
    --checkpoint runs/ddnet_real/checkpoint_stage3.pt \
    --out results/

# training loss plots (10 figures, per-stage panels)
python eval_metrics.py --log runs/ddnet_real/training_log.json

# compare two training runs
python eval_metrics.py \
    --log saved_runs/run_001/logs/training_log.json \
    --log saved_runs/run_002/logs/training_log.json \
    --labels "baseline" "with_epipolar" \
    --compare
```

### What the evaluation metrics mean

| Metric | Description | Target |
|---|---|---|
| `silog` | Scale-invariant log depth error | < 0.15 |
| `abs_rel` | Mean absolute relative depth error | < 0.10 |
| `rmse` | Root mean squared depth error (metres) | < 0.5 |
| Rotation error (°) | Angle between predicted and GT rotation | < 5° |
| Translation error (°) | Angular error between translation directions | < 5° |
| Acc @ 5°/10°/20° | % of pairs below each threshold | > 50% @ 10° |

---

## Inspecting the model architecture

```bash
# print layer tree, tensor shapes, parameter counts, memory estimate
python inspect_model.py

# full-size model at your training resolution
python inspect_model.py --vit vit_base_patch16_224 --height 480 --width 640

# save to file
python inspect_model.py > architecture.txt
```

---

## GitHub workflow

```bash
# push changes
git add .
git commit -m "description of change"
git push

# pull on server
cd /nas2/zahra/ddnet/DDnet
git pull

# upload data to server (resumes if interrupted)
rsync -avz --progress /local/Data/ user@server:/nas2/zahra/ddnet/Data/
```

---

## Loss functions

| Loss | Stage | Supervision | What it enforces |
|---|---|---|---|
| `L_depth_sup` | 1+ | GSplat depth GT | Depth decoder accuracy |
| `L_desc_sup` | 1+ | GSplat descriptor GT | Descriptor decoder accuracy |
| `L_feat` | 1+ | EMA features (stop_grad) | Feature-level self-distillation |
| `L_reproj` | 2+ | Self-supervised | Reprojected 3D points match descriptor correspondences |
| `L_desc_mv` | 2+ | Self-supervised | Same scene point has same descriptor from both views |
| `L_depth_mv` | 2+ | Self-supervised | Predicted depth consistent with triangulated depth |
| `L_smooth` | 1+ | Regularisation | Depth smooth in flat regions, discontinuous at edges |

---

## Known issues and fixes

**Decoder outputs half resolution (240×320 instead of 480×640)**
Add a 4th upsample block to each decoder head in `models/decoders.py`.

**SVD numerical instability in geometry block**
Add epsilon before SVD in `_weighted_procrustes`:
```python
H_mat = H_mat + 1e-6 * torch.eye(3, device=H_mat.device, dtype=H_mat.dtype)
```

**L_reproj explodes to billions**
Set `lambda_reproj: 0.0001` in config. Do not increase until stage 1 converges.

**CUDA out of memory in stage 2**
Reduce `batch_size_geo` to 2 or 1 and `n_corr_samples` to 512.

**ViT resolution mismatch error**
Add `img_size` and `dynamic_img_size=True` to `timm.create_model` in `encoder.py`.

**Training plots show overlapping lines from multiple runs**
The logger now uses `overwrite=True` by default — each `python train.py` starts a fresh log.

---

## Folder structure at runtime

```
DDNet/
  runs/ddnet_real/          ← active run, overwritten each training
    training_log.json
    training_log.csv
    checkpoint_stage1.pt
    checkpoint_stage2.pt
    checkpoint_stage3.pt
    plots/

  saved_runs/               ← permanent named archives
    run_001/
    run_002/

  results/                  ← latest evaluation output (overwritten)
```

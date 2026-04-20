# DD-Net v3

Two-view dense depth and descriptor network with EMA self-distillation.

## Setup

```bash
conda env create -f environment.yml
conda activate ddnet
```

## Project structure

```
ddnet/
├── configs/
│   └── default.yaml          # all hyperparameters
├── models/
│   ├── encoder.py            # SiameseEncoder (ViT + FPN)
│   ├── cross_attention.py    # CrossViewAttention
│   ├── decoders.py           # SiameseDecoders (depth, desc, conf, ptmap)
│   ├── geometry.py           # GeometryBlock (backproject → PnP → reproject)
│   └── ddnet.py              # DDNet top-level model
├── training/
│   ├── losses.py             # all loss functions + LossManager
│   └── trainer.py            # staged training loop
├── data/
│   └── dataset.py            # BaseStereoDataset + DummyDataset
├── utils/
│   ├── misc.py               # seed, checkpoint helpers
│   ├── visualization.py      # depth/desc/corr visualisation
│   └── geometry_utils.py     # pose error, SILog depth metric
├── train.py                  # entry point
└── evaluate.py               # evaluation script
```

## Weight sharing summary

Every module below is **siamese** — one set of weights called twice:

| Module | Weights | View A output | View B output |
|---|---|---|---|
| SiameseEncoder | θ_enc | F_a^s | F_b^s |
| DepthDecoder | θ_depth | Z_a | Z_b |
| DescDecoder | θ_desc | D_a | D_b |
| ConfHead | θ_conf | M_a | M_b |
| PointmapHead | θ_pt | X_a | X_b |

CrossViewAttention is the **only non-siamese** module — it jointly
processes both views and cannot be split.

EMA copy mirrors the encoder structure but is updated via:
```
θ_ema ← 0.996 · θ_ema + 0.004 · θ
```
No gradient flows into θ_ema.

## Training

```bash
# default config
python train.py

# override any param
python train.py training.batch_size=4 model.vit_arch=vit_small_patch16_224
```

## Adding your own dataset

Subclass `BaseStereoDataset` in `data/dataset.py` and implement `_load_item`:

```python
class MyDataset(BaseStereoDataset):
    def __len__(self): ...
    def _load_item(self, idx) -> dict:
        # return: img_a, img_b (PIL), K_a, K_b (np [3,3]),
        #         Z_gt_a, Z_gt_b (np [H,W]), D_gt_a, D_gt_b (np [C,H,W]),
        #         mask_a, mask_b (np bool [H,W])
        ...
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoint_stage3.pt
```

"""
data/dataset.py  —  DD-Net real data loader

Replaces DummyDataset with NPZPairDataset which loads
your train_pair_XXXX_to_YYYY.npz files directly.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────
# Base class (unchanged from original)
# ─────────────────────────────────────────────────────────────

class BaseStereoDataset(Dataset):
    """
    Abstract base.  Subclass and implement __len__ and _load_item.
    _load_item must return a dict with PIL images and numpy arrays.
    __getitem__ handles resizing, normalisation, and tensor conversion.
    """

    def __init__(self, img_h: int = 384, img_w: int = 512,
                 split: str = 'train'):
        self.img_h = img_h
        self.img_w = img_w
        self.split = split

        self.img_transform = T.Compose([
            T.Resize((img_h, img_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _load_item(self, idx: int) -> dict:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raw = self._load_item(idx)

        # ── images ──────────────────────────────────────────
        img_a = self.img_transform(raw['img_a'])   # [3, H, W]
        img_b = self.img_transform(raw['img_b'])

        orig_H, orig_W = raw['orig_hw']

        # ── helper: resize a [H,W] map to training resolution ──
        def resize_map(arr, mode='nearest'):
            # arr: [H,W] numpy → [1,1,H,W] → resize → [1,H_tr,W_tr] tensor
            t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(self.img_h, self.img_w), mode=mode)
            return t.squeeze(0)   # [1, H, W]

        def resize_desc(arr):
            # arr: [C,H,W] numpy
            t = torch.from_numpy(arr).float().unsqueeze(0)   # [1,C,H,W]
            t = F.interpolate(t, size=(self.img_h, self.img_w),
                              mode='bilinear', align_corners=False)
            return t.squeeze(0)   # [C, H, W]

        # ── depth ────────────────────────────────────────────
        Z_a = resize_map(raw['Z_gt_a'])     # [1, H, W]
        Z_b = resize_map(raw['Z_gt_b'])

        # ── descriptors ──────────────────────────────────────
        D_a = resize_desc(raw['D_gt_a'])    # [C, H, W]
        D_b = resize_desc(raw['D_gt_b'])

        # ── masks ────────────────────────────────────────────
        mask_a = resize_map(raw['mask_a'])  # [1, H, W]
        mask_b = resize_map(raw['mask_b'])

        # ── intrinsics — scale K to training resolution ──────
        K_a = scale_intrinsics(raw['K_a'], orig_H, orig_W,
                               self.img_h, self.img_w)
        K_b = scale_intrinsics(raw['K_b'], orig_H, orig_W,
                               self.img_h, self.img_w)

        out = dict(
            img_a=img_a, img_b=img_b,
            K_a=torch.from_numpy(K_a).float(),
            K_b=torch.from_numpy(K_b).float(),
            Z_gt_a=Z_a, Z_gt_b=Z_b,
            D_gt_a=D_a, D_gt_b=D_b,
            mask_a=mask_a, mask_b=mask_b,
        )

        # ── optional: relative pose from T_a, T_b ────────────
        if 'T_a' in raw and 'T_b' in raw:
            R_rel, t_rel = relative_pose(raw['T_a'], raw['T_b'])
            out['R_gt'] = torch.from_numpy(R_rel).float()
            out['t_gt'] = torch.from_numpy(t_rel).float()

        return out


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def scale_intrinsics(K: np.ndarray,
                     orig_H: int, orig_W: int,
                     new_H: int, new_W: int) -> np.ndarray:
    """
    Scale a [3,3] intrinsic matrix from original image resolution
    to training resolution.  Must be done whenever images are resized.
    """
    K = K.copy().astype(np.float32)
    K[0, 0] *= new_W / orig_W   # fx
    K[1, 1] *= new_H / orig_H   # fy
    K[0, 2] *= new_W / orig_W   # cx
    K[1, 2] *= new_H / orig_H   # cy
    return K


def relative_pose(T_a: np.ndarray,
                  T_b: np.ndarray) -> tuple:
    """
    Compute relative pose R_ab, t_ab from two [3,4] or [1,3,4]
    camera-to-world pose matrices.

    Convention used by Nerfstudio / GSplat:
        T = [R | t]  where  p_world = R @ p_cam + t

    Relative pose A→B:
        R_ab = R_b^T @ R_a    (world→B composed with A→world)
        t_ab = R_b^T @ (t_a - t_b)

    Returns R_ab [3,3], t_ab [3]  both float32
    """
    T_a = T_a.squeeze().astype(np.float64)   # [3,4]
    T_b = T_b.squeeze().astype(np.float64)

    R_a, t_a = T_a[:3, :3], T_a[:3, 3]
    R_b, t_b = T_b[:3, :3], T_b[:3, 3]

    R_ab = R_b.T @ R_a
    t_ab = R_b.T @ (t_a - t_b)

    return R_ab.astype(np.float32), t_ab.astype(np.float32)


def _smooth_quantised_depth(depth, quant_threshold=500):
    valid = depth > 0
    if valid.sum() == 0 or len(np.unique(depth[valid].round(3))) >= quant_threshold:
        return depth
    pad = np.pad(depth, 1, mode='edge')
    h, w = depth.shape
    smoothed = (pad[:-2,:-2] + pad[:-2,1:-1] + pad[:-2,2:] +
                pad[1:-1,:-2] + pad[1:-1,1:-1] + pad[1:-1,2:] +
                pad[2:,:-2]  + pad[2:,1:-1]  + pad[2:,2:]) / 9.0
    return np.where(valid, smoothed.astype(np.float32), 0.0)


# ─────────────────────────────────────────────────────────────
# NPZPairDataset  —  your real data
# ─────────────────────────────────────────────────────────────

class NPZPairDataset(BaseStereoDataset):
    """
    Loads train_pair_XXXX_to_YYYY.npz files.

    Args:
        source   : either a directory (all *.npz files are used)
                   or a .txt file listing one .npz path per line
        img_h    : training image height  (must match cfg.data.img_height)
        img_w    : training image width
        desc_dim : descriptor channels C  (must match cfg.model.desc_dim)
        split    : 'train' or 'val'  (informational only)
    """

    def __init__(self, source: str, img_h: int = 384, img_w: int = 512,
                 desc_dim: int = 128, split: str = 'train'):
        super().__init__(img_h, img_w, split)
        self.desc_dim = desc_dim
        self.files    = self._collect_files(source)

        if len(self.files) == 0:
            raise RuntimeError(
                f'NPZPairDataset: no .npz files found in {source}')
        print(f'[NPZPairDataset] {split}: {len(self.files)} pairs  '
              f'from {source}')

    @staticmethod
    def _collect_files(source: str) -> list:
        if source.endswith('.txt'):
            with open(source) as f:
                return [l.strip() for l in f if l.strip()]
        elif os.path.isdir(source):
            return sorted(glob.glob(os.path.join(source, '*.npz')))
        else:
            raise ValueError(
                f'source must be a directory or a .txt file, got: {source}')

    def __len__(self) -> int:
        return len(self.files)

    def _load_item(self, idx: int) -> dict:
        path = self.files[idx]
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f'Failed to load {path}: {e}')

        # ── validate descriptor dimension ─────────────────
        D_a = data['D_gt_a']   # [C, H_orig, W_orig]
        C_file = D_a.shape[0]
        if C_file != self.desc_dim:
            raise ValueError(
                f'{path}: descriptor dim {C_file} != '
                f'cfg.model.desc_dim {self.desc_dim}. '
                f'Update cfg.model.desc_dim or re-export descriptors.')

        # ── original resolution ───────────────────────────
        H_orig, W_orig = data['rgb_a'].shape[:2]

        # ── intrinsics ────────────────────────────────────
        # squeeze handles both [3,3] and [1,3,3]
        K_a = data['K_a'].squeeze().astype(np.float32)
        K_b = data['K_b'].squeeze().astype(np.float32)


        # ── depth ─────────────────────────────────────────
        Z_a = data['Z_gt_a'].astype(np.float32)
        Z_b = data['Z_gt_b'].astype(np.float32)

        # smooth quantisation artifacts from JPG-saved depth
        # only applied when depth was saved via JPG (few unique values)
        Z_a = _smooth_quantised_depth(Z_a)
        Z_b = _smooth_quantised_depth(Z_b)

        return dict(
            img_a = Image.fromarray(data['rgb_a'].astype(np.uint8)).convert('RGB'),
            img_b = Image.fromarray(data['rgb_b'].astype(np.uint8)).convert('RGB'),

            # numpy arrays
            K_a    = K_a,
            K_b    = K_b,
            Z_gt_a = Z_a,   # [H, W] metres
            Z_gt_b = Z_b,
            D_gt_a = D_a.astype(np.float32),               # [C, H, W]
            D_gt_b = data['D_gt_b'].astype(np.float32),
            mask_a = data['mask_a'].astype(np.float32),    # [H, W]
            mask_b = data['mask_b'].astype(np.float32),

            # poses (optional — used for evaluation only)
            T_a    = data['T_a'] if 'T_a' in data else None,
            T_b    = data['T_b'] if 'T_b' in data else None,

            # pass original resolution so K can be scaled correctly
            orig_hw = (H_orig, W_orig),
        )


# ─────────────────────────────────────────────────────────────
# DummyDataset  —  kept for quick testing without real data
# ─────────────────────────────────────────────────────────────

class DummyDataset(BaseStereoDataset):
    """Synthetic random dataset for unit tests and smoke tests."""

    def __init__(self, length=64, img_h=384, img_w=512, desc_dim=128,
                 with_pose_gt=False):
        super().__init__(img_h, img_w)
        self.length       = length
        self.desc_dim     = desc_dim
        self.with_pose_gt = with_pose_gt

    def __len__(self): return self.length

    def _load_item(self, idx):
        H, W = self.img_h, self.img_w
        K = np.array([[500., 0., W/2],
                      [0., 500., H/2],
                      [0., 0., 1.]], dtype=np.float32)
        raw = dict(
            img_a  = Image.fromarray(
                np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)),
            img_b  = Image.fromarray(
                np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)),
            K_a    = K.copy(), K_b = K.copy(),
            Z_gt_a = np.random.uniform(1, 10, (H, W)).astype(np.float32),
            Z_gt_b = np.random.uniform(1, 10, (H, W)).astype(np.float32),
            D_gt_a = np.random.randn(self.desc_dim, H, W).astype(np.float32),
            D_gt_b = np.random.randn(self.desc_dim, H, W).astype(np.float32),
            mask_a = np.ones((H, W), dtype=np.float32),
            mask_b = np.ones((H, W), dtype=np.float32),
            orig_hw = (H, W),
        )
        if self.with_pose_gt:
            angle = np.random.uniform(0.05, 0.30)
            axis  = np.random.randn(3); axis /= np.linalg.norm(axis)
            Kmat  = np.array([[0,-axis[2],axis[1]],
                              [axis[2],0,-axis[0]],
                              [-axis[1],axis[0],0]], dtype=np.float32)
            R = (np.eye(3) + np.sin(angle)*Kmat
                 + (1-np.cos(angle))*(Kmat@Kmat)).astype(np.float32)
            raw['T_a'] = np.eye(3, 4, dtype=np.float32)[np.newaxis]
            raw['T_b'] = np.hstack([R, np.random.randn(3,1)*0.3]
                                   ).astype(np.float32)[np.newaxis]
        return raw


class DummyDatasetWithPose(DummyDataset):
    def __init__(self, **kwargs):
        kwargs['with_pose_gt'] = True
        super().__init__(**kwargs)

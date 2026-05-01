"""
Dataset base class for DD-Net.

Expects a dataset that returns pairs of images with:
  - two RGB images (img_a, img_b)
  - camera intrinsics (K_a, K_b)
  - GSplat pseudo-GT depth maps (Z_gt_a, Z_gt_b)
  - GSplat pseudo-GT descriptor maps (D_gt_a, D_gt_b)
  - valid pixel masks (mask_a, mask_b)

Subclass this for your specific dataset (ScanNet, MegaDepth, etc.).
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class BaseStereoDataset(Dataset):
    """
    Abstract base class. Subclass and implement __len__ and _load_item.

    Args:
        img_h, img_w : output image resolution
        split        : 'train' or 'val'
    """

    def __init__(self, img_h: int = 384, img_w: int = 512, split: str = "train"):
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
        """
        Override this in your subclass.
        Must return a dict with keys:
            img_a, img_b  : PIL images
            K_a, K_b      : np.ndarray [3, 3]
            Z_gt_a, Z_gt_b: np.ndarray [H, W]  (GSplat depth)
            D_gt_a, D_gt_b: np.ndarray [C, H, W] (GSplat descriptors)
            mask_a, mask_b: np.ndarray [H, W]  bool (valid pixels)
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raw = self._load_item(idx)

        img_a = self.img_transform(raw["img_a"])
        img_b = self.img_transform(raw["img_b"])

        def to_tensor_depth(arr):
            t = torch.from_numpy(arr).float()
            return t.unsqueeze(0) if t.dim() == 2 else t

        def to_tensor_desc(arr):
            return torch.from_numpy(arr).float()

        def resize_depth(t):
            return torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.img_h, self.img_w), mode="nearest"
            ).squeeze(0)

        Z_gt_a = resize_depth(to_tensor_depth(raw["Z_gt_a"]))
        Z_gt_b = resize_depth(to_tensor_depth(raw["Z_gt_b"]))

        # resize descriptors
        D_gt_a = torch.nn.functional.interpolate(
            to_tensor_desc(raw["D_gt_a"]).unsqueeze(0),
            size=(self.img_h, self.img_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        D_gt_b = torch.nn.functional.interpolate(
            to_tensor_desc(raw["D_gt_b"]).unsqueeze(0),
            size=(self.img_h, self.img_w), mode="bilinear", align_corners=False
        ).squeeze(0)

        mask_a = resize_depth(to_tensor_depth(raw["mask_a"].astype(np.float32)))
        mask_b = resize_depth(to_tensor_depth(raw["mask_b"].astype(np.float32)))

        K_a = torch.from_numpy(raw["K_a"]).float()
        K_b = torch.from_numpy(raw["K_b"]).float()

        return dict(
            img_a=img_a, img_b=img_b,
            K_a=K_a, K_b=K_b,
            Z_gt_a=Z_gt_a, Z_gt_b=Z_gt_b,
            D_gt_a=D_gt_a, D_gt_b=D_gt_b,
            mask_a=mask_a, mask_b=mask_b,
        )


class DummyDataset(BaseStereoDataset):
    """
    Synthetic random dataset for testing the forward pass without real data.
    Generates random images, intrinsics, and targets.
    """

    def __init__(self, length: int = 64, img_h: int = 384, img_w: int = 512, desc_dim: int = 128):
        super().__init__(img_h, img_w)
        self.length   = length
        self.desc_dim = desc_dim

    def __len__(self):
        return self.length

    def _load_item(self, idx):
        from PIL import Image
        import numpy as np

        H, W = self.img_h, self.img_w

        # random RGB images
        img_a = Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
        img_b = Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))

        # simple pinhole intrinsics
        fx = fy = 500.0
        K = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=np.float32)

        return dict(
            img_a=img_a, img_b=img_b,
            K_a=K.copy(), K_b=K.copy(),
            Z_gt_a=np.random.uniform(1, 10, (H, W)).astype(np.float32),
            Z_gt_b=np.random.uniform(1, 10, (H, W)).astype(np.float32),
            D_gt_a=np.random.randn(self.desc_dim, H, W).astype(np.float32),
            D_gt_b=np.random.randn(self.desc_dim, H, W).astype(np.float32),
            mask_a=np.ones((H, W), dtype=bool),
            mask_b=np.ones((H, W), dtype=bool),
        )


class DummyDatasetWithPose(DummyDataset):
    """
    DummyDataset extended with random ground-truth pose (R_gt, t_gt).
    Used for testing pose evaluation without real data.
    """

    def _load_item(self, idx):
        raw = super()._load_item(idx)

        # random rotation (small angle, realistic)
        import numpy as np
        angle = np.random.uniform(0.05, 0.40)         # radians
        axis  = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        K_mat = np.array([[0,-axis[2],axis[1]],
                          [axis[2],0,-axis[0]],
                          [-axis[1],axis[0],0]], dtype=np.float32)
        R = np.eye(3) + np.sin(angle)*K_mat + (1-np.cos(angle))*(K_mat@K_mat)
        t = np.random.randn(3).astype(np.float32) * 0.3

        raw["R_gt"] = R
        raw["t_gt"] = t
        return raw

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        raw  = self._load_item(idx)
        item["R_gt"] = torch.from_numpy(raw["R_gt"]).float()
        item["t_gt"] = torch.from_numpy(raw["t_gt"]).float()
        return item

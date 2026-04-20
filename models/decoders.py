"""
Decoder heads — all siamese (shared weights, called per view).

Four heads:
  DepthDecoder    : F_cross -> Z  [B, 1, H, W]
  DescDecoder     : F_cross -> D  [B, desc_dim, H, W]  (L2-normalised)
  ConfHead        : F_cross -> M  [B, 1, H, W]  (sigmoid, 0-1)
  PointmapHead    : F_cross -> X  [B, 3, H, W]  (3D coords per pixel)

All heads upsample from stride-8 back to full resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """2x bilinear upsample + conv + BN + ReLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthDecoder(nn.Module):
    """
    Predicts a dense metric depth map Z.

    Upsamples from stride-8 to full resolution through 3 upsampling
    steps (8x total), then predicts depth with a sigmoid scaled to
    [depth_min, depth_max].

    Args:
        in_channels : C (FPN output channels)
        depth_min   : minimum depth value
        depth_max   : maximum depth value
    """

    def __init__(
        self,
        in_channels: int = 256,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
    ):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.up1 = _upsample_block(in_channels, 128)
        self.up2 = _upsample_block(128, 64)
        self.up3 = _upsample_block(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:  F [B, C, H/8, W/8]
        Returns: Z [B, 1, H, W]  metric depth
        """
        x = self.up1(F)
        x = self.up2(x)
        x = self.up3(x)
        x = self.head(x)
        # sigmoid → scale to [depth_min, depth_max]
        depth = torch.sigmoid(x) * (self.depth_max - self.depth_min) + self.depth_min
        return depth


class DescDecoder(nn.Module):
    """
    Predicts dense per-pixel descriptors D.
    Output is L2-normalised so cosine similarity == dot product.

    Args:
        in_channels : C
        desc_dim    : descriptor output channels
    """

    def __init__(self, in_channels: int = 256, desc_dim: int = 128):
        super().__init__()
        self.up1 = _upsample_block(in_channels, 128)
        self.up2 = _upsample_block(128, 64)
        self.up3 = _upsample_block(64, 32)
        self.head = nn.Conv2d(32, desc_dim, kernel_size=1)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:  F [B, C, H/8, W/8]
        Returns: D [B, desc_dim, H, W]  L2-normalised
        """
        x = self.up1(F)
        x = self.up2(x)
        x = self.up3(x)
        x = self.head(x)
        return F.normalize(x, dim=1)


class ConfHead(nn.Module):
    """
    Predicts a per-pixel confidence / matchability map M in [0, 1].
    Pixels with M near 0 (sky, textureless walls) are down-weighted
    in all geometric losses.

    Args:
        in_channels : C
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.up1 = _upsample_block(in_channels, 128)
        self.up2 = _upsample_block(128, 64)
        self.up3 = _upsample_block(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:  F [B, C, H/8, W/8]
        Returns: M [B, 1, H, W]  confidence in [0,1]
        """
        x = self.up1(F)
        x = self.up2(x)
        x = self.up3(x)
        return torch.sigmoid(self.head(x))


class PointmapHead(nn.Module):
    """
    Directly regresses 3D world coordinates for every pixel.
    X[b, :, u, v] = (X, Y, Z) of the scene point visible at pixel (u,v).

    This avoids the depth->backproject->PnP chain for pose estimation.
    The geometry block can align X_a and X_b directly via Procrustes.

    Args:
        in_channels : C
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.up1 = _upsample_block(in_channels, 128)
        self.up2 = _upsample_block(128, 64)
        self.up3 = _upsample_block(64, 32)
        self.head = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Args:  F [B, C, H/8, W/8]
        Returns: X [B, 3, H, W]  predicted 3D pointmap
        """
        x = self.up1(F)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)


class SiameseDecoders(nn.Module):
    """
    Wrapper that holds one copy of each decoder head and applies
    each to both views (siamese — shared weights).

    Args:
        in_channels : C (encoder output channels)
        depth_min   : min depth
        depth_max   : max depth
        desc_dim    : descriptor output channels
    """

    def __init__(
        self,
        in_channels: int = 256,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        desc_dim: int = 128,
    ):
        super().__init__()
        self.depth_dec  = DepthDecoder(in_channels, depth_min, depth_max)
        self.desc_dec   = DescDecoder(in_channels, desc_dim)
        self.conf_head  = ConfHead(in_channels)
        self.ptmap_head = PointmapHead(in_channels)

    def decode_one(self, F: torch.Tensor) -> tuple:
        """
        Decode a single view's features.
        Returns: Z, D, M, X
        """
        Z = self.depth_dec(F)
        D = self.desc_dec(F)
        M = self.conf_head(F)
        X = self.ptmap_head(F)
        return Z, D, M, X

    def forward(
        self, F_a: torch.Tensor, F_b: torch.Tensor
    ) -> tuple:
        """
        Decode both views.
        Returns: Za, Da, Ma, Xa, Zb, Db, Mb, Xb
        """
        Za, Da, Ma, Xa = self.decode_one(F_a)
        Zb, Db, Mb, Xb = self.decode_one(F_b)
        return Za, Da, Ma, Xa, Zb, Db, Mb, Xb

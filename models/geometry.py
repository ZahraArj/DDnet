"""
Geometry block — the core coupling mechanism.

Steps:
  1. Backproject pixels from view A to 3D using Z_a and K_a
  2. Find soft correspondences in view B using descriptor similarity
  3. Estimate pose R, t via confidence-weighted PnP
  4. Reproject 3D points into view B and compute residuals

All operations are differentiable so gradients flow back into
Z_a, D_a, D_b, M_a through this block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def backproject(
    depth: torch.Tensor,
    K_inv: torch.Tensor,
    pixels: torch.Tensor,
) -> torch.Tensor:
    """
    Backproject sampled pixels to 3D using the depth map.

    Args:
        depth  : [B, 1, H, W]  — metric depth map
        K_inv  : [B, 3, 3]     — inverse intrinsic matrix
        pixels : [B, N, 2]     — (u, v) pixel coordinates

    Returns:
        pts3d  : [B, N, 3]     — 3D points in camera frame
    """
    B, N, _ = pixels.shape

    # look up depth at sampled pixel locations
    # grid_sample expects coords in [-1, 1]
    H = depth.shape[2]
    W = depth.shape[3]
    grid = pixels.clone().float()
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0   # u -> [-1,1]
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0   # v -> [-1,1]
    grid = grid.unsqueeze(2)                               # [B, N, 1, 2]

    z = F.grid_sample(depth, grid, align_corners=True, mode="bilinear")
    z = z.squeeze(-1).squeeze(1)                           # [B, N]

    # homogeneous pixel coordinates
    ones = torch.ones(B, N, 1, device=pixels.device)
    uvw = torch.cat([pixels.float(), ones], dim=-1)        # [B, N, 3]

    # apply K_inv: ray directions
    rays = torch.bmm(uvw, K_inv.transpose(1, 2))          # [B, N, 3]

    # scale by depth
    pts3d = rays * z.unsqueeze(-1)                        # [B, N, 3]
    return pts3d


def soft_correspondence(
    pts3d_a: torch.Tensor,
    pixels_a: torch.Tensor,
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    conf_a: torch.Tensor,
    temperature: float = 0.1,
    corr_stride: int = 2,
) -> tuple:
    """
    For each 3D point (from view A), find its soft expected pixel
    location in view B via descriptor softmax matching.

    Args:
        pts3d_a    : [B, N, 3]    — sampled 3D points from view A
        pixels_a   : [B, N, 2]    — pixel (u,v) of each sampled point
        desc_a     : [B, C, H, W] — view A descriptor map
        desc_b     : [B, C, H, W] — view B descriptor map
        conf_a     : [B, 1, H, W] — view A confidence map
        temperature : softmax sharpness
        corr_stride : additional spatial downsampling before computing
                      similarity (reduces memory by corr_stride^2).
                      corr_stride=2 → 4× less memory, minimal accuracy loss.

    Returns:
        p_b_soft  : [B, N, 2]  — soft expected pixel in view B (full res)
        weights   : [B, N]     — per-point confidence weights
    """
    B, N, _ = pts3d_a.shape
    _, C, H, W = desc_b.shape

    # optionally downsample descriptors for memory efficiency
    if corr_stride > 1:
        desc_b_small = F.avg_pool2d(desc_b, kernel_size=corr_stride, stride=corr_stride)
        desc_b_small = F.normalize(desc_b_small, dim=1)
    else:
        desc_b_small = desc_b
    _, _, Hc, Wc = desc_b_small.shape  # coarse H, W

    # look up descriptors at sampled pixels in view A
    grid_a = pixels_a.clone().float()
    grid_a[..., 0] = 2.0 * grid_a[..., 0] / (W - 1) - 1.0
    grid_a[..., 1] = 2.0 * grid_a[..., 1] / (H - 1) - 1.0
    grid_a = grid_a.unsqueeze(2)                            # [B, N, 1, 2]
    d_a = F.grid_sample(desc_a, grid_a, align_corners=True)
    d_a = d_a.squeeze(3).permute(0, 2, 1)                  # [B, N, C]

    # flatten coarse desc_b spatial → [B, C, Hc*Wc]
    d_b_flat = desc_b_small.view(B, C, -1)                 # [B, C, Hc*Wc]

    # similarity scores: [B, N, Hc*Wc]
    sim = torch.bmm(d_a, d_b_flat) / temperature
    attn = sim.softmax(dim=-1)                             # [B, N, Hc*Wc]

    # build pixel coordinate grid for coarse view B,
    # then scale back to full-res pixel coordinates
    ys, xs = torch.meshgrid(
        torch.arange(Hc, device=desc_b.device, dtype=torch.float32),
        torch.arange(Wc, device=desc_b.device, dtype=torch.float32),
        indexing="ij",
    )
    # scale coarse coords back to full-resolution pixel space
    xs_full = xs * (W / Wc)
    ys_full = ys * (H / Hc)
    coords = torch.stack([xs_full, ys_full], dim=-1).view(-1, 2)  # [Hc*Wc, 2]
    coords = coords.unsqueeze(0).expand(B, -1, -1)                # [B, Hc*Wc, 2]

    # soft expected pixel location in full resolution: [B, N, 2]
    p_b_soft = torch.bmm(attn, coords)

    # confidence weights from view A
    conf_samples = F.grid_sample(conf_a, grid_a, align_corners=True)
    weights = conf_samples.squeeze(3).squeeze(1).squeeze(-1)  # [B, N]

    return p_b_soft, weights


def project(pts3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Perspective projection.

    Args:
        pts3d : [B, N, 3]   — 3D points in camera frame
        K     : [B, 3, 3]   — intrinsic matrix

    Returns:
        pixels : [B, N, 2]  — projected (u, v) coordinates
    """
    # [B, N, 3] x [B, 3, 3]^T → [B, N, 3]
    proj = torch.bmm(pts3d, K.transpose(1, 2))
    z = proj[..., 2:3].clamp(min=1e-6)
    return proj[..., :2] / z


class GeometryBlock(nn.Module):
    """
    Differentiable geometry block.

    Couples depth, descriptors, confidence and camera intrinsics
    into a reprojection pipeline that produces multi-view consistency
    loss signals.

    Note: Full differentiable PnP (∇-RANSAC) requires an external
    library (e.g. poselib or kornia.geometry.epipolar). Here we use
    a weighted DLT approximation that is fully differentiable and
    sufficient for gradient flow. Swap in kornia's essential matrix
    solver for production use.

    Args:
        n_samples   : number of pixels to sample per image
        temperature : softmax temperature for soft correspondence
    """

    def __init__(self, n_samples: int = 2048, temperature: float = 0.1):
        super().__init__()
        self.n_samples = n_samples
        self.temperature = temperature

    def _sample_pixels(self, B: int, H: int, W: int, device) -> torch.Tensor:
        """Randomly sample N pixel (u, v) coordinates. [B, N, 2]"""
        us = torch.randint(0, W, (B, self.n_samples), device=device)
        vs = torch.randint(0, H, (B, self.n_samples), device=device)
        return torch.stack([us, vs], dim=-1).float()

    def _weighted_procrustes(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple:
        """
        Weighted Procrustes / Kabsch algorithm.
        Finds R, t such that dst ≈ R @ src + t.

        Args:
            src, dst : [B, N, 3]
            weights  : [B, N]

        Returns:
            R : [B, 3, 3]
            t : [B, 3]
        """
        w = weights.unsqueeze(-1)                          # [B, N, 1]
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # weighted centroids
        src_c = (src * w).sum(dim=1, keepdim=True) / w_sum  # [B, 1, 3]
        dst_c = (dst * w).sum(dim=1, keepdim=True) / w_sum

        # centre
        src_n = src - src_c
        dst_n = dst - dst_c

        # weighted cross-covariance
        H_mat = torch.bmm(
            (src_n * w).transpose(1, 2), dst_n
        )  # [B, 3, 3]

        U, _, Vh = torch.linalg.svd(H_mat)
        # ensure proper rotation (det = +1)
        d = torch.linalg.det(torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2)))
        D = torch.eye(3, device=src.device).unsqueeze(0).expand(src.shape[0], -1, -1).clone()
        D[:, 2, 2] = d

        R = torch.bmm(Vh.transpose(1, 2), torch.bmm(D, U.transpose(1, 2)))
        t = dst_c.squeeze(1) - torch.bmm(R, src_c.transpose(1, 2)).squeeze(2)
        return R, t

    def forward(
        self,
        Za: torch.Tensor,
        Da: torch.Tensor,
        Ma: torch.Tensor,
        Xa: torch.Tensor,
        Zb: torch.Tensor,
        Db: torch.Tensor,
        Mb: torch.Tensor,
        Xb: torch.Tensor,
        K_a: torch.Tensor,
        K_b: torch.Tensor,
    ) -> dict:
        """
        Args:
            Za, Zb : [B, 1, H, W]   depth maps
            Da, Db : [B, C, H, W]   descriptor maps
            Ma, Mb : [B, 1, H, W]   confidence maps
            Xa, Xb : [B, 3, H, W]   pointmaps
            K_a, K_b : [B, 3, 3]   camera intrinsics

        Returns dict with keys:
            R_ab, t_ab   : pose from PnP path
            R_pt, t_pt   : pose from pointmap path
            p_b_proj     : reprojected pixels in B  [B, N, 2]
            p_b_soft     : soft correspondence pixels  [B, N, 2]
            pts3d_a      : backprojected 3D points from A  [B, N, 3]
            weights      : per-point confidence  [B, N]
            z_geom       : geometric depth at reprojected pixels  [B, N]
        """
        B, _, H, W = Za.shape
        device = Za.device
        K_inv_a = torch.linalg.inv(K_a)

        # ── Step 1: sample pixels and backproject ──
        pixels_a = self._sample_pixels(B, H, W, device)     # [B, N, 2]
        pts3d_a = backproject(Za, K_inv_a, pixels_a)        # [B, N, 3]

        # ── Step 2: soft correspondence ──
        p_b_soft, weights = soft_correspondence(
            pts3d_a, pixels_a, Da, Db, Ma, self.temperature
        )

        # ── Step 3a: pose from descriptors via weighted Procrustes
        # (we use the pointmap to get 3D on the B side at soft locations)
        # sample Xb at p_b_soft locations
        grid_b = p_b_soft.clone()
        grid_b[..., 0] = 2.0 * grid_b[..., 0] / (W - 1) - 1.0
        grid_b[..., 1] = 2.0 * grid_b[..., 1] / (H - 1) - 1.0
        pts3d_b_soft = F.grid_sample(
            Xb,
            grid_b.unsqueeze(2),
            align_corners=True,
        ).squeeze(3).permute(0, 2, 1)                        # [B, N, 3]

        R_ab, t_ab = self._weighted_procrustes(pts3d_a, pts3d_b_soft, weights)

        # ── Step 3b: pose from pointmaps directly (Procrustes on dense maps)
        # flatten and sample a subset
        Xa_flat = Xa.view(B, 3, -1).permute(0, 2, 1)       # [B, HW, 3]
        Xb_flat = Xb.view(B, 3, -1).permute(0, 2, 1)
        Ma_flat = Ma.view(B, -1)                            # [B, HW]
        # subsample for efficiency
        idx = torch.randperm(H * W, device=device)[:self.n_samples]
        R_pt, t_pt = self._weighted_procrustes(
            Xa_flat[:, idx],
            Xb_flat[:, idx],
            Ma_flat[:, idx],
        )

        # ── Step 4: reproject pts3d_a into view B ──
        pts3d_in_b = torch.bmm(pts3d_a, R_ab.transpose(1, 2)) + t_ab.unsqueeze(1)
        p_b_proj = project(pts3d_in_b, K_b)                 # [B, N, 2]
        z_geom = pts3d_in_b[..., 2]                         # [B, N]

        return {
            "R_ab": R_ab,
            "t_ab": t_ab,
            "R_pt": R_pt,
            "t_pt": t_pt,
            "p_b_proj": p_b_proj,
            "p_b_soft": p_b_soft,
            "pts3d_a": pts3d_a,
            "pixels_a": pixels_a,
            "weights": weights,
            "z_geom": z_geom,
        }

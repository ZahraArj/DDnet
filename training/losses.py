"""
Loss functions for DD-Net.

Each loss is a standalone function.  The LossManager class
combines them according to the active stage and lambda weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# Individual loss functions
# ─────────────────────────────────────────

def loss_depth_sup(
    Z_pred: torch.Tensor,
    Z_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Supervised depth loss vs GSplat pseudo-GT.
    Huber (smooth L1) for robustness to outliers.

    Args:
        Z_pred : [B, 1, H, W]
        Z_gt   : [B, 1, H, W]
        mask   : [B, 1, H, W]  valid pixel mask (optional)
    """
    if Z_gt.shape[2:] != Z_pred.shape[2:]:
        Z_gt = F.interpolate(Z_gt, size=Z_pred.shape[2:], mode="nearest")
        if mask is not None and mask.shape[2:] != Z_pred.shape[2:]:
            mask = F.interpolate(mask, size=Z_pred.shape[2:], mode="nearest")

    loss = F.huber_loss(Z_pred, Z_gt, reduction="none", delta=1.0)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def loss_desc_sup(
    D_pred: torch.Tensor,
    D_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Supervised descriptor loss vs GSplat descriptors.
    Cosine distance = 1 - cosine_similarity.
    D_pred already L2-normalised; normalise D_gt here.

    Args:
        D_pred : [B, C, H, W]  (L2-normalised)
        D_gt   : [B, C, H, W]
        mask   : [B, 1, H, W]
    """
    if D_gt.shape != D_pred.shape:
        D_gt = F.interpolate(D_gt, size=D_pred.shape[-2:], mode="bilinear", align_corners=False)
        if mask is not None:
            mask = F.interpolate(mask, size=D_pred.shape[-2:], mode="nearest")

    D_gt_n = F.normalize(D_gt, dim=1)
    cos_sim = (D_pred * D_gt_n).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    loss = 1.0 - cos_sim
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def loss_feat(
    F_student: torch.Tensor,
    F_ema: torch.Tensor,
    projector: nn.Module,
) -> torch.Tensor:
    """
    EMA self-distillation loss.
    Aligns projected student features to (detached) EMA features.

    Args:
        F_student  : [B, C, H, W]   student features (with grad)
        F_ema      : [B, C, H, W]   EMA features (no grad)
        projector  : 1x1 conv to align dims if needed
    """
    F_s_proj = projector(F_student)
    F_t = F_ema.detach()
    return F.mse_loss(F_s_proj, F_t)


def loss_reproj(
    p_b_proj: torch.Tensor,
    p_b_soft: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Reprojection consistency loss.
    Projected pixel location should match soft correspondence.

    Args:
        p_b_proj : [B, N, 2]   pixels from geometric reprojection
        p_b_soft : [B, N, 2]   soft correspondence pixels
        weights  : [B, N]      per-point confidence
    """
    dist = torch.norm(p_b_proj - p_b_soft, dim=-1)        # [B, N]
    loss = F.huber_loss(dist, torch.zeros_like(dist), reduction="none", delta=2.0)
    loss = (loss * weights).sum() / weights.sum().clamp(min=1)
    return loss


def loss_depth_mv(
    z_geom: torch.Tensor,
    Zb: torch.Tensor,
    p_b_proj: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-view depth consistency.
    Geometric depth at reprojected location should match predicted Z_b.

    Args:
        z_geom   : [B, N]      geometric depth (Z coord after transform)
        Zb       : [B, 1, H, W] predicted depth map for view B
        p_b_proj : [B, N, 2]   reprojected pixel locations in B
        weights  : [B, N]
    """
    B, N, _ = p_b_proj.shape
    H, W = Zb.shape[2], Zb.shape[3]

    grid = p_b_proj.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    z_pred = F.grid_sample(Zb, grid.unsqueeze(2), align_corners=True)
    z_pred = z_pred.squeeze(3).squeeze(1)                  # [B, N]

    diff = torch.abs(z_geom - z_pred)
    loss = (diff * weights).sum() / weights.sum().clamp(min=1)
    return loss


def loss_desc_mv(
    Da: torch.Tensor,
    Db: torch.Tensor,
    pixels_a: torch.Tensor,
    p_b_proj: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-view descriptor consistency.
    Descriptors at corresponding pixels across views should match.

    Args:
        Da, Db   : [B, C, H, W]  descriptor maps
        pixels_a : [B, N, 2]     source pixels in view A
        p_b_proj : [B, N, 2]     corresponding pixels in view B
        weights  : [B, N]
    """
    B, N, _ = pixels_a.shape
    H, W = Da.shape[2], Da.shape[3]

    def sample(desc, pix):
        g = pix.clone()
        g[..., 0] = 2.0 * g[..., 0] / (W - 1) - 1.0
        g[..., 1] = 2.0 * g[..., 1] / (H - 1) - 1.0
        d = F.grid_sample(desc, g.unsqueeze(2), align_corners=True)
        return d.squeeze(3).permute(0, 2, 1)               # [B, N, C]

    da = sample(Da, pixels_a)
    db = sample(Db, p_b_proj)

    # cosine distance
    cos_sim = (da * db).sum(dim=-1)                        # [B, N]
    loss_vals = 1.0 - cos_sim
    loss = (loss_vals * weights).sum() / weights.sum().clamp(min=1)
    return loss


def loss_smooth(
    Z: torch.Tensor,
    img: torch.Tensor,
) -> torch.Tensor:
    """
    Edge-aware depth smoothness.
    |∇Z| * exp(-|∇I|)  — penalises depth discontinuities where
    the image is smooth, tolerates them at image edges.

    Args:
        Z   : [B, 1, H, W]  depth
        img : [B, 3, H, W]  corresponding RGB image
    """
    # resize img to match depth spatial dims if they differ
    if img.shape[2:] != Z.shape[2:]:
        img = F.interpolate(img, size=Z.shape[2:], mode='bilinear', align_corners=False)

    # depth gradients
    dz_dx = torch.abs(Z[:, :, :, :-1] - Z[:, :, :, 1:])
    dz_dy = torch.abs(Z[:, :, :-1, :] - Z[:, :, 1:, :])

    # image gradients (mean across channels)
    img_gray = img.mean(dim=1, keepdim=True)
    di_dx = torch.abs(img_gray[:, :, :, :-1] - img_gray[:, :, :, 1:])
    di_dy = torch.abs(img_gray[:, :, :-1, :] - img_gray[:, :, 1:, :])

    loss = (dz_dx * torch.exp(-di_dx)).mean() + \
           (dz_dy * torch.exp(-di_dy)).mean()
    return loss


# ─────────────────────────────────────────
# Loss manager
# ─────────────────────────────────────────

class LossManager(nn.Module):
    """
    Combines all losses according to the active stage and lambdas.

    The projector aligns student feature channels to EMA feature
    channels for L_feat (needed if dims differ; here they match,
    but the projection head is still good practice per DINO).

    Args:
        cfg : training config section
        feat_channels : student feature channel dim
    """

    def __init__(self, cfg, feat_channels: int = 256):
        super().__init__()
        self.cfg = cfg
        # projection head for L_feat (1x1 conv)
        self.projector = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)

    def forward(
        self,
        pred: dict,
        batch: dict,
        active_losses: list,
    ) -> dict:
        """
        Compute total loss and a breakdown dict for logging.

        Args:
            pred          : output dict from DDNet.forward()
            batch         : data batch (must contain Z_gt_a, Z_gt_b,
                            D_gt_a, D_gt_b, img_a, img_b, mask_a, mask_b)
            active_losses : list of loss name strings for this stage

        Returns:
            dict with 'total' and individual loss values
        """
        cfg = self.cfg
        losses = {}

        if "depth_sup" in active_losses:
            losses["depth_sup"] = (
                loss_depth_sup(pred["Za"], batch["Z_gt_a"], batch.get("mask_a")) +
                loss_depth_sup(pred["Zb"], batch["Z_gt_b"], batch.get("mask_b"))
            ) * 0.5 * cfg.lambda_depth_sup

        if "desc_sup" in active_losses:
            losses["desc_sup"] = (
                loss_desc_sup(pred["Da"], batch["D_gt_a"], batch.get("mask_a")) +
                loss_desc_sup(pred["Db"], batch["D_gt_b"], batch.get("mask_b"))
            ) * 0.5 * cfg.lambda_desc_sup

        if "feat" in active_losses:
            losses["feat"] = (
                loss_feat(pred["Fa_s"], pred["Fa_ema"], self.projector) +
                loss_feat(pred["Fb_s"], pred["Fb_ema"], self.projector)
            ) * 0.5 * cfg.lambda_feat

        if "smooth" in active_losses:
            losses["smooth"] = (
                loss_smooth(pred["Za"], batch["img_a"]) +
                loss_smooth(pred["Zb"], batch["img_b"])
            ) * 0.5 * cfg.lambda_smooth

        # geometry losses — only available when geometry block ran
        if "geometry" in pred:
            geo = pred["geometry"]
            w   = geo["weights"]
            p_proj = geo["p_b_proj"]
            p_soft = geo["p_b_soft"]

            if "reproj" in active_losses:
                losses["reproj"] = loss_reproj(p_proj, p_soft, w) * cfg.lambda_reproj

            if "depth_mv" in active_losses:
                losses["depth_mv"] = loss_depth_mv(
                    geo["z_geom"], pred["Zb"], p_proj, w
                ) * cfg.lambda_depth_mv

            if "desc_mv" in active_losses:
                losses["desc_mv"] = loss_desc_mv(
                    pred["Da"], pred["Db"],
                    geo["pixels_a"], p_proj, w
                ) * cfg.lambda_desc_mv

        losses["total"] = sum(losses.values())
        return losses

"""
Standalone geometry utilities used outside the model
(e.g. evaluation scripts).
"""

import torch
import numpy as np


def pose_error(R_pred: torch.Tensor, t_pred: torch.Tensor,
               R_gt: torch.Tensor,   t_gt: torch.Tensor) -> dict:
    """
    Compute rotation and translation errors between predicted and GT pose.

    Args:
        R_pred, R_gt : [B, 3, 3]
        t_pred, t_gt : [B, 3]

    Returns:
        dict with 'rot_deg' and 'trans_norm' (both [B])
    """
    # rotation error in degrees
    R_err = torch.bmm(R_pred, R_gt.transpose(1, 2))            # should be I
    trace = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    rot_deg = torch.acos(cos_angle) * 180.0 / torch.pi

    # translation error — angle between direction vectors
    t_pred_n = torch.nn.functional.normalize(t_pred, dim=-1)
    t_gt_n   = torch.nn.functional.normalize(t_gt,   dim=-1)
    cos_t = (t_pred_n * t_gt_n).sum(dim=-1).clamp(-1.0, 1.0)
    trans_deg = torch.acos(cos_t) * 180.0 / torch.pi

    return {"rot_deg": rot_deg, "trans_deg": trans_deg}


def scale_invariant_depth_error(pred: torch.Tensor, gt: torch.Tensor,
                                 mask: torch.Tensor = None) -> dict:
    """
    Scale-invariant log depth error (SILog), commonly used for
    monocular depth evaluation.

    Args:
        pred, gt : [B, 1, H, W]
        mask     : [B, 1, H, W]  optional valid pixel mask
    Returns:
        dict with 'silog', 'abs_rel', 'rmse'
    """
    if mask is None:
        mask = (gt > 0).float()

    log_diff = torch.log(pred.clamp(min=1e-6)) - torch.log(gt.clamp(min=1e-6))
    log_diff = log_diff * mask

    n = mask.sum(dim=[1, 2, 3]).clamp(min=1)
    mean_log = log_diff.sum(dim=[1, 2, 3]) / n
    silog = torch.sqrt(((log_diff ** 2).sum(dim=[1, 2, 3]) / n
                        - mean_log ** 2).clamp(min=0)).mean()

    abs_rel = ((torch.abs(pred - gt) / gt.clamp(min=1e-6)) * mask).sum() / mask.sum().clamp(min=1)
    rmse = torch.sqrt(((((pred - gt) ** 2) * mask).sum() / mask.sum().clamp(min=1)).clamp(min=0))

    return {"silog": silog, "abs_rel": abs_rel, "rmse": rmse}

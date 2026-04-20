"""
Visualization helpers for depth maps, descriptors, and correspondences.
Uses matplotlib — no display required (saves to file / returns numpy array).
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def depth_to_color(depth: torch.Tensor, vmin=None, vmax=None) -> np.ndarray:
    """
    Convert a depth map tensor to a colormapped RGB numpy image.

    Args:
        depth : [1, H, W] or [H, W]  tensor
    Returns:
        [H, W, 3]  uint8 RGB
    """
    d = depth.squeeze().cpu().float().numpy()
    vmin = vmin or d.min()
    vmax = vmax or d.max()
    d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    colored = (cm.plasma(d_norm)[..., :3] * 255).astype(np.uint8)
    return colored


def desc_to_pca_color(desc: torch.Tensor) -> np.ndarray:
    """
    Visualise descriptor map via PCA projection to 3 channels → RGB.

    Args:
        desc : [C, H, W]  L2-normalised descriptor map
    Returns:
        [H, W, 3]  uint8 RGB
    """
    C, H, W = desc.shape
    d = desc.cpu().float().numpy().reshape(C, -1).T   # [HW, C]

    # PCA via SVD — take first 3 components
    d -= d.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    proj = d @ Vt[:3].T                               # [HW, 3]
    proj = proj.reshape(H, W, 3)

    # normalise each channel to [0, 1]
    for i in range(3):
        ch = proj[..., i]
        proj[..., i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)

    return (proj * 255).astype(np.uint8)


def draw_correspondences(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    n_draw: int = 50,
    save_path: str = None,
) -> np.ndarray:
    """
    Draw correspondences between two images side by side.

    Args:
        img_a, img_b : [H, W, 3]  uint8 RGB
        pts_a, pts_b : [N, 2]     (u, v) pixel coordinates
        n_draw       : max correspondences to draw
        save_path    : if given, save figure to this path

    Returns:
        [H, W*2+gap, 3]  combined RGB image
    """
    H, W, _ = img_a.shape
    gap = 10
    canvas = np.ones((H, W * 2 + gap, 3), dtype=np.uint8) * 128
    canvas[:, :W] = img_a
    canvas[:, W + gap:] = img_b

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.imshow(canvas)
    ax.axis("off")

    idx = np.random.choice(len(pts_a), min(n_draw, len(pts_a)), replace=False)
    colors = plt.cm.hsv(np.linspace(0, 1, len(idx)))

    for i, j in enumerate(idx):
        u_a, v_a = pts_a[j]
        u_b, v_b = pts_b[j]
        u_b_shifted = u_b + W + gap
        ax.plot([u_a, u_b_shifted], [v_a, v_b], "-", color=colors[i], linewidth=0.8, alpha=0.7)
        ax.plot(u_a, v_a, "o", color=colors[i], markersize=3)
        ax.plot(u_b_shifted, v_b, "o", color=colors[i], markersize=3)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return result


def log_predictions_to_tensorboard(writer, pred: dict, batch: dict, step: int):
    """
    Log depth maps, descriptors, and confidence maps to TensorBoard.

    Args:
        writer : SummaryWriter
        pred   : model output dict
        batch  : data batch dict
        step   : global step
    """
    # depth maps — take first item in batch
    Za_color = depth_to_color(pred["Za"][0])
    Zb_color = depth_to_color(pred["Zb"][0])
    Zgt_color = depth_to_color(batch["Z_gt_a"][0])

    writer.add_image("depth/pred_a",  Za_color.transpose(2, 0, 1),  step)
    writer.add_image("depth/pred_b",  Zb_color.transpose(2, 0, 1),  step)
    writer.add_image("depth/gt_a",    Zgt_color.transpose(2, 0, 1), step)

    # descriptor PCA
    Da_color = desc_to_pca_color(pred["Da"][0])
    writer.add_image("desc/pred_a", Da_color.transpose(2, 0, 1), step)

    # confidence map
    Ma = pred["Ma"][0].squeeze().cpu().numpy()
    writer.add_image("conf/pred_a",
                     (cm.viridis(Ma)[..., :3] * 255).astype(np.uint8).transpose(2, 0, 1),
                     step)

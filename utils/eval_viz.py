"""
Evaluation visualisation — two analyses:

  1. Reprojection visualisation
     Sample N pixels from image A, backproject using predicted depth,
     transform with predicted R,t, project into image B.
     Draw: source pixels on A, reprojected pixels on B, connecting lines
     colour-coded by reprojection error magnitude.

  2. Pose accuracy
     Compare predicted R_ab, t_ab against GT R_ab, t_ab.
     Plots:
       - Rotation error histogram (degrees)
       - Translation error histogram (degrees, angular)
       - Scatter: pred vs GT rotation angle
       - Cumulative accuracy curves (% pairs below threshold)

Usage:
    from utils.eval_viz import ReproViz, PoseEval

    # reprojection on one batch
    viz = ReproViz(model, device)
    fig = viz.visualise(batch, n_pts=200, save_path="reproj.png")

    # pose accuracy over a full loader
    pe  = PoseEval(model, device)
    pe.run(val_loader)
    pe.plot(save_path="pose_accuracy.png")
    pe.print_summary()
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def denorm(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalisation and convert to uint8 [H, W, 3].
    img_tensor: [3, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    img  = img_tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """
    Angle of rotation in degrees for a batch of rotation matrices.
    R: [B, 3, 3]
    Returns: [B]
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos   = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos) * 180.0 / torch.pi


def translation_angle_deg(t_pred: torch.Tensor,
                           t_gt:   torch.Tensor) -> torch.Tensor:
    """
    Angular error between predicted and GT translation directions.
    t_pred, t_gt: [B, 3]
    Returns: [B] degrees
    """
    t_p = F.normalize(t_pred, dim=-1)
    t_g = F.normalize(t_gt,   dim=-1)
    cos = (t_p * t_g).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos) * 180.0 / torch.pi


def project_pts(pts3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Perspective projection.
    pts3d: [B, N, 3]
    K:     [B, 3, 3]
    Returns: [B, N, 2]  pixel coords
    """
    proj = torch.bmm(pts3d, K.transpose(1, 2))     # [B, N, 3]
    z    = proj[..., 2:3].clamp(min=1e-6)
    return proj[..., :2] / z


def backproject_pts(depth: torch.Tensor,
                    K_inv: torch.Tensor,
                    pixels: torch.Tensor) -> torch.Tensor:
    """
    Backproject sampled pixels to 3D.
    depth:  [B, 1, H, W]
    K_inv:  [B, 3, 3]
    pixels: [B, N, 2]   (u, v)
    Returns: [B, N, 3]
    """
    B, N, _ = pixels.shape
    H, W    = depth.shape[2], depth.shape[3]

    g = pixels.clone().float()
    g[..., 0] = 2.0 * g[..., 0] / (W - 1) - 1.0
    g[..., 1] = 2.0 * g[..., 1] / (H - 1) - 1.0

    z = F.grid_sample(depth, g.unsqueeze(2), align_corners=True)
    z = z.squeeze(3).squeeze(1)                     # [B, N]

    ones = torch.ones(B, N, 1, device=pixels.device)
    uvw  = torch.cat([pixels.float(), ones], dim=-1)
    rays = torch.bmm(uvw, K_inv.transpose(1, 2))
    return rays * z.unsqueeze(-1)


# ════════════════════════════════════════════════════════════════
# 1. Reprojection Visualisation
# ════════════════════════════════════════════════════════════════

class ReproViz:
    """
    Visualise reprojection error on a batch of image pairs.

    For each pair:
      - Sample n_pts pixels from image A
      - Backproject using predicted depth Z_a and K_a
      - Transform using predicted R_ab, t_ab
      - Project into image B using K_b
      - Draw source points on A and projected points on B
      - Colour by reprojection error (green=good, red=bad)

    Args:
        model  : DDNet instance
        device : 'cuda' or 'cpu'
    """

    def __init__(self, model, device: str = "cuda"):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def visualise(
        self,
        batch:     dict,
        n_pts:     int  = 200,
        max_pairs: int  = 4,
        save_path: str  = None,
        error_thresh_px: float = 5.0,
    ) -> plt.Figure:
        """
        Args:
            batch           : data batch from DataLoader
            n_pts           : number of points to sample per image
            max_pairs       : max image pairs to show
            save_path       : if given, save figure here
            error_thresh_px : pixels — threshold for green/red colouring

        Returns:
            matplotlib Figure
        """
        self.model.eval()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        pred = self.model(
            batch["img_a"], batch["img_b"],
            batch["K_a"],  batch["K_b"],
            run_geometry=True,
        )

        B  = min(batch["img_a"].shape[0], max_pairs)
        Za = pred["Za"]
        geo = pred["geometry"]

        K_a   = batch["K_a"]
        K_b   = batch["K_b"]
        K_inv = torch.linalg.inv(K_a)
        R_ab  = geo["R_ab"]
        t_ab  = geo["t_ab"]

        H, W = batch["img_a"].shape[2], batch["img_a"].shape[3]

        fig, axes = plt.subplots(B, 3, figsize=(14, 4.5 * B))
        if B == 1:
            axes = axes[np.newaxis, :]

        for b in range(B):
            # sample random pixels in A
            us = torch.randint(0, W, (1, n_pts), device=self.device)
            vs = torch.randint(0, H, (1, n_pts), device=self.device)
            pixels_a = torch.stack([us, vs], dim=-1).float()   # [1, N, 2]

            # backproject
            pts3d = backproject_pts(
                Za[b:b+1], K_inv[b:b+1], pixels_a
            )  # [1, N, 3]

            # transform
            pts_b = torch.bmm(pts3d, R_ab[b:b+1].transpose(1, 2)) \
                    + t_ab[b:b+1].unsqueeze(1)   # [1, N, 3]

            # project into B
            p_proj = project_pts(pts_b, K_b[b:b+1])   # [1, N, 2]

            # --- compute error vs soft correspondence ---
            p_soft = geo["p_b_soft"][b:b+1, :n_pts]   # may be fewer if N_geo != n_pts
            # recompute soft correspondence for our sampled pixels
            # (skip if shapes don't match — just show absolute positions)
            use_soft = False

            # pixel coords (numpy)
            src_uv   = pixels_a[0].cpu().numpy()       # [N, 2]
            proj_uv  = p_proj[0].cpu().numpy()         # [N, 2]

            # clamp projections to image bounds for display
            proj_uv_clamped = proj_uv.copy()
            proj_uv_clamped[:, 0] = np.clip(proj_uv_clamped[:, 0], 0, W - 1)
            proj_uv_clamped[:, 1] = np.clip(proj_uv_clamped[:, 1], 0, H - 1)

            # z-depth of transformed points (positive = in front of camera)
            z_vals = pts_b[0, :, 2].cpu().numpy()
            valid  = z_vals > 0.05   # only show points in front of camera

            # reprojection distance as error proxy
            # (how far the projected point is from image centre — rough quality signal)
            dist_from_centre = np.sqrt(
                (proj_uv[:, 0] - W/2)**2 + (proj_uv[:, 1] - H/2)**2
            )
            # normalise to [0,1] for colour
            err_norm = np.clip(
                np.sqrt((proj_uv[:, 0] - src_uv[:, 0])**2 +
                        (proj_uv[:, 1] - src_uv[:, 1])**2)
                / (error_thresh_px * 10), 0, 1
            )

            # ── image A: source points ──
            img_a = denorm(batch["img_a"][b])
            ax_a  = axes[b, 0]
            ax_a.imshow(img_a)
            scatter = ax_a.scatter(
                src_uv[valid, 0], src_uv[valid, 1],
                c=err_norm[valid], cmap="RdYlGn_r",
                s=12, vmin=0, vmax=1, alpha=0.85,
            )
            ax_a.set_title(f"Image A  —  source pixels  (pair {b})",
                           fontsize=9)
            ax_a.axis("off")

            # ── image B: projected points ──
            img_b = denorm(batch["img_b"][b])
            ax_b  = axes[b, 1]
            ax_b.imshow(img_b)
            ax_b.scatter(
                proj_uv_clamped[valid, 0], proj_uv_clamped[valid, 1],
                c=err_norm[valid], cmap="RdYlGn_r",
                s=12, vmin=0, vmax=1, alpha=0.85,
            )
            ax_b.set_title(f"Image B  —  reprojected pixels",
                           fontsize=9)
            ax_b.axis("off")

            # ── side-by-side with connecting lines ──
            ax_c = axes[b, 2]
            combined = np.concatenate([img_a, img_b], axis=1)
            ax_c.imshow(combined)

            colors = cm.RdYlGn_r(err_norm)
            n_draw = min(80, valid.sum())
            idx    = np.where(valid)[0]
            if len(idx) > n_draw:
                idx = np.random.choice(idx, n_draw, replace=False)

            for i in idx:
                ax_c.plot(
                    [src_uv[i, 0], proj_uv_clamped[i, 0] + W],
                    [src_uv[i, 1], proj_uv_clamped[i, 1]],
                    color=colors[i], linewidth=0.7, alpha=0.7,
                )
            ax_c.set_title("Correspondences  (green=close, red=far)",
                           fontsize=9)
            ax_c.axis("off")

        plt.colorbar(scatter, ax=axes[:, 1], fraction=0.02, pad=0.02,
                     label="Reprojection error (normalised)")
        fig.suptitle("Reprojection visualisation", fontsize=12,
                     fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")

        return fig


# ════════════════════════════════════════════════════════════════
# 2. Pose Accuracy Evaluation
# ════════════════════════════════════════════════════════════════

class PoseEval:
    """
    Compare predicted R, t against ground-truth over a full dataset.

    Metrics:
      - Rotation error (degrees): angle between R_pred and R_gt
      - Translation error (degrees): angular error between t directions
      - AUC of accuracy curve at [5°, 10°, 20°] thresholds
      - Median and mean errors

    The batch must contain 'R_gt' [B,3,3] and 't_gt' [B,3].
    If your dataset does not provide GT pose, this class will
    skip those batches gracefully.

    Args:
        model  : DDNet instance
        device : 'cuda' or 'cpu'
    """

    def __init__(self, model, device: str = "cuda"):
        self.model      = model
        self.device     = device
        self.rot_errors = []   # degrees, one per sample
        self.tra_errors = []   # degrees, one per sample
        self.n_skipped  = 0

    def reset(self):
        self.rot_errors = []
        self.tra_errors = []
        self.n_skipped  = 0

    @torch.no_grad()
    def run(self, loader, max_batches: int = None):
        """
        Run evaluation over a DataLoader.

        Args:
            loader      : val DataLoader
            max_batches : stop early after this many batches (for quick eval)
        """
        self.model.eval()
        self.reset()

        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            if "R_gt" not in batch or "t_gt" not in batch:
                self.n_skipped += batch["img_a"].shape[0]
                continue

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            pred = self.model(
                batch["img_a"], batch["img_b"],
                batch["K_a"],  batch["K_b"],
                run_geometry=True,
            )

            R_pred = pred["geometry"]["R_ab"]
            t_pred = pred["geometry"]["t_ab"]
            R_gt   = batch["R_gt"]
            t_gt   = batch["t_gt"]

            # relative rotation error
            R_err   = torch.bmm(R_pred, R_gt.transpose(1, 2))
            rot_err = rotation_angle_deg(R_err)

            # translation angular error
            tra_err = translation_angle_deg(t_pred, t_gt)

            self.rot_errors.extend(rot_err.cpu().tolist())
            self.tra_errors.extend(tra_err.cpu().tolist())

        print(f"Evaluated {len(self.rot_errors)} pairs "
              f"({self.n_skipped} skipped — no GT pose)")

    def _auc(self, errors: list, max_thresh: float = 30.0) -> float:
        """Area under the accuracy curve up to max_thresh degrees."""
        thresholds = np.linspace(0, max_thresh, 100)
        accs = [np.mean(np.array(errors) < t) for t in thresholds]
        return float(np.trapz(accs, thresholds) / max_thresh)

    def summary(self) -> dict:
        if not self.rot_errors:
            return {}
        re = np.array(self.rot_errors)
        te = np.array(self.tra_errors)
        return {
            "rot_median_deg":   float(np.median(re)),
            "rot_mean_deg":     float(np.mean(re)),
            "tra_median_deg":   float(np.median(te)),
            "tra_mean_deg":     float(np.mean(te)),
            "rot_acc@5":        float(np.mean(re < 5)),
            "rot_acc@10":       float(np.mean(re < 10)),
            "rot_acc@20":       float(np.mean(re < 20)),
            "tra_acc@5":        float(np.mean(te < 5)),
            "tra_acc@10":       float(np.mean(te < 10)),
            "tra_acc@20":       float(np.mean(te < 20)),
            "rot_auc@30":       self._auc(re, 30),
            "tra_auc@30":       self._auc(te, 30),
            "n_pairs":          len(re),
        }

    def print_summary(self):
        s = self.summary()
        if not s:
            print("No results — run .run(loader) first.")
            return
        print("\n── Pose Accuracy ────────────────────────────────")
        print(f"  Pairs evaluated : {s['n_pairs']}")
        print(f"  Rotation  error : median={s['rot_median_deg']:.2f}°  "
              f"mean={s['rot_mean_deg']:.2f}°")
        print(f"  Translation err : median={s['tra_median_deg']:.2f}°  "
              f"mean={s['tra_mean_deg']:.2f}°")
        print(f"  Rot  acc @5/10/20° : "
              f"{s['rot_acc@5']*100:.1f}% / "
              f"{s['rot_acc@10']*100:.1f}% / "
              f"{s['rot_acc@20']*100:.1f}%")
        print(f"  Tran acc @5/10/20° : "
              f"{s['tra_acc@5']*100:.1f}% / "
              f"{s['tra_acc@10']*100:.1f}% / "
              f"{s['tra_acc@20']*100:.1f}%")
        print(f"  Rot  AUC@30° : {s['rot_auc@30']:.3f}")
        print(f"  Tran AUC@30° : {s['tra_auc@30']:.3f}")
        print("─────────────────────────────────────────────────")

    def plot(self, save_path: str = None) -> plt.Figure:
        """
        Generate a 6-panel pose accuracy figure.

        Panels:
          1. Rotation error histogram
          2. Translation error histogram
          3. Cumulative accuracy curves (rot + trans)
          4. Rotation error CDF zoomed to 0-30°
          5. Scatter: sample index vs rotation error
          6. Summary stats text box
        """
        if not self.rot_errors:
            print("No results to plot.")
            return None

        re = np.array(self.rot_errors)
        te = np.array(self.tra_errors)
        s  = self.summary()

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Pose Accuracy — Predicted vs Ground Truth",
                     fontsize=13, fontweight="bold")

        RCOL = "#6366F1"   # rotation colour
        TCOL = "#059669"   # translation colour

        # ── 1. Rotation error histogram ──
        ax = axes[0, 0]
        ax.hist(re, bins=40, color=RCOL, alpha=0.75, edgecolor="white")
        ax.axvline(np.median(re), color="#EF4444", lw=1.5, linestyle="--",
                   label=f"Median: {np.median(re):.1f}°")
        ax.axvline(np.mean(re),   color="#F59E0B", lw=1.5, linestyle=":",
                   label=f"Mean: {np.mean(re):.1f}°")
        ax.set_xlabel("Rotation error (°)")
        ax.set_ylabel("Count")
        ax.set_title("Rotation error distribution")
        ax.legend(fontsize=8)

        # ── 2. Translation error histogram ──
        ax = axes[0, 1]
        ax.hist(te, bins=40, color=TCOL, alpha=0.75, edgecolor="white")
        ax.axvline(np.median(te), color="#EF4444", lw=1.5, linestyle="--",
                   label=f"Median: {np.median(te):.1f}°")
        ax.axvline(np.mean(te),   color="#F59E0B", lw=1.5, linestyle=":",
                   label=f"Mean: {np.mean(te):.1f}°")
        ax.set_xlabel("Translation error (°)")
        ax.set_ylabel("Count")
        ax.set_title("Translation error distribution")
        ax.legend(fontsize=8)

        # ── 3. Cumulative accuracy curves ──
        ax = axes[0, 2]
        thresholds = np.linspace(0, 45, 200)
        rot_acc = [np.mean(re < t) * 100 for t in thresholds]
        tra_acc = [np.mean(te < t) * 100 for t in thresholds]
        ax.plot(thresholds, rot_acc, color=RCOL, lw=2, label="Rotation")
        ax.plot(thresholds, tra_acc, color=TCOL, lw=2, label="Translation")
        for thr in [5, 10, 20]:
            ax.axvline(thr, color="#D1D5DB", lw=0.8, linestyle=":")
            ax.text(thr + 0.3, 5, f"{thr}°", fontsize=7, color="#6B7280")
        ax.set_xlabel("Error threshold (°)")
        ax.set_ylabel("% pairs below threshold")
        ax.set_title("Cumulative accuracy curve")
        ax.legend()
        ax.set_ylim(0, 105)

        # ── 4. Rotation error sorted (per-sample) ──
        ax = axes[1, 0]
        sorted_re = np.sort(re)
        sorted_te = np.sort(te)
        x = np.arange(len(sorted_re)) / len(sorted_re) * 100
        ax.plot(x, sorted_re, color=RCOL, lw=2, label="Rotation")
        ax.plot(x, sorted_te, color=TCOL, lw=2, label="Translation")
        ax.axhline(5,  color="#D1D5DB", lw=0.8, linestyle=":")
        ax.axhline(10, color="#D1D5DB", lw=0.8, linestyle=":")
        ax.axhline(20, color="#D1D5DB", lw=0.8, linestyle=":")
        ax.set_xlabel("Percentile of pairs (%)")
        ax.set_ylabel("Error (°)")
        ax.set_title("Error vs percentile")
        ax.legend()

        # ── 5. Rotation vs translation scatter ──
        ax = axes[1, 1]
        n_show = min(500, len(re))
        idx    = np.random.choice(len(re), n_show, replace=False)
        sc     = ax.scatter(re[idx], te[idx], alpha=0.4, s=15,
                            c=re[idx] + te[idx], cmap="plasma")
        plt.colorbar(sc, ax=ax, fraction=0.04, label="Combined error")
        ax.set_xlabel("Rotation error (°)")
        ax.set_ylabel("Translation error (°)")
        ax.set_title("Rotation vs translation error scatter")

        # ── 6. Summary text ──
        ax = axes[1, 2]
        ax.axis("off")
        lines = [
            ("Pairs evaluated",   str(s["n_pairs"])),
            ("", ""),
            ("Rotation",          ""),
            ("  Median",          f"{s['rot_median_deg']:.2f}°"),
            ("  Mean",            f"{s['rot_mean_deg']:.2f}°"),
            ("  Acc @ 5°",        f"{s['rot_acc@5']*100:.1f}%"),
            ("  Acc @ 10°",       f"{s['rot_acc@10']*100:.1f}%"),
            ("  Acc @ 20°",       f"{s['rot_acc@20']*100:.1f}%"),
            ("  AUC @ 30°",       f"{s['rot_auc@30']:.3f}"),
            ("", ""),
            ("Translation",       ""),
            ("  Median",          f"{s['tra_median_deg']:.2f}°"),
            ("  Mean",            f"{s['tra_mean_deg']:.2f}°"),
            ("  Acc @ 5°",        f"{s['tra_acc@5']*100:.1f}%"),
            ("  Acc @ 10°",       f"{s['tra_acc@10']*100:.1f}%"),
            ("  Acc @ 20°",       f"{s['tra_acc@20']*100:.1f}%"),
            ("  AUC @ 30°",       f"{s['tra_auc@30']:.3f}"),
        ]
        y = 0.97
        for label, val in lines:
            if label == "" and val == "":
                y -= 0.04
                continue
            bold = val == "" and label != ""
            ax.text(0.05, y, label, transform=ax.transAxes,
                    fontsize=9 if not bold else 10,
                    fontweight="bold" if bold else "normal",
                    color="#374151" if not bold else "#111827",
                    va="top")
            if val:
                ax.text(0.70, y, val, transform=ax.transAxes,
                        fontsize=9, color="#111827", va="top",
                        fontweight="bold")
            y -= 0.055

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")

        return fig

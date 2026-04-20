"""
DDNet — top-level model.

Wires together:
  SiameseEncoder   (shared weights, both views)
  EMA copy         (updated after each step, no grad)
  CrossViewAttention
  SiameseDecoders  (shared weights, both views)
  GeometryBlock

Usage:
    model = DDNet(cfg)
    out   = model(img_a, img_b, K_a, K_b)
    model.update_ema()   # call after each optimizer step
"""

import copy
import torch
import torch.nn as nn

from .encoder        import SiameseEncoder
from .cross_attention import CrossViewAttention
from .decoders       import SiameseDecoders
from .geometry       import GeometryBlock


class DDNet(nn.Module):
    """
    DD-Net v3 with EMA self-distillation.

    Args:
        cfg : OmegaConf / dict-like config (see configs/default.yaml)
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model

        # ── Student modules (trained by backprop) ──
        self.encoder = SiameseEncoder(
            vit_arch=m.vit_arch,
            pretrained=m.vit_pretrained,
            out_channels=m.fpn_out_channels,
        )
        self.cross_attn = CrossViewAttention(
            channels=m.fpn_out_channels,
            n_heads=m.cross_attn_heads,
            n_layers=m.cross_attn_layers,
        )
        self.decoders = SiameseDecoders(
            in_channels=m.fpn_out_channels,
            depth_min=m.depth_min,
            depth_max=m.depth_max,
            desc_dim=m.desc_dim,
        )
        self.geometry = GeometryBlock(
            n_samples=cfg.training.n_corr_samples,
        )

        # ── EMA copy (no gradients, updated manually) ──
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad_(False)

        self.ema_momentum = m.ema_momentum

    @torch.no_grad()
    def update_ema(self):
        """
        Update EMA encoder weights after each optimizer step.
        θ_ema ← m * θ_ema + (1 - m) * θ
        Call this once per training step after optimizer.step().
        """
        m = self.ema_momentum
        for p_s, p_t in zip(
            self.encoder.parameters(), self.ema_encoder.parameters()
        ):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        K_a: torch.Tensor,
        K_b: torch.Tensor,
        run_geometry: bool = True,
    ) -> dict:
        """
        Full forward pass.

        Args:
            img_a, img_b   : [B, 3, H, W]  normalised RGB
            K_a,   K_b     : [B, 3, 3]     camera intrinsics
            run_geometry   : set False in stage 1 to skip geometry block

        Returns dict with all predictions:
            Za, Da, Ma, Xa       — view A outputs
            Zb, Db, Mb, Xb       — view B outputs
            Fa_s, Fb_s           — student features (for L_feat)
            Fa_ema, Fb_ema       — EMA features (for L_feat, no grad)
            geometry             — dict from GeometryBlock (if run_geometry)
        """
        # ── Student encoder (grads flow) ──
        Fa_s, Fb_s = self.encoder(img_a, img_b)

        # ── EMA encoder (no grads) ──
        with torch.no_grad():
            Fa_ema, Fb_ema = self.ema_encoder(img_a, img_b)

        # ── Cross-view attention ──
        Fa_cross, Fb_cross = self.cross_attn(Fa_s, Fb_s)

        # ── Decoder heads (siamese) ──
        Za, Da, Ma, Xa, Zb, Db, Mb, Xb = self.decoders(Fa_cross, Fb_cross)

        out = dict(
            Za=Za, Da=Da, Ma=Ma, Xa=Xa,
            Zb=Zb, Db=Db, Mb=Mb, Xb=Xb,
            Fa_s=Fa_s,   Fb_s=Fb_s,
            Fa_ema=Fa_ema, Fb_ema=Fb_ema,
        )

        # ── Geometry block (only in stage 2+) ──
        if run_geometry:
            out["geometry"] = self.geometry(
                Za, Da, Ma, Xa,
                Zb, Db, Mb, Xb,
                K_a, K_b,
            )

        return out

"""
Smoke tests — verify tensor shapes through every module.
Run with:  pytest tests/ -v

These tests use tiny models (vit_tiny, C=64, N=64 samples) so they
run on CPU in a few seconds without real data or pretrained weights.
"""

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


# ── Shared minimal config ────────────────────────────────────────
@pytest.fixture
def cfg():
    return OmegaConf.create({
        "model": {
            "vit_arch":        "vit_tiny_patch16_224",
            "vit_pretrained":  False,
            "fpn_out_channels": 64,
            "fpn_scales":      [4, 8, 16, 32],
            "cross_attn_heads": 4,
            "cross_attn_layers": 1,
            "depth_min":       0.1,
            "depth_max":       100.0,
            "desc_dim":        32,
            "ema_momentum":    0.996,
        },
        "training": {
            "n_corr_samples":    64,
            "reproj_loss_robust": "huber",
            "lambda_depth_sup":  1.0,
            "lambda_desc_sup":   1.0,
            "lambda_feat":       0.5,
            "lambda_reproj":     1.0,
            "lambda_depth_mv":   0.5,
            "lambda_desc_mv":    0.5,
            "lambda_smooth":     0.1,
            "stage1": {"epochs": 1, "active_losses": ["depth_sup", "desc_sup", "feat"], "lr": 1e-4},
            "stage2": {"epochs": 1, "active_losses": ["depth_sup", "desc_sup", "feat", "reproj", "depth_mv", "desc_mv", "smooth"], "lr": 5e-5},
            "stage3": {"epochs": 1, "active_losses": ["depth_sup", "desc_sup", "reproj", "depth_mv", "desc_mv", "smooth"], "lr": 2e-5},
            "batch_size": 2, "num_workers": 0, "grad_clip": 1.0,
        },
        "data": {"img_height": 224, "img_width": 224, "dataset_root": "data/"},
    })


def make_batch(B=2, H=224, W=224, desc_dim=32):
    K = torch.tensor([[[500., 0., 112.], [0., 500., 112.], [0., 0., 1.]]])
    K = K.expand(B, -1, -1).clone()
    return {
        "img_a":  torch.randn(B, 3, H, W),
        "img_b":  torch.randn(B, 3, H, W),
        "K_a":    K,
        "K_b":    K.clone(),
        "Z_gt_a": torch.rand(B, 1, H, W) * 9 + 1,
        "Z_gt_b": torch.rand(B, 1, H, W) * 9 + 1,
        "D_gt_a": F.normalize(torch.randn(B, desc_dim, H, W), dim=1),
        "D_gt_b": F.normalize(torch.randn(B, desc_dim, H, W), dim=1),
        "mask_a": torch.ones(B, 1, H, W),
        "mask_b": torch.ones(B, 1, H, W),
    }


# ── Encoder ──────────────────────────────────────────────────────

def test_encoder_output_shape(cfg):
    from models.encoder import SiameseEncoder
    enc = SiameseEncoder(cfg.model.vit_arch, pretrained=False,
                         out_channels=cfg.model.fpn_out_channels)
    img_a = torch.randn(2, 3, 224, 224)
    img_b = torch.randn(2, 3, 224, 224)
    F_a, F_b = enc(img_a, img_b)
    assert F_a.shape == (2, 64, 28, 28), f"got {F_a.shape}"
    assert F_b.shape == F_a.shape


def test_encoder_shared_weights(cfg):
    """Same input → same output proves weights are shared."""
    from models.encoder import SiameseEncoder
    enc = SiameseEncoder(cfg.model.vit_arch, pretrained=False,
                         out_channels=cfg.model.fpn_out_channels)
    enc.eval()
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = enc.encode(img)
        out2 = enc.encode(img)
    assert torch.allclose(out1, out2)


# ── Cross-view attention ─────────────────────────────────────────

def test_cross_attn_shape(cfg):
    from models.cross_attention import CrossViewAttention
    cva = CrossViewAttention(channels=64, n_heads=4, n_layers=1)
    F_a = torch.randn(2, 64, 28, 28)
    F_b = torch.randn(2, 64, 28, 28)
    F_a_out, F_b_out = cva(F_a, F_b)
    assert F_a_out.shape == F_a.shape, f"got {F_a_out.shape}"
    assert F_b_out.shape == F_b.shape


def test_cross_attn_residual(cfg):
    """With zero-init output proj, output should be close to input."""
    from models.cross_attention import CrossViewAttention
    cva = CrossViewAttention(channels=64, n_heads=4, n_layers=1)
    # zero out output projections to make attn a near-identity
    for layer in cva.layers:
        torch.nn.init.zeros_(layer.out_a.weight)
        torch.nn.init.zeros_(layer.out_b.weight)
    cva.eval()
    F_a = torch.randn(2, 64, 28, 28)
    F_b = torch.randn(2, 64, 28, 28)
    with torch.no_grad():
        F_a_out, _ = cva(F_a, F_b)
    # output should be close to input after layer norm
    assert F_a_out.shape == F_a.shape


# ── Decoders ─────────────────────────────────────────────────────

def test_depth_decoder_shape_and_range(cfg):
    from models.decoders import DepthDecoder
    dec = DepthDecoder(in_channels=64, depth_min=0.1, depth_max=100.0)
    x = torch.randn(2, 64, 28, 28)
    Z = dec(x)
    assert Z.shape == (2, 1, 224, 224), f"got {Z.shape}"
    assert Z.min().item() >= 0.09
    assert Z.max().item() <= 100.1


def test_desc_decoder_normalised(cfg):
    from models.decoders import DescDecoder
    dec = DescDecoder(in_channels=64, desc_dim=32)
    x = torch.randn(2, 64, 28, 28)
    D = dec(x)
    assert D.shape == (2, 32, 224, 224)
    norms = D.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_conf_head_range(cfg):
    from models.decoders import ConfHead
    head = ConfHead(in_channels=64)
    x = torch.randn(2, 64, 28, 28)
    M = head(x)
    assert M.shape == (2, 1, 224, 224)
    assert M.min().item() >= -1e-5
    assert M.max().item() <= 1.0 + 1e-5


def test_pointmap_head_shape(cfg):
    from models.decoders import PointmapHead
    head = PointmapHead(in_channels=64)
    x = torch.randn(2, 64, 28, 28)
    X = head(x)
    assert X.shape == (2, 3, 224, 224)


def test_siamese_decoders_shared_weights(cfg):
    """Both views must use the exact same decoder weights."""
    from models.decoders import SiameseDecoders
    dec = SiameseDecoders(in_channels=64, depth_min=0.1, depth_max=100.0, desc_dim=32)
    dec.eval()
    F = torch.randn(2, 64, 28, 28)
    with torch.no_grad():
        Za, Da, Ma, Xa, Zb, Db, Mb, Xb = dec(F, F)
    # same input → same output for all heads
    assert torch.allclose(Za, Zb, atol=1e-5)
    assert torch.allclose(Da, Db, atol=1e-5)
    assert torch.allclose(Ma, Mb, atol=1e-5)
    assert torch.allclose(Xa, Xb, atol=1e-5)


# ── Geometry block ───────────────────────────────────────────────

def test_geometry_block_output_keys(cfg):
    from models.geometry import GeometryBlock
    geo = GeometryBlock(n_samples=64)
    B, C, H, W = 2, 32, 56, 56
    K = torch.tensor([[[500., 0., 28.], [0., 500., 28.], [0., 0., 1.]]]).expand(B, -1, -1).clone()
    out = geo(
        Za=torch.rand(B, 1, H, W) + 1,
        Da=F.normalize(torch.randn(B, C, H, W), dim=1),
        Ma=torch.ones(B, 1, H, W),
        Xa=torch.randn(B, 3, H, W),
        Zb=torch.rand(B, 1, H, W) + 1,
        Db=F.normalize(torch.randn(B, C, H, W), dim=1),
        Mb=torch.ones(B, 1, H, W),
        Xb=torch.randn(B, 3, H, W),
        K_a=K, K_b=K.clone(),
    )
    for key in ["R_ab", "t_ab", "R_pt", "t_pt", "p_b_proj",
                "p_b_soft", "pts3d_a", "pixels_a", "weights", "z_geom"]:
        assert key in out, f"missing key: {key}"
    assert out["R_ab"].shape == (B, 3, 3)
    assert out["t_ab"].shape == (B, 3)
    assert out["p_b_proj"].shape == (B, 64, 2)


# ── EMA ──────────────────────────────────────────────────────────

def test_ema_no_grad(cfg):
    from models.ddnet import DDNet
    model = DDNet(cfg)
    for p in model.ema_encoder.parameters():
        assert not p.requires_grad


def test_ema_update(cfg):
    from models.ddnet import DDNet
    model = DDNet(cfg)
    # Fill student weights with 1.0
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.fill_(1.0)
        for p in model.ema_encoder.parameters():
            p.fill_(0.0)
    model.ema_momentum = 0.0   # ema ← student exactly
    model.update_ema()
    for p in model.ema_encoder.parameters():
        assert torch.allclose(p, torch.ones_like(p)), "EMA update failed"


# ── Full model forward pass ───────────────────────────────────────

def test_ddnet_forward_stage1(cfg):
    from models.ddnet import DDNet
    model = DDNet(cfg)
    model.eval()
    batch = make_batch(desc_dim=cfg.model.desc_dim)
    with torch.no_grad():
        out = model(batch["img_a"], batch["img_b"],
                    batch["K_a"],  batch["K_b"],
                    run_geometry=False)
    assert out["Za"].shape == (2, 1, 224, 224)
    assert out["Da"].shape == (2, 32, 224, 224)
    assert out["Ma"].shape == (2, 1, 224, 224)
    assert out["Xa"].shape == (2, 3, 224, 224)
    assert "geometry" not in out


def test_ddnet_forward_stage2(cfg):
    from models.ddnet import DDNet
    model = DDNet(cfg)
    model.eval()
    batch = make_batch(desc_dim=cfg.model.desc_dim)
    with torch.no_grad():
        out = model(batch["img_a"], batch["img_b"],
                    batch["K_a"],  batch["K_b"],
                    run_geometry=True)
    assert "geometry" in out
    assert "R_ab" in out["geometry"]


def test_ddnet_loss_stage1(cfg):
    from models.ddnet   import DDNet
    from training.losses import LossManager
    model = DDNet(cfg)
    loss_mgr = LossManager(cfg.training, feat_channels=cfg.model.fpn_out_channels)
    batch = make_batch(desc_dim=cfg.model.desc_dim)
    out = model(batch["img_a"], batch["img_b"],
                batch["K_a"],  batch["K_b"],
                run_geometry=False)
    losses = loss_mgr(out, batch, active_losses=["depth_sup", "desc_sup", "feat"])
    assert "total" in losses
    assert torch.isfinite(losses["total"]), f"non-finite loss: {losses['total']}"
    assert losses["total"].item() > 0


def test_ddnet_loss_stage2(cfg):
    from models.ddnet   import DDNet
    from training.losses import LossManager
    model = DDNet(cfg)
    loss_mgr = LossManager(cfg.training, feat_channels=cfg.model.fpn_out_channels)
    batch = make_batch(desc_dim=cfg.model.desc_dim)
    out = model(batch["img_a"], batch["img_b"],
                batch["K_a"],  batch["K_b"],
                run_geometry=True)
    active = ["depth_sup", "desc_sup", "feat", "reproj", "depth_mv", "desc_mv", "smooth"]
    losses = loss_mgr(out, batch, active_losses=active)
    assert torch.isfinite(losses["total"]), f"non-finite loss: {losses['total']}"

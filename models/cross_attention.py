"""
Cross-view attention block.

NOT siamese — takes both views jointly.
Each view queries the other, updating its features with
context from the opposite view before decoding.

Input:  F_a, F_b  [B, C, H, W]
Output: F_a_cross, F_b_cross  [B, C, H, W]  (same shape, richer content)
"""

import torch
import torch.nn as nn
from einops import rearrange


class CrossViewAttentionLayer(nn.Module):
    """
    One cross-attention layer.
    View A queries view B, and view B queries view A.
    Both updates are residual so early training is stable.

    Args:
        channels : feature channel dim C
        n_heads  : number of attention heads
    """

    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        assert channels % n_heads == 0

        self.n_heads = n_heads
        self.scale = (channels // n_heads) ** -0.5

        # projections — separate for each direction
        self.q_a = nn.Linear(channels, channels)
        self.k_a = nn.Linear(channels, channels)
        self.v_a = nn.Linear(channels, channels)

        self.q_b = nn.Linear(channels, channels)
        self.k_b = nn.Linear(channels, channels)
        self.v_b = nn.Linear(channels, channels)

        self.out_a = nn.Linear(channels, channels)
        self.out_b = nn.Linear(channels, channels)

        # layer norms and FFNs
        self.norm_a1 = nn.LayerNorm(channels)
        self.norm_b1 = nn.LayerNorm(channels)
        self.norm_a2 = nn.LayerNorm(channels)
        self.norm_b2 = nn.LayerNorm(channels)

        self.ffn_a = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def _attn(self, q, k, v):
        """
        Scaled dot-product attention.
        q, k, v : [B, n_heads, N, head_dim]
        """
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, Nq, Nk]
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, v)                                 # [B, H, Nq, head_dim]

    def forward(
        self, F_a: torch.Tensor, F_b: torch.Tensor
    ) -> tuple:
        """
        Args:
            F_a, F_b : [B, C, H, W]
        Returns:
            F_a_out, F_b_out : [B, C, H, W]
        """
        B, C, H, W = F_a.shape
        N = H * W

        # flatten spatial → token sequence
        a = rearrange(F_a, "b c h w -> b (h w) c")   # [B, N, C]
        b = rearrange(F_b, "b c h w -> b (h w) c")

        h = self.n_heads
        hd = C // h

        def split_heads(x):
            return rearrange(x, "b n (h d) -> b h n d", h=h, d=hd)

        # ── A queries B ──
        qa = split_heads(self.q_a(a))
        kb = split_heads(self.k_b(b))
        vb = split_heads(self.v_b(b))
        attn_ab = rearrange(self._attn(qa, kb, vb), "b h n d -> b n (h d)")
        a = self.norm_a1(a + self.out_a(attn_ab))
        a = self.norm_a2(a + self.ffn_a(a))

        # ── B queries A ──
        qb = split_heads(self.q_b(b))
        ka = split_heads(self.k_a(a))
        va = split_heads(self.v_a(a))
        attn_ba = rearrange(self._attn(qb, ka, va), "b h n d -> b n (h d)")
        b = self.norm_b1(b + self.out_b(attn_ba))
        b = self.norm_b2(b + self.ffn_b(b))

        # reshape back to spatial
        F_a_out = rearrange(a, "b (h w) c -> b c h w", h=H, w=W)
        F_b_out = rearrange(b, "b (h w) c -> b c h w", h=H, w=W)
        return F_a_out, F_b_out


class CrossViewAttention(nn.Module):
    """
    Stack of N cross-view attention layers.

    Args:
        channels : feature channel dim C
        n_heads  : attention heads per layer
        n_layers : number of stacked layers
    """

    def __init__(self, channels: int, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossViewAttentionLayer(channels, n_heads)
            for _ in range(n_layers)
        ])

    def forward(
        self, F_a: torch.Tensor, F_b: torch.Tensor
    ) -> tuple:
        """
        Args:
            F_a, F_b : [B, C, H, W]
        Returns:
            F_a_cross, F_b_cross : [B, C, H, W]
        """
        for layer in self.layers:
            F_a, F_b = layer(F_a, F_b)
        return F_a, F_b

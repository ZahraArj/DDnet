"""
Siamese ViT + FPN encoder.

One instance shared across both views — call .encode(img) per view.
Shared weights guarantee F_a and F_b live in the same feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck.
    Fuses 4 ViT intermediate feature levels into a single stride-8 map.

    Args:
        in_channels_list : channel dims of the 4 ViT levels
        out_channels     : unified output channel dim (C)
    """

    def __init__(self, in_channels_list: list, out_channels: int):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels_list
        ])
        self.outputs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features : list[Tensor] — 4 levels, finest to coarsest
        Returns:
            [B, C, H/8, W/8]
        """
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        # top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        # return stride-8 level
        return self.outputs[1](laterals[1])


class SiameseEncoder(nn.Module):
    """
    Siamese ViT-B/16 + FPN encoder.
    Same weights called for both views.

    Args:
        vit_arch     : timm model name
        pretrained   : load timm pretrained weights
        out_channels : FPN output channels (C)
    """

    def __init__(
        self,
        vit_arch: str = "vit_base_patch16_224",
        pretrained: bool = True,
        out_channels: int = 256,
        img_size: tuple = (384, 512),
    ):
        super().__init__()
        self.backbone = timm.create_model(
            vit_arch,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 6, 9, 11),
            img_size=(384, 512),
        )
        # infer input channel dims
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size[0], img_size[1])
            feats = self.backbone(dummy)
            in_channels = [f.shape[1] for f in feats]

        self.fpn = FPNNeck(in_channels, out_channels)
        self.out_channels = out_channels

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode one image.
        Args:  img [B, 3, H, W]
        Returns: [B, C, H/8, W/8]
        """
        return self.fpn(self.backbone(img))

    def forward(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> tuple:
        """
        Encode both views with shared weights.
        Returns: F_a [B,C,H/8,W/8],  F_b [B,C,H/8,W/8]
        """
        return self.encode(img_a), self.encode(img_b)

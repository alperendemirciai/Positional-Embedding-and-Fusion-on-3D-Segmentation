from typing import Tuple

import torch
import torch.nn as nn

from models.base_unet import UNet3D
from models.fusion_head import build_fusion_head
from models.pe_modules import FiLMPE


MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]


class V2SeparateBackbones(nn.Module):
    """Late-fusion with four independent per-modality U-Net backbones.

    Each backbone is a modality specialist that only ever receives one channel.
    Backbones are forwarded sequentially to stay within GPU memory budget.

    Input channels per backbone:
        pe_type='none'   → 1 channel  (modality only)
        pe_type='film'   → 1 channel  (modality only; FiLM injects position)
        pe_type='concat' → 4 channels (1 modality + 3 coord grids)

    Missing-modality inference: pass active_modalities=(0, 2) etc.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        pe_cfg     = cfg.get("pe", {})
        model_cfg  = cfg.get("model", {})
        fusion_cfg = cfg.get("fusion", {})
        self.pe_type = pe_cfg.get("type", "none")

        in_ch = 4 if self.pe_type == "concat" else 1

        self.backbones = nn.ModuleDict()
        for name in MODALITY_NAMES:
            film_cond = None
            if self.pe_type == "film":
                bottleneck_ch = model_cfg.get("base_channels", 16) * (
                    2 ** model_cfg.get("depth", 4))
                film_cond = FiLMPE(
                    num_freqs=pe_cfg.get("sinusoidal_num_freqs", 5),
                    bottleneck_ch=bottleneck_ch,
                    hidden_dim=pe_cfg.get("film_hidden_dim", 128),
                )
            self.backbones[name] = UNet3D(
                in_channels=in_ch,
                out_channels=3,
                base_channels=model_cfg.get("base_channels", 16),
                depth=model_cfg.get("depth", 4),
                pe_type=self.pe_type,
                film_conditioner=film_cond,
            )

        self.fusion = build_fusion_head(
            strategy=fusion_cfg.get("strategy", "mean"),
            num_modalities=4,
            num_classes=3,
        )

    def forward(self, modalities: torch.Tensor,
                patch_center=None, coord_channels=None,
                active_modalities: Tuple[int, ...] = (0, 1, 2, 3)):
        logit_list = []
        for i in active_modalities:
            name = MODALITY_NAMES[i]
            x = modalities[:, i:i + 1, ...]           # (B, 1, H, W, D)
            if self.pe_type == "concat":
                x = torch.cat([x, coord_channels], dim=1)   # (B, 4, H, W, D)
            logit_list.append(self.backbones[name](x, patch_center))

        strategy = self.fusion.__class__.__name__
        if strategy == "MeanFusion":
            return self.fusion(logit_list)
        else:
            return self.fusion(logit_list, active_indices=active_modalities)

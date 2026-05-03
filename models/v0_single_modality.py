import torch
import torch.nn as nn

from models.base_unet import UNet3D


MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]


class V0SingleModality(nn.Module):
    """Single-modality 3D U-Net baseline.

    Receives the standard (B, 4, H, W, D) input from the data pipeline but
    discards all channels except the one specified by ``modality_index``.
    No positional encoding.

    Config fields read from cfg["model"]:
        modality_index  int  0=T1, 1=T1ce, 2=T2, 3=FLAIR  (default: 0)
        base_channels   int  (default: 16)
        depth           int  (default: 4)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get("model", {})
        self.modality_index = model_cfg.get("modality_index", 0)
        assert 0 <= self.modality_index <= 3, (
            f"modality_index must be 0–3, got {self.modality_index}"
        )
        self.unet = UNet3D(
            in_channels=1,
            out_channels=3,
            base_channels=model_cfg.get("base_channels", 16),
            depth=model_cfg.get("depth", 4),
            pe_type="none",
            film_conditioner=None,
        )

    def forward(self, modalities: torch.Tensor,
                patch_center=None, coord_channels=None):
        x = modalities[:, self.modality_index:self.modality_index + 1, ...]
        return self.unet(x)

import torch
import torch.nn as nn

from models.base_unet import UNet3D
from models.pe_modules import FiLMPE


class V0EarlyFusion(nn.Module):
    """Early-fusion 3D U-Net baseline.

    All 4 modalities are concatenated along the channel dimension before the
    network. Fusion is implicit in the first convolutional layer.

    pe_type:
        'none'   – 4-channel input, no PE
        'film'   – 4-channel input + FiLM conditioning at bottleneck
        'concat' – 7-channel input (4 modalities + 3 coord grids)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        pe_cfg      = cfg.get("pe", {})
        model_cfg   = cfg.get("model", {})
        self.pe_type = pe_cfg.get("type", "none")

        in_ch = 7 if self.pe_type == "concat" else 4

        film_cond = None
        if self.pe_type == "film":
            bottleneck_ch = model_cfg.get("bottleneck_channels", 256)
            film_cond = FiLMPE(
                num_freqs=pe_cfg.get("sinusoidal_num_freqs", 5),
                bottleneck_ch=bottleneck_ch,
                hidden_dim=pe_cfg.get("film_hidden_dim", 128),
            )

        self.unet = UNet3D(
            in_channels=in_ch,
            out_channels=3,
            base_channels=model_cfg.get("base_channels", 16),
            depth=model_cfg.get("depth", 4),
            pe_type=self.pe_type,
            film_conditioner=film_cond,
        )

    def forward(self, modalities: torch.Tensor,
                patch_center=None, coord_channels=None):
        # modalities: (B, 4, H, W, D)
        if self.pe_type == "concat":
            x = torch.cat([modalities, coord_channels], dim=1)  # (B, 7, ...)
        else:
            x = modalities
        return self.unet(x, patch_center)

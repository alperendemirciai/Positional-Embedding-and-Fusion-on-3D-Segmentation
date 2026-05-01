import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net with optional FiLM conditioning at the bottleneck.

    Args:
        in_channels:        number of input channels
        out_channels:       number of output channels (3 for WT/TC/ET)
        base_channels:      channels at the first encoder level (doubles each level)
        depth:              number of encoder levels (bottleneck = depth + 1)
        pe_type:            'none' | 'film' | 'concat'
        film_conditioner:   external FiLMConditioner module (injected from pe_modules)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        base_channels: int = 16,
        depth: int = 4,
        pe_type: str = "none",
        film_conditioner=None,
    ):
        super().__init__()
        self.pe_type = pe_type
        self.film_conditioner = film_conditioner

        ch = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoders.append(ConvBlock(in_channels, ch[0]))
        for i in range(1, depth + 1):
            self.pools.append(nn.MaxPool3d(2))
            self.encoders.append(ConvBlock(ch[i - 1], ch[i]))

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(ch[i + 1], ch[i], kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(ch[i] * 2, ch[i]))

        self.out_conv = nn.Conv3d(ch[0], out_channels, kernel_size=1)

    def forward(self, x, patch_center=None):
        skips = []
        for i, enc in enumerate(self.encoders[:-1]):
            x = enc(x)
            skips.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.encoders[-1](x)

        if self.pe_type == "film" and self.film_conditioner is not None:
            x = self.film_conditioner(x, patch_center)

        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            # Handle size mismatch from odd input dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)

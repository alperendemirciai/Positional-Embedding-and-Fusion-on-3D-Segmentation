import math
import numpy as np
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Encodes a (B, 3) normalised coordinate vector using sinusoidal features.

    For each of the 3 coordinates, applies sin and cos at num_freqs frequencies
    (π·2^0 … π·2^(L-1)), producing a (B, 3·2·L) output vector.
    """

    def __init__(self, num_freqs: int = 5):
        super().__init__()
        freqs = [math.pi * (2 ** i) for i in range(num_freqs)]
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, 3)
        feats = []
        for f in self.freqs:
            feats.append(torch.sin(f * coords))
            feats.append(torch.cos(f * coords))
        return torch.cat(feats, dim=-1)  # (B, 3*2*num_freqs)

    @property
    def out_dim(self) -> int:
        return 3 * 2 * len(self.freqs)


class FiLMConditioner(nn.Module):
    """Maps a PE vector to per-channel scale (γ) and shift (β) for FiLM.

    Applied to bottleneck features as: features = γ * features + β
    """

    def __init__(self, pe_dim: int, bottleneck_ch: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * bottleneck_ch),
        )
        # Initialise so FiLM starts as identity (γ=1, β=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[:bottleneck_ch] = 1.0  # γ bias → 1

    def forward(self, features: torch.Tensor, pe_vec: torch.Tensor) -> torch.Tensor:
        # features: (B, C, H, W, D)
        # pe_vec:   (B, pe_dim)
        params = self.mlp(pe_vec)                           # (B, 2C)
        gamma, beta = params.chunk(2, dim=-1)               # each (B, C)
        gamma = gamma.view(gamma.shape[0], -1, 1, 1, 1)
        beta  = beta.view(beta.shape[0], -1, 1, 1, 1)
        return gamma * features + beta


class FiLMPE(nn.Module):
    """Combines SinusoidalPE + FiLMConditioner into a single module."""

    def __init__(self, num_freqs: int = 5, bottleneck_ch: int = 256,
                 hidden_dim: int = 128):
        super().__init__()
        self.pe = SinusoidalPE(num_freqs)
        self.conditioner = FiLMConditioner(self.pe.out_dim, bottleneck_ch, hidden_dim)

    def forward(self, features: torch.Tensor, patch_center: torch.Tensor) -> torch.Tensor:
        pe_vec = self.pe(patch_center)
        return self.conditioner(features, pe_vec)


def make_coord_channels(
    patch_center: torch.Tensor,
    patch_size,
    vol_size=(240, 240, 155),
) -> torch.Tensor:
    """Build (B, 3, H, W, D) normalised coordinate grids for the current patch.

    patch_center: (B, 3) — normalised centre in [0, 1]
    patch_size:   (H, W, D)
    vol_size:     full volume dimensions (used to normalise coordinates)
    """
    B = patch_center.shape[0]
    H, W, D = patch_size

    coords = []
    for b in range(B):
        cx, cy, cz = patch_center[b].tolist()
        x0 = cx * vol_size[0] - H / 2
        y0 = cy * vol_size[1] - W / 2
        z0 = cz * vol_size[2] - D / 2

        xs = (x0 + np.arange(H)) / vol_size[0]
        ys = (y0 + np.arange(W)) / vol_size[1]
        zs = (z0 + np.arange(D)) / vol_size[2]

        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        grid = np.stack([xg, yg, zg], axis=0).astype(np.float32)
        coords.append(torch.from_numpy(grid))

    return torch.stack(coords, dim=0).to(patch_center.device)  # (B, 3, H, W, D)

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFusion(nn.Module):
    """Simple equal-weight average of per-modality logit maps. No parameters."""

    def forward(self, logit_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(logit_list, dim=0).mean(dim=0)


class WeightedFusion(nn.Module):
    """Learnable scalar weight per modality (softmax-normalised).

    Handles missing modalities naturally: only weights for active modalities
    are selected and re-normalised via softmax.
    """

    def __init__(self, num_modalities: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities))

    def forward(self, logit_list: List[torch.Tensor],
                active_indices: Tuple[int, ...] = (0, 1, 2, 3)) -> torch.Tensor:
        assert len(active_indices) == len(logit_list), (
            f"Length mismatch: {len(active_indices)} indices, {len(logit_list)} logits"
        )
        w = F.softmax(self.weights[list(active_indices)], dim=0)  # (n_active,)
        stacked = torch.stack(logit_list, dim=0)                   # (n_active, B, C, H, W, D)
        return (w[:, None, None, None, None, None] * stacked).sum(dim=0)


class AttentionFusion(nn.Module):
    """Voxel-wise attention over concatenated logit maps.

    A small 1×1×1 conv predicts per-voxel attention weights for each modality,
    then returns the weighted sum.
    """

    def __init__(self, num_modalities: int = 4, num_classes: int = 3):
        super().__init__()
        in_ch = num_modalities * num_classes
        self.attn_conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch * 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch * 4, num_modalities, kernel_size=1),
        )
        self.num_modalities = num_modalities
        self.num_classes = num_classes

    def forward(self, logit_list: List[torch.Tensor],
                active_indices: Tuple[int, ...] = (0, 1, 2, 3)) -> torch.Tensor:
        assert len(active_indices) == len(logit_list), (
            f"Length mismatch: {len(active_indices)} indices, {len(logit_list)} logits"
        )
        assert all(0 <= i < self.num_modalities for i in active_indices), (
            f"active_indices out of range [0, {self.num_modalities})"
        )
        assert len(set(active_indices)) == len(active_indices), (
            "Duplicate indices in active_indices"
        )
        n = len(logit_list)
        stacked = torch.stack(logit_list, dim=1)        # (B, n_active, C, H, W, D)
        B, _, C, H, W, D = stacked.shape

        # Build full-size attention input with logits at their correct modality slots.
        # Appending zeros at the end (old approach) is wrong: it places e.g. T2 at
        # the T1ce slot when T1ce is missing, corrupting learned attention weights.
        if n < self.num_modalities:
            full = torch.zeros(B, self.num_modalities, C, H, W, D,
                               device=stacked.device, dtype=stacked.dtype)
            for slot, src_idx in enumerate(active_indices):
                full[:, src_idx] = stacked[:, slot]
        else:
            full = stacked

        cat = full.view(B, self.num_modalities * C, H, W, D)
        attn = F.softmax(self.attn_conv(cat), dim=1)    # (B, num_modalities, H, W, D)

        # Only keep attention weights for active modalities and re-normalise
        active_attn = attn[:, list(active_indices), ...]  # (B, n_active, H, W, D)
        active_attn = active_attn / (active_attn.sum(dim=1, keepdim=True) + 1e-8)

        fused = (active_attn.unsqueeze(2) * stacked).sum(dim=1)  # (B, C, H, W, D)
        return fused


def build_fusion_head(strategy: str, num_modalities: int = 4,
                      num_classes: int = 3) -> nn.Module:
    if strategy == "mean":
        return MeanFusion()
    elif strategy == "weighted":
        return WeightedFusion(num_modalities)
    elif strategy == "attention":
        return AttentionFusion(num_modalities, num_classes)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy!r}. "
                         f"Choose from: mean, weighted, attention")

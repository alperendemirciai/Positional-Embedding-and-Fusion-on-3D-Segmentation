import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd,
    MapTransform,
)
from monai.config import KeysCollection


MODALITY_KEYS = ["t1", "t1ce", "t2", "flair"]
ALL_KEYS = MODALITY_KEYS + ["seg"]
VOL_SIZE = (240, 240, 155)


def _find_crop_start(vol) -> list:
    """Accumulate all SpatialCrop roi_start offsets from a MONAI MetaTensor.

    In MONAI 1.x each spatial crop (CropForegroundd, RandCropByPosNegLabeld,
    etc.) records itself in tensor.meta["applied_operations"] with
    extra_info["roi_start"].  Both crops must be summed to recover the voxel
    position in the original 240×240×155 volume:

        global_start = CropForegroundd.roi_start + RandCropByPosNegLabeld.roi_start

    Returns a 3-element int list, or None when metadata is unavailable.
    """
    if not hasattr(vol, "meta"):
        return None
    total = [0, 0, 0]
    found = False
    for op in vol.meta.get("applied_operations", []):
        roi_start = op.get("extra_info", {}).get("roi_start")
        if roi_start is not None:
            for i, v in enumerate(roi_start):
                total[i] += int(v)
            found = True
    return total if found else None


class RecordPatchCenterd(MapTransform):
    """Reads crop metadata from a MONAI MetaTensor and stores the normalised
    patch centre in batch["patch_center"] as a (3,) float32 tensor."""

    def __init__(self, ref_key: str = "t1", vol_size=VOL_SIZE):
        super().__init__(keys=[ref_key], allow_missing_keys=True)
        self.ref_key = ref_key
        self.vol_size = vol_size

    def __call__(self, data):
        d = dict(data)
        vol = d[self.ref_key]
        start = _find_crop_start(vol)

        if start is not None:
            patch_size = vol.shape[-3:]
            cx = (start[0] + patch_size[0] / 2) / self.vol_size[0]
            cy = (start[1] + patch_size[1] / 2) / self.vol_size[1]
            cz = (start[2] + patch_size[2] / 2) / self.vol_size[2]
            d["patch_center"] = torch.tensor([cx, cy, cz], dtype=torch.float32)
        else:
            d["patch_center"] = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        return d


class AddCoordChannelsd(MapTransform):
    """Generates 3 normalised coordinate grids for the current patch and stores
    them in batch["coord_channels"] as a (3, H, W, D) float32 tensor."""

    def __init__(self, ref_key: str = "t1", vol_size=VOL_SIZE):
        super().__init__(keys=[ref_key], allow_missing_keys=True)
        self.ref_key = ref_key
        self.vol_size = vol_size

    def __call__(self, data):
        d = dict(data)
        vol = d[self.ref_key]
        start = _find_crop_start(vol) or [0, 0, 0]

        H, W, D = vol.shape[-3], vol.shape[-2], vol.shape[-1]
        xs = (start[0] + np.arange(H)) / self.vol_size[0]
        ys = (start[1] + np.arange(W)) / self.vol_size[1]
        zs = (start[2] + np.arange(D)) / self.vol_size[2]
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        coords = np.stack([xg, yg, zg], axis=0).astype(np.float32)
        d["coord_channels"] = torch.from_numpy(coords)
        return d


def build_file_list(split_ids: List[str], data_root: str) -> List[dict]:
    files = []
    for pid in split_ids:
        base = Path(data_root) / pid / pid
        files.append({
            "t1":        str(base) + "_t1.nii.gz",
            "t1ce":      str(base) + "_t1ce.nii.gz",
            "t2":        str(base) + "_t2.nii.gz",
            "flair":     str(base) + "_flair.nii.gz",
            "seg":       str(base) + "_seg.nii.gz",
            "patient_id": pid,
        })
    return files


def _base_transforms() -> List:
    return [
        LoadImaged(keys=ALL_KEYS, image_only=False),
        EnsureChannelFirstd(keys=ALL_KEYS),
        NormalizeIntensityd(keys=MODALITY_KEYS, nonzero=True, channel_wise=True),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"]),
        CropForegroundd(keys=ALL_KEYS, source_key="t1", allow_smaller=True),
    ]


def build_train_transform(patch_size, num_samples: int = 2,
                          pe_type: str = "none") -> Compose:
    transforms = _base_transforms() + [
        SpatialPadd(keys=ALL_KEYS, spatial_size=patch_size, mode="constant", method="end"),
        RandCropByPosNegLabeld(
            keys=ALL_KEYS,
            label_key="seg",
            spatial_size=patch_size,
            pos=1, neg=1,
            num_samples=num_samples,
            image_key="t1",
            image_threshold=0,
        ),
        RandFlipd(keys=ALL_KEYS, spatial_axis=0, prob=0.5),
        RandFlipd(keys=ALL_KEYS, spatial_axis=1, prob=0.5),
        RandFlipd(keys=ALL_KEYS, spatial_axis=2, prob=0.5),
        RandRotate90d(keys=ALL_KEYS, prob=0.2, max_k=3),
        RandScaleIntensityd(keys=MODALITY_KEYS, factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=MODALITY_KEYS, offsets=0.1, prob=0.15),
        RandGaussianNoised(keys=MODALITY_KEYS, std=0.05, prob=0.1),
    ]
    if pe_type in ("film", "concat"):
        transforms.append(RecordPatchCenterd(ref_key="t1"))
    if pe_type == "concat":
        transforms.append(AddCoordChannelsd(ref_key="t1"))
    transforms.append(ToTensord(keys=ALL_KEYS))
    return Compose(transforms)


def build_val_transform(pe_type: str = "none") -> Compose:
    transforms = _base_transforms()
    transforms.append(ToTensord(keys=ALL_KEYS))
    return Compose(transforms)


def stack_modalities(batch: dict) -> torch.Tensor:
    """Concatenate the 4 separate modality tensors into (B, 4, H, W, D)."""
    return torch.cat([batch[k] for k in MODALITY_KEYS], dim=1)


def build_datasets(cfg: dict, pe_type: str = "none"):
    splits_file = cfg["data"]["splits_file"]
    data_root   = cfg["data"]["root"]
    patch_size  = cfg["data"]["patch_size"]
    num_samples = cfg["training"]["num_patches_per_sample"]
    cache_rate  = cfg["data"].get("cache_rate", 0.0)
    num_workers = cfg["data"].get("num_workers", 4)

    with open(splits_file) as f:
        splits = json.load(f)

    train_files = build_file_list(splits["train"], data_root)
    val_files   = build_file_list(splits["val"],   data_root)
    test_files  = build_file_list(splits["test"],  data_root)

    train_tf = build_train_transform(patch_size, num_samples, pe_type)
    val_tf   = build_val_transform(pe_type)

    DS = CacheDataset if cache_rate > 0 else Dataset
    kwargs = {"cache_rate": cache_rate, "num_workers": num_workers} if cache_rate > 0 else {}

    train_ds = DS(data=train_files, transform=train_tf, **kwargs)
    val_ds   = DS(data=val_files,   transform=val_tf,   **kwargs)
    test_ds  = DS(data=test_files,  transform=val_tf,   **kwargs)

    return train_ds, val_ds, test_ds

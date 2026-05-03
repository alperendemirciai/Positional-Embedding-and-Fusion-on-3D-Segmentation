"""Sliding-window inference and evaluation for a single experiment."""

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from data_utils.brats_dataset import MODALITY_KEYS, VOL_SIZE, stack_modalities, _find_crop_start
from models.pe_modules import make_coord_channels


REGION_NAMES = ["WT", "TC", "ET"]

# All 15 non-empty subsets of 4 modalities
ALL_MODALITY_SUBSETS: List[Tuple[int, ...]] = []
for size in range(1, 5):
    for combo in combinations(range(4), size):
        ALL_MODALITY_SUBSETS.append(combo)

SUBSET_KEYS = {
    combo: "+".join(MODALITY_KEYS[i] for i in combo)
    for combo in ALL_MODALITY_SUBSETS
}


def _sanitize(obj):
    """Recursively replace float nan/inf with None so json.dump never raises."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _pred_to_label_map(pred: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W, D) binary prediction (WT/TC/ET) to a single-channel
    label map with BraTS label encoding {0, 1, 2, 4}."""
    pred_np = pred.cpu().numpy().astype(bool)
    label = np.zeros(pred_np.shape[1:], dtype=np.uint8)
    # ED  = WT & ~TC  (label 2)
    label[pred_np[0] & ~pred_np[1]] = 2
    # NCR = TC & ~ET  (label 1)
    label[pred_np[1] & ~pred_np[2]] = 1
    # ET  = ET        (label 4)
    label[pred_np[2]] = 4
    return label


def _save_prediction(label_map: np.ndarray, patient_id: str,
                     ref_nifti_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(ref_nifti_path)
    img = nib.Nifti1Image(label_map, affine=ref.affine, header=ref.header)
    nib.save(img, str(out_dir / f"{patient_id}_pred.nii.gz"))


def _build_gaussian_importance_map(patch_size, sigma_scale: float, device) -> torch.Tensor:
    """Gaussian importance map matching MONAI's sigma_scale convention."""
    coords = [
        torch.arange(p, dtype=torch.float32, device=device) - (p - 1) / 2.0
        for p in patch_size
    ]
    grids = torch.meshgrid(*coords, indexing="ij")
    imp = torch.exp(
        -0.5 * sum((g / (sigma_scale * p)) ** 2 for g, p in zip(grids, patch_size))
    )
    return imp.clamp(min=1e-6)


@torch.no_grad()
def run_inference(
    model,
    patient_data: dict,
    cfg: dict,
    device: torch.device,
    active_modalities: Tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Run full-volume sliding-window inference for one patient.

    Returns logits of shape (1, 3, H, W, D).

    Each sliding window receives its actual normalised position in the original
    240×240×155 volume as PE conditioning, derived from the CropForeground
    offset stored in patient_data["t1"].meta plus the window's position inside
    the cropped volume.
    """
    pe_type       = cfg.get("pe", {}).get("type", "none")
    patch_size    = tuple(cfg["data"]["patch_size"])
    inf_cfg       = cfg.get("inference", {})
    overlap       = inf_cfg.get("sw_overlap", 0.5)
    sw_batch_size = inf_cfg.get("sw_batch_size", 4)
    sigma_scale   = inf_cfg.get("sigma_scale", 0.125)
    mode          = inf_cfg.get("sw_mode", "gaussian")
    model_variant = cfg.get("model", {}).get("variant", "V0EarlyFusion")
    is_v0         = "V0" in model_variant

    # Foreground crop offset — needed to compute global voxel coordinates
    fg_offset = _find_crop_start(patient_data["t1"])
    if fg_offset is None:
        fg_offset = [0, 0, 0]

    # Stack modalities: each key has shape (1, H, W, D) in a test sample.
    x = torch.stack(
        [patient_data[k].squeeze(0) for k in ["t1", "t1ce", "t2", "flair"]], dim=0
    ).unsqueeze(0).to(device)  # (1, 4, H, W, D)

    _, C, H, W, D = x.shape

    # Pad so every dimension covers at least one full patch
    pad_h = max(0, patch_size[0] - H)
    pad_w = max(0, patch_size[1] - W)
    pad_d = max(0, patch_size[2] - D)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
    _, C, Hp, Wp, Dp = x.shape

    # Sliding window start positions
    scan = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)
    def _starts(size, p_size, stride):
        s = list(range(0, size - p_size + 1, stride))
        if not s or s[-1] + p_size < size:
            s.append(size - p_size)
        return s

    starts_h = _starts(Hp, patch_size[0], scan[0])
    starts_w = _starts(Wp, patch_size[1], scan[1])
    starts_d = _starts(Dp, patch_size[2], scan[2])
    positions = [(h, w, d) for h in starts_h for w in starts_w for d in starts_d]

    imp = (
        _build_gaussian_importance_map(patch_size, sigma_scale, device)
        if mode == "gaussian"
        else torch.ones(patch_size, dtype=torch.float32, device=device)
    )

    output    = torch.zeros(1, 3, Hp, Wp, Dp, device=device)
    count_map = torch.zeros(1, 1, Hp, Wp, Dp, device=device)

    model.eval()
    for batch_start in range(0, len(positions), sw_batch_size):
        batch_pos = positions[batch_start:batch_start + sw_batch_size]
        patches = torch.stack([
            x[0, :, h:h + patch_size[0], w:w + patch_size[1], d:d + patch_size[2]]
            for h, w, d in batch_pos
        ], dim=0)  # (n, C, pH, pW, pD)

        pc = cc = None
        if pe_type in ("film", "concat"):
            centers = [
                [
                    (fg_offset[0] + h + patch_size[0] / 2) / VOL_SIZE[0],
                    (fg_offset[1] + w + patch_size[1] / 2) / VOL_SIZE[1],
                    (fg_offset[2] + d + patch_size[2] / 2) / VOL_SIZE[2],
                ]
                for h, w, d in batch_pos
            ]
            pc = torch.tensor(centers, dtype=torch.float32, device=device)
        if pe_type == "concat":
            cc = make_coord_channels(pc, patch_size, VOL_SIZE)

        if is_v0:
            out = model(patches, patch_center=pc, coord_channels=cc)
        else:
            out = model(patches, patch_center=pc, coord_channels=cc,
                        active_modalities=active_modalities)

        for j, (h, w, d) in enumerate(batch_pos):
            output[0, :, h:h + patch_size[0], w:w + patch_size[1], d:d + patch_size[2]] += \
                out[j] * imp.unsqueeze(0)
            count_map[0, 0, h:h + patch_size[0], w:w + patch_size[1], d:d + patch_size[2]] += imp

    result = output / count_map.clamp(min=1e-8)
    return result[:, :, :H, :W, :D]


def _infer_and_score(
    model, sample: dict, cfg: dict, device: torch.device,
    active_modalities: Tuple[int, ...],
    dice_metric: DiceMetric,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Run inference once, accumulate into dice_metric, return (pred, target, scores).

    Centralises the inference+scoring logic so neither the main loop nor the
    missing-modality loop need to call run_inference twice.
    """
    threshold = cfg.get("inference", {}).get("threshold", 0.5)
    logits = run_inference(model, sample, cfg, device, active_modalities)
    pred   = (logits.sigmoid() > threshold).long()
    target = sample["seg"].unsqueeze(0).long().to(device)

    dice_metric(y_pred=pred, y=target)

    scores: Dict[str, float] = {}
    for i, r in enumerate(REGION_NAMES):
        p = pred[0, i].float()
        t = target[0, i].float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            scores[r] = 1.0  # both pred and GT absent — perfect agreement
        else:
            scores[r] = (2 * inter / union).item()

    return pred, target, scores


def run_evaluation(
    model,
    test_ds,
    cfg: dict,
    device: torch.device,
    experiment_name: str,
    missing_modality: bool = False,
) -> dict:
    """Evaluate model on the full test set and optionally on all modality subsets.

    Saves per-patient predictions as .nii.gz and returns a results dict.
    """
    results_dir  = Path(cfg.get("logging", {}).get("results_dir", "results"))
    pred_dir     = results_dir / "predictions" / experiment_name
    save_preds   = cfg.get("inference", {}).get("save_predictions", True)
    compute_hd95 = cfg.get("evaluation", {}).get("compute_hd95", False)
    data_root    = cfg["data"]["root"]

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch",
                             get_not_nans=False)
    hd95_metric = (
        HausdorffDistanceMetric(include_background=True, percentile=95,
                                reduction="mean_batch", get_not_nans=False)
        if compute_hd95 else None
    )

    per_patient: Dict[str, dict] = {}

    for idx in range(len(test_ds)):
        sample     = test_ds[idx]
        patient_id = sample.get("patient_id", f"patient_{idx:04d}")
        if isinstance(patient_id, list):
            patient_id = patient_id[0]

        pred, target, scores = _infer_and_score(
            model, sample, cfg, device, (0, 1, 2, 3), dice_metric
        )
        per_patient[patient_id] = scores

        if hd95_metric is not None:
            hd95_metric(y_pred=pred, y=target)

        if save_preds:
            label_map = _pred_to_label_map(pred[0])
            ref_path  = str(
                Path(data_root) / patient_id / f"{patient_id}_t1.nii.gz"
            )
            _save_prediction(label_map, patient_id, ref_path, pred_dir)

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(test_ds)}] {patient_id} "
                  f"WT={scores['WT']:.3f} TC={scores['TC']:.3f} ET={scores['ET']:.3f}")

    # Aggregate full-modality scores
    agg_dice = dice_metric.aggregate().tolist()
    dice_metric.reset()
    mean_scores = {r: agg_dice[i] for i, r in enumerate(REGION_NAMES)}

    agg_hd95 = None
    if hd95_metric is not None:
        agg_hd95 = hd95_metric.aggregate().tolist()
        hd95_metric.reset()

    # ---- Missing-modality evaluation (V1 / V2 only) ----
    missing_results: dict = {}
    if missing_modality:
        print("\n  Running missing-modality evaluation …")
        for subset in ALL_MODALITY_SUBSETS:
            if subset == (0, 1, 2, 3):
                continue
            key = SUBSET_KEYS[subset]
            dm  = DiceMetric(include_background=True, reduction="mean_batch",
                             get_not_nans=False)
            for idx in range(len(test_ds)):
                sample = test_ds[idx]
                pred, _, _ = _infer_and_score(
                    model, sample, cfg, device, subset, dm
                )
                if save_preds:
                    lm  = _pred_to_label_map(pred[0])
                    pid = sample.get("patient_id", f"patient_{idx:04d}")
                    if isinstance(pid, list):
                        pid = pid[0]
                    ref_path = str(
                        Path(data_root) / pid / f"{pid}_t1.nii.gz"
                    )
                    _save_prediction(lm, pid, ref_path,
                                     pred_dir / "missing_modality" / key)
            agg = dm.aggregate().tolist()
            dm.reset()
            missing_results[key] = {r: agg[i] for i, r in enumerate(REGION_NAMES)}
            print(f"    {key:20s}  WT={agg[0]:.3f} TC={agg[1]:.3f} ET={agg[2]:.3f}")

    # Per-region std across patients
    std_scores: Dict[str, float] = {
        r: float(np.std([v[r] for v in per_patient.values()]))
        for r in REGION_NAMES
    }

    results = {
        "per_patient":      per_patient,
        "mean":             mean_scores,
        "std":              std_scores,
        "hd95_mean":        {r: agg_hd95[i] for i, r in enumerate(REGION_NAMES)}
                            if agg_hd95 else None,
        "missing_modality": missing_results,
    }

    out_path = results_dir / f"{experiment_name}_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    print(f"\n  Results saved to {out_path}")
    return results

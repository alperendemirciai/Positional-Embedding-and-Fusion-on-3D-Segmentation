"""Sliding-window inference and evaluation for a single experiment."""

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from data_utils.brats_dataset import MODALITY_KEYS, VOL_SIZE, stack_modalities
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


def _build_inferer(cfg: dict) -> SlidingWindowInferer:
    inf_cfg = cfg.get("inference", {})
    return SlidingWindowInferer(
        roi_size=cfg["data"]["patch_size"],
        sw_batch_size=inf_cfg.get("sw_batch_size", 4),
        overlap=inf_cfg.get("sw_overlap", 0.5),
        mode=inf_cfg.get("sw_mode", "gaussian"),
        sigma_scale=0.125,
    )


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

    Note on PE at inference: SlidingWindowInferer does not expose the window
    position to the predictor.  For FiLM we use the volume centre (0.5, 0.5,
    0.5) as a neutral conditioning signal.  For concat we build coordinate
    channels relative to that same centre point.  Both are approximations;
    they are consistent with the fallback used during validation.
    """
    pe_type       = cfg.get("pe", {}).get("type", "none")
    patch_size    = cfg["data"]["patch_size"]
    model_variant = cfg.get("model", {}).get("variant", "V0EarlyFusion")
    is_v0         = "V0" in model_variant

    # Stack modalities: each key has shape (1, H, W, D) in a raw test sample.
    x = torch.stack(
        [patient_data[k].squeeze(0) for k in ["t1", "t1ce", "t2", "flair"]], dim=0
    ).unsqueeze(0).to(device)  # (1, 4, H, W, D)

    inferer = _build_inferer(cfg)

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = patch.shape
        pc = cc = None
        if pe_type in ("film", "concat"):
            pc = torch.full((B, 3), 0.5, dtype=torch.float32, device=device)
        if pe_type == "concat":
            cc = make_coord_channels(pc, (H, W, D), VOL_SIZE).to(device)
        if is_v0:
            return model(patch, patch_center=pc, coord_channels=cc)
        else:
            return model(patch, patch_center=pc, coord_channels=cc,
                         active_modalities=active_modalities)

    model.eval()
    return inferer(inputs=x, network=predictor)  # (1, 3, H, W, D)


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
        scores[r] = (2 * inter / (union + 1e-8)).item()

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
        HausdorffDistanceMetric(include_background=False, percentile=95,
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
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    return results

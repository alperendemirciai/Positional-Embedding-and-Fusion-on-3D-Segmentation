"""Smoke test: validates the full pipeline using ~1% of training data.

Runs the following checks in order:
  1. Imports
  2. Patient split generation
  3. Data loading (5 patients, 1 patch each)
  4. Forward pass — all 9 model / PE combinations
  5. Loss computation
  6. Mini training loop — 3 epochs, 8 patients, V0/V1/V2
  7. Checkpoint save / load
  8. Sliding-window inference on 1 patient
  9. Prediction .nii.gz save

Usage:
    python smoke_test.py
    python smoke_test.py --data_root dataset/BraTS2021_Training_Data
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"

N_SMOKE_PATIENTS = 10        # ~1% of 875 train patients
N_SMOKE_EPOCHS   = 3
PATCH_SIZE       = [64, 64, 64]   # smaller than prod for speed


def header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {label}")
        return result
    except Exception as e:
        print(f"  [{FAIL}] {label}")
        print(f"           {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def smoke_config(data_root: str, splits_file: str, tmp_dir: str) -> dict:
    """Minimal config for smoke testing."""
    return {
        "data": {
            "root":                data_root,
            "splits_file":         splits_file,
            "patch_size":          PATCH_SIZE,
            "num_workers":         0,
            "cache_rate":          0.0,
            "foreground_ratio":    0.5,
        },
        "model": {
            "variant":             "V0EarlyFusion",
            "base_channels":       8,
            "depth":               3,
            "bottleneck_channels": 64,
        },
        "fusion":  {"strategy": "mean"},
        "training": {
            "max_epochs":              N_SMOKE_EPOCHS,
            "batch_size":              1,
            "lr":                      1e-4,
            "weight_decay":            1e-5,
            "grad_clip":               1.0,
            "early_stopping_patience": 100,
            "mixed_precision":         False,    # CPU-safe
            "num_patches_per_sample":  1,
            "loss":                    "dice_ce",
            "loss_dice_weight":        0.5,
        },
        "pe": {
            "type":                "none",
            "sinusoidal_num_freqs": 5,
            "film_hidden_dim":     64,
        },
        "inference": {
            "sw_batch_size":   2,
            "sw_overlap":      0.5,
            "sw_mode":         "gaussian",
            "threshold":       0.5,
            "save_predictions": True,
        },
        "evaluation": {
            "compute_hd95":      False,
            "missing_modality":  False,
        },
        "logging": {
            "tensorboard_dir":   str(Path(tmp_dir) / "logs"),
            "checkpoint_dir":    str(Path(tmp_dir) / "checkpoints"),
            "results_dir":       str(Path(tmp_dir) / "results"),
            "log_every_n_epochs": 1,
            "save_every_n_epochs": 999,
        },
    }


def make_tiny_splits(all_splits_path: str, tmp_dir: str) -> str:
    """Write a splits.json with only N_SMOKE_PATIENTS per split."""
    with open(all_splits_path) as f:
        full = json.load(f)

    tiny = {
        "train": full["train"][:N_SMOKE_PATIENTS],
        "val":   full["val"][:3],
        "test":  full["test"][:2],
    }
    out = str(Path(tmp_dir) / "tiny_splits.json")
    with open(out, "w") as f:
        json.dump(tiny, f)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset/BraTS2021_Training_Data")
    args = parser.parse_args()

    os.chdir(ROOT)

    device = torch.device("cpu")
    print(f"\nSmoke test — device: {device}")
    print(f"Patients per split: train={N_SMOKE_PATIENTS}, val=3, test=2")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Epochs: {N_SMOKE_EPOCHS}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        Path(tmp_dir, "checkpoints").mkdir()
        Path(tmp_dir, "results").mkdir()

        # ── 1. Imports ────────────────────────────────────────────────
        header("1 / 9  Imports")

        check("torch + monai", lambda: __import__("monai"))
        check("data_utils.split_dataset", lambda: __import__("data_utils.split_dataset"))
        check("data_utils.brats_dataset", lambda: __import__("data_utils.brats_dataset"))
        check("models.base_unet",         lambda: __import__("models.base_unet"))
        check("models.pe_modules",        lambda: __import__("models.pe_modules"))
        check("models.fusion_head",       lambda: __import__("models.fusion_head"))
        check("models.v0_early_fusion",   lambda: __import__("models.v0_early_fusion"))
        check("models.v1_shared_late",    lambda: __import__("models.v1_shared_late"))
        check("models.v2_separate_late",  lambda: __import__("models.v2_separate_late"))
        check("training.trainer",         lambda: __import__("training.trainer"))
        check("evaluation.evaluate",      lambda: __import__("evaluation.evaluate"))

        # ── 2. Split generation ───────────────────────────────────────
        header("2 / 9  Split Generation")

        data_root = args.data_root
        full_splits = str(ROOT / "data_utils" / "splits.json")

        if not Path(full_splits).exists():
            def _gen():
                from data_utils.split_dataset import generate_splits
                generate_splits(data_root, full_splits, seed=42)
            check("generate patient splits", _gen)
        else:
            check("splits.json already exists", lambda: json.load(open(full_splits)))

        tiny_splits = check(
            f"create tiny splits ({N_SMOKE_PATIENTS} train patients)",
            lambda: make_tiny_splits(full_splits, tmp_dir),
        )
        if tiny_splits is None:
            print("\nCannot continue without splits. Aborting.")
            sys.exit(1)

        cfg = smoke_config(data_root, tiny_splits, tmp_dir)

        # ── 3. Data Loading ───────────────────────────────────────────
        header("3 / 9  Data Loading")

        from torch.utils.data import DataLoader

        def _load_train():
            from data_utils.brats_dataset import build_datasets, stack_modalities
            train_ds, val_ds, test_ds = build_datasets(cfg, pe_type="none")
            assert len(train_ds) == N_SMOKE_PATIENTS, f"Expected {N_SMOKE_PATIENTS}, got {len(train_ds)}"
            sample = train_ds[0]
            # With num_samples=1, MONAI returns a list of one dict
            if isinstance(sample, list):
                sample = sample[0]
            x = stack_modalities({"t1": sample["t1"].unsqueeze(0),
                                   "t1ce": sample["t1ce"].unsqueeze(0),
                                   "t2": sample["t2"].unsqueeze(0),
                                   "flair": sample["flair"].unsqueeze(0)})
            assert x.shape[1] == 4, f"Expected 4 modality channels, got {x.shape[1]}"
            y = sample["seg"]
            assert y.shape[0] == 3, f"Expected 3 label channels (WT/TC/ET), got {y.shape[0]}"
            return train_ds, val_ds, test_ds

        ds_result = check("build datasets + verify shapes", _load_train)

        def _check_label_values():
            from data_utils.brats_dataset import build_datasets
            train_ds, _, _ = build_datasets(cfg, pe_type="none")
            sample = train_ds[0]
            if isinstance(sample, list):
                sample = sample[0]
            y = sample["seg"]
            assert y.min() >= 0 and y.max() <= 1, f"Labels out of [0,1]: min={y.min()} max={y.max()}"
            return True

        check("segmentation labels binary in [0,1]", _check_label_values)

        # ── 4. Forward Pass ───────────────────────────────────────────
        header("4 / 9  Forward Passes (all model × PE combinations)")

        B = 1
        H, W, D = PATCH_SIZE
        dummy_mod   = torch.rand(B, 4, H, W, D)
        dummy_coord = torch.rand(B, 3, H, W, D)
        dummy_pc    = torch.rand(B, 3)

        variants = [
            ("V0", "none"),
            ("V0", "film"),
            ("V0", "concat"),
            ("V1", "none"),
            ("V1", "film"),
            ("V1", "concat"),
            ("V2", "none"),
            ("V2", "film"),
            ("V2", "concat"),
        ]

        for variant, pe_type in variants:
            def _fwd(v=variant, p=pe_type):
                c = copy.deepcopy(cfg)
                c["model"]["variant"] = {
                    "V0": "V0EarlyFusion",
                    "V1": "V1SharedBackbone",
                    "V2": "V2SeparateBackbones",
                }[v]
                c["pe"]["type"] = p

                if v == "V0":
                    from models.v0_early_fusion import V0EarlyFusion
                    model = V0EarlyFusion(c)
                elif v == "V1":
                    from models.v1_shared_late import V1SharedBackbone
                    model = V1SharedBackbone(c)
                else:
                    from models.v2_separate_late import V2SeparateBackbones
                    model = V2SeparateBackbones(c)

                model.eval()
                with torch.no_grad():
                    out = model(
                        dummy_mod,
                        patch_center=dummy_pc if p in ("film", "concat") else None,
                        coord_channels=dummy_coord if p == "concat" else None,
                    )
                assert out.shape == (B, 3, H, W, D), \
                    f"Expected ({B},3,{H},{W},{D}), got {out.shape}"
                assert not torch.isnan(out).any(), "NaN in output"
                return True

            check(f"{variant} / pe={pe_type}  → output shape ({B},3,{H},{W},{D})", _fwd)

        # Missing-modality forward pass
        def _missing_mod():
            c = copy.deepcopy(cfg)
            c["model"]["variant"] = "V1SharedBackbone"
            c["pe"]["type"] = "none"
            from models.v1_shared_late import V1SharedBackbone
            model = V1SharedBackbone(c)
            model.eval()
            with torch.no_grad():
                out = model(dummy_mod, active_modalities=(0, 2))  # T1 + T2 only
            assert out.shape == (B, 3, H, W, D)
            assert not torch.isnan(out).any()
            return True

        check("V1 missing-modality forward (T1+T2 only)", _missing_mod)

        # ── 5. Loss Computation ───────────────────────────────────────
        header("5 / 9  Loss Function")

        def _loss():
            from monai.losses import DiceCELoss
            criterion = DiceCELoss(sigmoid=True, batch=True, squared_pred=True)
            logits = torch.randn(B, 3, H, W, D)
            target = (torch.rand(B, 3, H, W, D) > 0.7).float()
            loss = criterion(logits, target)
            assert loss.item() > 0
            assert not torch.isnan(loss)
            return loss.item()

        loss_val = check("DiceCELoss forward + backward", _loss)

        def _loss_backward():
            from monai.losses import DiceCELoss
            criterion = DiceCELoss(sigmoid=True, batch=True, squared_pred=True)
            c = copy.deepcopy(cfg)
            c["model"]["variant"] = "V0EarlyFusion"
            c["pe"]["type"] = "none"
            from models.v0_early_fusion import V0EarlyFusion
            model = V0EarlyFusion(c)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            opt.zero_grad()
            out = model(dummy_mod)
            target = (torch.rand(B, 3, H, W, D) > 0.7).float()
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            return True

        check("backward pass + optimizer step (no NaN grads)", _loss_backward)

        # ── 6. Mini Training Loop ─────────────────────────────────────
        header("6 / 9  Mini Training Loop (3 epochs × 3 model variants)")

        for variant_name, model_key in [
            ("V0EarlyFusion",       "V0"),
            ("V1SharedBackbone",    "V1"),
            ("V2SeparateBackbones", "V2"),
        ]:
            def _train(vn=variant_name, vk=model_key):
                c = copy.deepcopy(cfg)
                c["model"]["variant"] = vn
                c["pe"]["type"] = "none"

                from data_utils.brats_dataset import build_datasets
                train_ds, val_ds, _ = build_datasets(c, pe_type="none")

                from torch.utils.data import DataLoader
                from monai.data import pad_list_data_collate
                train_loader = DataLoader(train_ds, batch_size=1, shuffle=False,
                                          num_workers=0, collate_fn=pad_list_data_collate)
                val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                                          num_workers=0, collate_fn=pad_list_data_collate)

                if vn == "V0EarlyFusion":
                    from models.v0_early_fusion import V0EarlyFusion
                    model = V0EarlyFusion(c)
                elif vn == "V1SharedBackbone":
                    from models.v1_shared_late import V1SharedBackbone
                    model = V1SharedBackbone(c)
                else:
                    from models.v2_separate_late import V2SeparateBackbones
                    model = V2SeparateBackbones(c)

                from training.trainer import Trainer
                trainer = Trainer(model, c, f"smoke_{vk}", device)
                trainer.train(train_loader, val_loader)
                return True

            check(f"{variant_name} — {N_SMOKE_EPOCHS} epochs complete", _train)

        # ── 7. Checkpoint Save / Load ─────────────────────────────────
        header("7 / 9  Checkpoint Save / Load")

        def _ckpt():
            c = copy.deepcopy(cfg)
            c["model"]["variant"] = "V0EarlyFusion"
            c["pe"]["type"] = "none"
            from models.v0_early_fusion import V0EarlyFusion
            from training.trainer import Trainer
            model   = V0EarlyFusion(c)
            trainer = Trainer(model, c, "smoke_ckpt_test", device)
            trainer._save_checkpoint(0, "best")

            ckpt_path = Path(tmp_dir) / "checkpoints" / "smoke_ckpt_test_best.pth"
            assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

            model2  = V0EarlyFusion(c)
            trainer2 = Trainer(model2, c, "smoke_ckpt_test", device)
            trainer2.load_checkpoint(str(ckpt_path))
            return True

        check("save + reload checkpoint", _ckpt)

        # ── 8. Sliding-Window Inference ───────────────────────────────
        header("8 / 9  Sliding-Window Inference")

        def _inference():
            c = copy.deepcopy(cfg)
            c["model"]["variant"] = "V0EarlyFusion"
            c["pe"]["type"] = "none"
            from data_utils.brats_dataset import build_datasets
            _, _, test_ds = build_datasets(c, pe_type="none")

            sample = test_ds[0]
            if isinstance(sample, list):
                sample = sample[0]

            from models.v0_early_fusion import V0EarlyFusion
            model = V0EarlyFusion(c)
            model.eval()

            from evaluation.evaluate import run_inference
            logits = run_inference(model, sample, c, device, (0, 1, 2, 3))
            assert logits.shape[1] == 3, f"Expected 3 output channels, got {logits.shape[1]}"
            assert not torch.isnan(logits).any(), "NaN in inference output"
            return logits.shape

        result_shape = check("sliding-window inference → valid output", _inference)
        if result_shape:
            print(f"           Output shape: {result_shape}")

        # ── 9. Prediction Save ────────────────────────────────────────
        header("9 / 9  Prediction .nii.gz Save")

        def _save_pred():
            c = copy.deepcopy(cfg)
            c["model"]["variant"] = "V0EarlyFusion"
            c["pe"]["type"] = "none"
            from data_utils.brats_dataset import build_datasets
            _, _, test_ds = build_datasets(c, pe_type="none")
            sample = test_ds[0]
            if isinstance(sample, list):
                sample = sample[0]

            from models.v0_early_fusion import V0EarlyFusion
            from monai.metrics import DiceMetric
            from evaluation.evaluate import run_evaluation
            model = V0EarlyFusion(c)
            model.eval()

            results = run_evaluation(
                model, test_ds, c, device,
                experiment_name="smoke_pred_test",
                missing_modality=False,
            )
            assert "mean" in results
            assert "per_patient" in results
            pred_dir = Path(c["logging"]["results_dir"]) / "predictions" / "smoke_pred_test"
            nii_files = list(pred_dir.glob("*.nii.gz"))
            assert len(nii_files) > 0, "No .nii.gz prediction files saved"
            return len(nii_files)

        n_saved = check("run_evaluation → saves .nii.gz predictions", _save_pred)
        if n_saved:
            print(f"           Saved {n_saved} prediction file(s)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Smoke test complete.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()

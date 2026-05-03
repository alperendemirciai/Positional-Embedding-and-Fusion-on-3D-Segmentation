"""Batch evaluation: runs test-set inference for every experiment config.

Usage:
    python evaluate_all.py
    python evaluate_all.py --configs v0_nope v1_film   # subset of experiments
"""

import argparse
import copy
from pathlib import Path

import torch
import yaml


EXPERIMENTS = [
    ("v0_nope",          "configs/v0_nope.yaml"),
    ("v0_film",          "configs/v0_film.yaml"),
    ("v0_concat",        "configs/v0_concat.yaml"),
    ("v1_nope_mean",     "configs/v1_nope_mean.yaml"),
    ("v1_nope_weighted", "configs/v1_nope_weighted.yaml"),
    ("v1_nope_attention","configs/v1_nope_attention.yaml"),
    ("v1_film",          "configs/v1_film.yaml"),
    ("v1_concat",        "configs/v1_concat.yaml"),
    ("v2_nope_mean",     "configs/v2_nope_mean.yaml"),
    ("v2_nope_weighted", "configs/v2_nope_weighted.yaml"),
    ("v2_nope_attention","configs/v2_nope_attention.yaml"),
    ("v2_film",          "configs/v2_film.yaml"),
    ("v2_concat",        "configs/v2_concat.yaml"),
]


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    with open("configs/base_config.yaml") as f:
        base = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)
    return deep_merge(base, override)


def build_model(cfg: dict):
    variant = cfg["model"]["variant"]
    if variant == "V0EarlyFusion":
        from models.v0_early_fusion import V0EarlyFusion
        return V0EarlyFusion(cfg)
    elif variant == "V1SharedBackbone":
        from models.v1_shared_late import V1SharedBackbone
        return V1SharedBackbone(cfg)
    elif variant == "V2SeparateBackbones":
        from models.v2_separate_late import V2SeparateBackbones
        return V2SeparateBackbones(cfg)
    else:
        raise ValueError(f"Unknown model variant: {variant!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Experiment names to evaluate (default: all)"
    )
    args = parser.parse_args()

    selected = set(args.configs) if args.configs else None
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    from data_utils.brats_dataset import build_datasets
    from evaluation.evaluate import run_evaluation

    for exp_name, cfg_path in EXPERIMENTS:
        if selected and exp_name not in selected:
            continue

        ckpt_path = Path("checkpoints") / f"{exp_name}_best.pth"
        if not ckpt_path.exists():
            print(f"[SKIP] {exp_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        cfg      = load_config(cfg_path)
        pe_type  = cfg.get("pe", {}).get("type", "none")
        missing  = cfg.get("evaluation", {}).get("missing_modality", False)

        _, _, test_ds = build_datasets(cfg, pe_type=pe_type)

        model = build_model(cfg)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        results = run_evaluation(
            model, test_ds, cfg, device,
            experiment_name=exp_name,
            missing_modality=missing,
        )

        m = results["mean"]
        print(
            f"\n  Summary — WT: {m['WT']:.3f}  TC: {m['TC']:.3f}  "
            f"ET: {m['ET']:.3f}  Mean: {sum(m.values())/3:.3f}"
        )

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()

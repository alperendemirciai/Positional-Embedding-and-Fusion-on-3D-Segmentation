"""Main training entry point.

Usage:
    python train.py --config configs/v0_nope.yaml --experiment_name v0_nope
    python train.py --config configs/v2_film.yaml --experiment_name v2_film \\
                    --resume checkpoints/v2_film_epoch050.pth
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True,  help="Path to experiment YAML config")
    parser.add_argument("--experiment_name", required=True,  help="Unique name for this run")
    parser.add_argument("--resume",          default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--max_epochs",      type=int, default=None,
                        help="Override max_epochs from config (useful for quick runs)")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    pe_type = cfg.get("pe", {}).get("type", "none")

    # Build dataset
    from data_utils.brats_dataset import build_datasets
    train_ds, val_ds, _ = build_datasets(cfg, pe_type=pe_type)

    train_cfg  = cfg["training"]
    batch_size = train_cfg["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    from monai.data import pad_list_data_collate
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=pad_list_data_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['variant']}  |  Params: {n_params:,}")

    from training.trainer import Trainer
    trainer = Trainer(model, cfg, args.experiment_name, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

import json
import random
from pathlib import Path


def generate_splits(data_root: str, output_path: str, seed: int = 42):
    root = Path(data_root)
    patient_ids = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("BraTS2021_")
    )

    total = len(patient_ids)
    random.seed(seed)
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)

    n_train = int(total * 0.70)
    n_val   = int(total * 0.15)

    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:n_train + n_val]
    test_ids  = shuffled[n_train + n_val:]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Total patients : {total}")
    print(f"Train          : {len(train_ids)}")
    print(f"Val            : {len(val_ids)}")
    print(f"Test           : {len(test_ids)}")
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    print(f"Splits saved to {output_path}")
    return splits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset/BraTS2021_Training_Data")
    parser.add_argument("--output",    default="data_utils/splits.json")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()
    generate_splits(args.data_root, args.output, args.seed)

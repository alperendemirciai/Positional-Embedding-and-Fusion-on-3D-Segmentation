"""Aggregate test results across all experiments and produce summary tables and plots.

Usage:
    python results_analysis.py
    python results_analysis.py --results_dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evaluate import ALL_MODALITY_SUBSETS, SUBSET_KEYS

EXPERIMENTS = [
    "v0_nope", "v0_film", "v0_concat",
    "v1_nope_mean", "v1_nope_weighted", "v1_nope_attention",
    "v1_film", "v1_concat",
    "v2_nope_mean", "v2_nope_weighted", "v2_nope_attention",
    "v2_film", "v2_concat",
]
REGIONS = ["WT", "TC", "ET"]


def load_results(results_dir: Path) -> dict:
    data = {}
    for exp in EXPERIMENTS:
        path = results_dir / f"{exp}_test_results.json"
        if path.exists():
            with open(path) as f:
                data[exp] = json.load(f)
        else:
            print(f"[WARN] Missing results for {exp}")
    return data


def build_summary_table(data: dict) -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        if exp not in data:
            continue
        m   = data[exp]["mean"]
        std = data[exp]["std"]
        mean_val = sum(m[r] for r in REGIONS) / len(REGIONS)
        mean_std = sum(std[r] for r in REGIONS) / len(REGIONS)
        row = {"Experiment": exp}
        for r in REGIONS:
            row[r]              = f"{m[r]:.3f} ± {std[r]:.3f}"
            row[f"{r}_val"]     = m[r]
        row["Mean"]      = f"{mean_val:.3f} ± {mean_std:.3f}"
        row["Mean_val"]  = mean_val
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame):
    display_cols = ["Experiment"] + REGIONS + ["Mean"]
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean ± std Dice on test set)")
    print("=" * 70)
    print(df[display_cols].to_string(index=False))
    print("=" * 70)


def save_markdown_table(df: pd.DataFrame, out_path: Path):
    display_cols = ["Experiment"] + REGIONS + ["Mean"]
    md = df[display_cols].to_markdown(index=False)
    out_path.write_text(md)
    print(f"Markdown table saved to {out_path}")


def plot_missing_modality(data: dict, out_dir: Path):
    """Plot per-region Dice vs number of available modalities for V1 and V2."""
    out_dir.mkdir(parents=True, exist_ok=True)

    late_fusion_exps = [e for e in EXPERIMENTS if e.startswith(("v1", "v2"))]

    for exp in late_fusion_exps:
        if exp not in data:
            continue
        mm = data[exp].get("missing_modality")
        if not mm:
            continue

        # Group by subset size
        size_to_scores = {s: {r: [] for r in REGIONS} for s in range(1, 5)}
        for subset_tuple in ALL_MODALITY_SUBSETS:
            key = SUBSET_KEYS[subset_tuple]
            size = len(subset_tuple)
            if key in mm:
                for r in REGIONS:
                    size_to_scores[size][r].append(mm[key][r])
        # Add full-modality (size=4)
        m_full = data[exp]["mean"]
        for r in REGIONS:
            size_to_scores[4][r] = [m_full[r]]

        fig, ax = plt.subplots(figsize=(7, 4))
        sizes = sorted(size_to_scores.keys())
        colors = {"WT": "#2196F3", "TC": "#FF5722", "ET": "#4CAF50"}
        for r in REGIONS:
            means = [np.mean(size_to_scores[s][r]) for s in sizes]
            stds  = [np.std(size_to_scores[s][r])  for s in sizes]
            ax.errorbar(sizes, means, yerr=stds, marker="o", label=r,
                        color=colors[r], capsize=4)

        ax.set_xlabel("Number of available modalities")
        ax.set_ylabel("Dice score")
        ax.set_title(f"Missing-modality robustness — {exp}")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / f"{exp}_missing_modality.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {fig_path}")


def answer_research_questions(data: dict):
    print("\n" + "=" * 70)
    print("RESEARCH QUESTIONS")
    print("=" * 70)

    def mean_dice(exp):
        if exp not in data:
            return None
        m = data[exp]["mean"]
        return sum(m[r] for r in REGIONS) / len(REGIONS)

    # Q1: Does late fusion generalise better under missing modalities?
    v0 = mean_dice("v0_nope")
    v1 = mean_dice("v1_nope_mean")
    v2 = mean_dice("v2_nope_mean")
    print(f"\nQ1 — Full-modality mean Dice:")
    if v0 is not None: print(f"  V0 (early): {v0:.3f}")
    if v1 is not None: print(f"  V1 (shared late): {v1:.3f}")
    if v2 is not None: print(f"  V2 (separate late): {v2:.3f}")

    # Q2: Shared vs separate backbones
    print(f"\nQ2 — Shared (V1) vs Separate (V2) backbone (no PE, mean fusion):")
    if v1 is not None and v2 is not None:
        winner = "V1 (shared)" if v1 > v2 else "V2 (separate)"
        print(f"  V1: {v1:.3f}   V2: {v2:.3f}   Winner: {winner}")

    # Q3: Does PE help?
    print(f"\nQ3 — Effect of positional encoding:")
    for base in ["v0_nope", "v1_nope_mean", "v2_nope_mean"]:
        variant = base.replace("_nope", "").replace("_mean", "")
        film_key = f"{variant}_film"
        concat_key = f"{variant}_concat"
        base_val   = mean_dice(base)
        film_val   = mean_dice(film_key)
        concat_val = mean_dice(concat_key)
        if base_val is None:
            continue
        film_str   = f"{film_val:.3f}"   if film_val   is not None else "n/a"
        concat_str = f"{concat_val:.3f}" if concat_val is not None else "n/a"
        print(f"  {base:<25s}: {base_val:.3f} (no PE) | "
              f"{film_str} (FiLM) | {concat_str} (concat)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = load_results(results_dir)

    if not data:
        print("No result files found. Run evaluate_all.py first.")
        return

    df = build_summary_table(data)
    print_comparison_table(df)

    csv_path = results_dir / "summary.csv"
    display_cols = ["Experiment"] + REGIONS + ["Mean"]
    df[display_cols].to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    save_markdown_table(df, results_dir / "summary.md")
    plot_missing_modality(data, results_dir / "plots")
    answer_research_questions(data)


if __name__ == "__main__":
    main()

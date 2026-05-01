# Multi-Modal Fusion Strategies and Positional Encoding for 3D Brain Tumor Segmentation

A comparative study on BraTS 2021 — AIN412 Final Project by Alperen Demirci.

---

## Overview

This project investigates two orthogonal design axes for 3D brain tumor segmentation on multi-parametric MRI:

1. **Fusion strategy** — how to combine four MRI modalities (T1, T1ce, T2, FLAIR)
2. **Positional encoding** — whether injecting spatial context about patch location improves segmentation

The result is a controlled empirical comparison across **13 configurations** that cleanly isolates the effect of each design decision.

---

## Research Questions

| # | Question |
|---|----------|
| Q1 | Does late fusion generalise better than early fusion under missing-modality conditions? |
| Q2 | Does a shared backbone match or surpass fully separate per-modality backbones? |
| Q3 | Does encoding the spatial location of the sliding-window patch improve segmentation quality? |

---

## Model Variants

### Fusion Strategies

| Variant | Description |
|---------|-------------|
| **V0 — Early Fusion** | Four modality volumes concatenated along the channel axis; fed as a single 4-channel input to one 3D U-Net. Standard BraTS baseline. |
| **V1 — Shared Backbone Late Fusion** | A single 3D U-Net applied independently to each modality (one channel at a time). Four output logit maps are combined by the fusion head. |
| **V2 — Separate Backbones Late Fusion** | Four independent 3D U-Nets, one per modality. Each becomes a modality specialist. Same fusion head combines their outputs. |

### Positional Encoding Types

| Type | Description |
|------|-------------|
| `none` | No positional information — baseline |
| `film` | **Fourier features + FiLM**: normalised patch centre (x,y,z) → sinusoidal encoding (30-dim) → MLP → γ,β → scale+shift bottleneck features |
| `concat` | **Coordinate channels**: three normalised coordinate grids appended as extra input channels |

### Late Fusion Head Options

| Strategy | Description |
|----------|-------------|
| `mean` | Simple equal-weight average of per-modality logit maps |
| `weighted` | Four learnable scalar weights (softmax-normalised) per modality |
| `attention` | Voxel-wise 1×1×1 conv attention over concatenated logit maps — spatially adaptive |

This gives **13 total experiment configurations**.

---

## Dataset

BraTS 2021 Task 1 — 1251 patient cases, each with:
- 4 co-registered 3D MRI volumes: **T1**, **T1ce**, **T2**, **T2-FLAIR** (240×240×155, 1mm³, skull-stripped)
- Manual segmentation mask with three tumour sub-regions:
  - **NCR** — Necrotic Tumour Core (label 1)
  - **ED** — Peritumoral Edema (label 2)
  - **ET** — Enhancing Tumour (label 4)

Evaluation is reported on three derived regions:

| Region | Definition |
|--------|------------|
| **WT** — Whole Tumour | NCR + ED + ET |
| **TC** — Tumour Core | NCR + ET |
| **ET** — Enhancing Tumour | ET only |

**Patient-wise split (seed=42, no data leakage):**

| Split | Patients |
|-------|----------|
| Train | 875 (70%) |
| Val   | 188 (15%) |
| Test  | 188 (15%) |

---

## Architecture

### 3D U-Net Backbone

```
Input (B, C_in, 96, 96, 96)
  │
  ├─ Enc1: ConvBlock(C_in → 16)            [96³]
  ├─ Pool + Enc2: ConvBlock(16 → 32)       [48³]
  ├─ Pool + Enc3: ConvBlock(32 → 64)       [24³]
  ├─ Pool + Enc4: ConvBlock(64 → 128)      [12³]
  └─ Pool + Bottleneck: ConvBlock(128→256) [6³]
              │
        [FiLM here if pe_type='film']
              │
  ├─ Dec4: TransConv + skip + ConvBlock(256→128)  [12³]
  ├─ Dec3: TransConv + skip + ConvBlock(128→64)   [24³]
  ├─ Dec2: TransConv + skip + ConvBlock(64→32)    [48³]
  └─ Dec1: TransConv + skip + ConvBlock(32→16)    [96³]
              │
        Conv1×1×1 → 3 logits (WT, TC, ET)
```

Each ConvBlock is: `[Conv3d → InstanceNorm3d → LeakyReLU(0.01)] × 2`

**Parameter count (base_channels=16, depth=4):**
- V0 / V1 single network: ~31M parameters
- V2 total (4 networks): ~124M parameters

### FiLM Positional Encoding

```
patch_centre (B,3) ∈ [0,1]³
      │
SinusoidalPE   [π·2⁰ ... π·2⁴, sin+cos per coord]
      │
  (B, 30)
      │
MLP: Linear(30→128) → ReLU → Linear(128→512)
      │
  (B, 512)  →  split  →  γ (B,256),  β (B,256)
      │
bottleneck_features = γ * features + β
```

---

## Project Structure

```
.
├── requirements.txt
├── train.py                    # training entry point
├── evaluate_all.py             # batch test-set evaluation
├── results_analysis.py         # tables, plots, research Q answers
│
├── data_utils/
│   ├── split_dataset.py        # generate patient-wise splits.json
│   └── brats_dataset.py        # MONAI dataset, transforms, patch sampling
│
├── models/
│   ├── base_unet.py            # custom 3D U-Net with FiLM support
│   ├── pe_modules.py           # SinusoidalPE, FiLMConditioner, coord channels
│   ├── fusion_head.py          # MeanFusion, WeightedFusion, AttentionFusion
│   ├── v0_early_fusion.py      # V0 model
│   ├── v1_shared_late.py       # V1 model
│   └── v2_separate_late.py     # V2 model
│
├── training/
│   └── trainer.py              # training loop, checkpointing, TensorBoard
│
├── evaluation/
│   └── evaluate.py             # sliding-window inference, saves .nii.gz
│
├── configs/
│   ├── base_config.yaml        # all default hyperparameters
│   ├── v0_nope.yaml            # ┐
│   ├── v0_film.yaml            # │ V0 variants
│   ├── v0_concat.yaml          # ┘
│   ├── v1_nope_mean.yaml       # ┐
│   ├── v1_nope_weighted.yaml   # │
│   ├── v1_nope_attention.yaml  # │ V1 variants
│   ├── v1_film.yaml            # │
│   ├── v1_concat.yaml          # ┘
│   ├── v2_nope_mean.yaml       # ┐
│   ├── v2_nope_weighted.yaml   # │
│   ├── v2_nope_attention.yaml  # │ V2 variants
│   ├── v2_film.yaml            # │
│   └── v2_concat.yaml          # ┘
│
├── experiments/
│   ├── run_all.sh              # one-shot: train → evaluate → analyse
│   └── logs/                   # TensorBoard logs per experiment
│
├── checkpoints/                # saved model weights
│
└── results/
    ├── predictions/            # per-patient .nii.gz segmentation outputs
    ├── summary.csv
    ├── summary.md
    └── plots/                  # missing-modality degradation curves
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate patient splits
python data_utils/split_dataset.py

# 3. Train one experiment
python train.py --config configs/v0_nope.yaml --experiment_name v0_nope

# 4. Run all 13 experiments (long — see USAGE.md for partial runs)
bash experiments/run_all.sh

# 5. Evaluate all trained models
python evaluate_all.py

# 6. Generate summary tables and plots
python results_analysis.py
```

See [USAGE.md](USAGE.md) for detailed instructions on every script and all configuration options.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR schedule | CosineAnnealingLR (T_max=300, η_min=1e-6) |
| Loss | DiceCE (0.5 × Dice + 0.5 × BCE) |
| Max epochs | 300 |
| Early stopping | patience=50 on val mean Dice |
| Patch size | 96 × 96 × 96 |
| Batch size | 2 (V0/V1), 1 (V2) |
| Mixed precision | FP16 (torch.cuda.amp) |
| Gradient clipping | max_norm=1.0 |
| Augmentation | random flips, 90° rotations, intensity scale/shift, Gaussian noise |

**Estimated training time (single ≤16GB GPU):**
- V0 / V1 experiment: ~6 hours
- V2 experiment: ~22 hours
- All 13 experiments: ~158 hours (~6.5 days)

---

## Evaluation

- **Primary metric**: Dice coefficient per region (WT, TC, ET)
- **Secondary metric**: 95th percentile Hausdorff distance (HD95)
- **Inference**: MONAI `SlidingWindowInferer`, 96³ patches, 50% overlap, Gaussian weighting
- **Predictions**: saved as `.nii.gz` in `results/predictions/` — compatible with ITK-SNAP and 3D Slicer
- **Missing-modality evaluation**: V1 and V2 models are additionally tested on all 15 non-empty subsets of the 4 modalities

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch ≥ 2.1 | Deep learning framework |
| MONAI ≥ 1.3 | Medical imaging transforms, sliding window, Dice loss, metrics |
| nibabel | NIfTI file I/O (via MONAI) |
| pandas | Results table construction |
| matplotlib | Missing-modality plots |
| TensorBoard | Training curve visualisation |
| PyYAML | Configuration loading |

---

## Citation

If you use this code or the BraTS 2021 dataset, please cite:

```
Baid, U. et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
Segmentation and Radiogenomic Classification. arXiv:2107.02314.

Menze, B.H. et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark
(BRATS). IEEE Transactions on Medical Imaging, 34(10), 1993-2024.

Bakas, S. et al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections
with expert segmentation labels and radiomic features. Nature Scientific Data, 4:170117.
```

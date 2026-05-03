#!/bin/bash
# Train and evaluate 5 no-PE baseline models:
#   single_t1     — UNet trained on T1 only
#   single_t1ce   — UNet trained on T1ce only
#   single_t2     — UNet trained on T2 only
#   single_flair  — UNet trained on FLAIR only
#   v0_nope       — UNet trained on all 4 modalities concatenated (early fusion)
#
# Usage:
#   bash experiments/run_single_modality.sh
#   bash experiments/run_single_modality.sh single_t1 v0_nope   # specific subset

set -euo pipefail   # strict mode for setup; relaxed per-experiment below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL_EXPERIMENTS=(
    single_t1
    single_t1ce
    single_t2
    single_flair
    v0_nope
)

if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

echo "================================================================"
echo "Single-Modality + V0 Baseline Training"
echo "Experiments : ${EXPERIMENTS[*]}"
echo "================================================================"

# ── Patient splits ────────────────────────────────────────────────────────────
if [ ! -f "data_utils/splits.json" ]; then
    echo ""
    echo ">>> Generating patient splits ..."
    python data_utils/split_dataset.py \
        --data_root dataset/BraTS2021_Training_Data \
        --output    data_utils/splits.json \
        --seed      42
fi

mkdir -p checkpoints experiments/logs results

# ── Training ──────────────────────────────────────────────────────────────────
FAILED=()
for EXP in "${EXPERIMENTS[@]}"; do
    CFG="configs/${EXP}.yaml"
    if [ ! -f "$CFG" ]; then
        echo "[SKIP] Config not found: $CFG"
        continue
    fi

    CKPT="checkpoints/${EXP}_best.pth"

    # Validate any existing checkpoint before trusting the skip
    if [ -f "$CKPT" ]; then
        if python -c "import torch; torch.load('$CKPT', map_location='cpu')" 2>/dev/null; then
            echo ""
            echo ">>> [SKIP] $EXP — valid checkpoint already exists at $CKPT"
            continue
        else
            echo ">>> [WARN] $EXP — checkpoint at $CKPT is corrupt, retraining"
            rm -f "$CKPT"
        fi
    fi

    echo ""
    echo "================================================================"
    echo ">>> Training: $EXP"
    echo "================================================================"
    set +e
    python train.py \
        --config          "$CFG" \
        --experiment_name "$EXP" \
        --seed            42
    STATUS=$?
    set -e

    if [ $STATUS -ne 0 ]; then
        echo "  [FAILED] $EXP exited with status $STATUS — continuing"
        FAILED+=("$EXP")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo ">>> WARNING: the following experiments failed: ${FAILED[*]}"
fi

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo ">>> Evaluating ..."
echo "================================================================"
python evaluate_all.py --configs "${EXPERIMENTS[@]}"

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo ">>> Generating summary ..."
echo "================================================================"
python results_analysis.py

echo ""
echo "================================================================"
echo "ALL DONE. Results in results/summary.md"
echo "================================================================"

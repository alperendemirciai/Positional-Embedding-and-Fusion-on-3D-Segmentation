#!/bin/bash
# Run all 13 experiments sequentially on a single GPU, then evaluate and analyse.
# Usage: bash experiments/run_all.sh
#        bash experiments/run_all.sh v0_nope v0_film   # run specific experiments only

set -euo pipefail   # strict mode for setup steps only; disabled per-experiment below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL_EXPERIMENTS=(
    v0_nope
    v0_film
    v0_concat
    v1_nope_mean
    v1_nope_weighted
    v1_nope_attention
    v1_film
    v1_concat
    v2_nope_mean
    v2_nope_weighted
    v2_nope_attention
    v2_film
    v2_concat
)

# If arguments supplied, run only those experiments
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

echo "================================================================"
echo "BraTS Fusion + PE Experiments"
echo "Experiments to run: ${EXPERIMENTS[*]}"
echo "================================================================"

# Generate splits if not already done
if [ ! -f "data_utils/splits.json" ]; then
    echo ""
    echo ">>> Generating patient splits ..."
    python data_utils/split_dataset.py \
        --data_root dataset/BraTS2021_Training_Data \
        --output    data_utils/splits.json \
        --seed      42
fi

# Training phase — failures are logged but do not abort remaining experiments
FAILED=()
for EXP in "${EXPERIMENTS[@]}"; do
    CFG="configs/${EXP}.yaml"
    if [ ! -f "$CFG" ]; then
        echo "[SKIP] Config not found: $CFG"
        continue
    fi

    CKPT="checkpoints/${EXP}_best.pth"

    # Validate checkpoint integrity before trusting the skip (Bug 20)
    if [ -f "$CKPT" ]; then
        if python -c "import torch; torch.load('$CKPT', map_location='cpu')" 2>/dev/null; then
            echo ""
            echo ">>> [SKIP] $EXP — valid checkpoint already exists at $CKPT"
            continue
        else
            echo ">>> [WARN] $EXP — corrupt checkpoint at $CKPT, removing and retraining"
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
    TRAIN_STATUS=$?
    set -e

    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "  [FAILED] $EXP exited with status $TRAIN_STATUS — continuing with remaining experiments"
        FAILED+=("$EXP")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo ">>> WARNING: the following experiments failed: ${FAILED[*]}"
fi

# Evaluation phase
echo ""
echo "================================================================"
echo ">>> Evaluating all experiments ..."
echo "================================================================"
python evaluate_all.py

# Results analysis
echo ""
echo "================================================================"
echo ">>> Generating summary tables and plots ..."
echo "================================================================"
python results_analysis.py

echo ""
echo "================================================================"
echo "ALL DONE. Results in results/summary.md and results/plots/"
echo "================================================================"

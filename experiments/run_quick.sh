#!/bin/bash
# Run all 13 experiments for a fixed number of epochs (default 20).
# Useful for sanity-checking that every config trains without errors.
#
# Usage:
#   bash experiments/run_quick.sh            # all 13 configs, 20 epochs each
#   bash experiments/run_quick.sh --epochs 50
#   bash experiments/run_quick.sh v0_nope v0_film   # subset of configs, 20 epochs
#   bash experiments/run_quick.sh --epochs 50 v0_nope v1_film

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Parse arguments --------------------------------------------------------
EPOCHS=20
EXPERIMENTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            EXPERIMENTS+=("$1")
            shift
            ;;
    esac
done

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

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

echo "================================================================"
echo "Quick run: ${#EXPERIMENTS[@]} experiments × ${EPOCHS} epochs"
echo "Experiments: ${EXPERIMENTS[*]}"
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

PASSED=()
FAILED=()

for EXP in "${EXPERIMENTS[@]}"; do
    CFG="configs/${EXP}.yaml"
    if [ ! -f "$CFG" ]; then
        echo "[SKIP] Config not found: $CFG"
        FAILED+=("$EXP (no config)")
        continue
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo ">>> $EXP  (${EPOCHS} epochs)"
    echo "----------------------------------------------------------------"

    if python train.py \
        --config          "$CFG" \
        --experiment_name "quick_${EXP}" \
        --max_epochs      "$EPOCHS" \
        --seed            42; then
        PASSED+=("$EXP")
    else
        echo "[FAILED] $EXP"
        FAILED+=("$EXP")
    fi
done

echo ""
echo "================================================================"
echo "Quick run complete"
echo "  PASSED (${#PASSED[@]}): ${PASSED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "Checkpoints saved as  checkpoints/quick_<name>_best.pth"
echo "================================================================"

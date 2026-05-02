#!/bin/bash
# Full pipeline: train every experiment then immediately evaluate it on the test set.
#
# Each experiment is trained to --epochs (default 300) with early stopping, then
# evaluated with sliding-window inference on the held-out test set.  Results are
# written to results/<exp>_test_results.json.  A final summary table is printed.
#
# Design decisions:
#   • train → evaluate back-to-back per experiment so a mid-run crash still leaves
#     complete results for every experiment that finished.
#   • Skips training if a checkpoint already exists (safe to re-run after a crash).
#   • Skips evaluation if a results JSON already exists.
#   • Never uses set -e on the outer loop; one failing experiment won't abort the rest.
#   • Logs each experiment's stdout+stderr to experiments/logs/<exp>.log.
#
# Usage:
#   bash experiments/run_training.sh                       # all 13, 300 epochs
#   bash experiments/run_training.sh --epochs 30           # all 13, 30 epochs
#   bash experiments/run_training.sh --epochs 30 v0_nope v1_film  # subset, 30 epochs

set -euo pipefail   # strict mode for setup; disabled per-experiment below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Parse arguments ──────────────────────────────────────────────────────────
EPOCHS=300
REQUESTED_EXPS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        *)        REQUESTED_EXPS+=("$1"); shift ;;
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

if [[ ${#REQUESTED_EXPS[@]} -gt 0 ]]; then
    EXPERIMENTS=("${REQUESTED_EXPS[@]}")
else
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

mkdir -p experiments/logs

echo "================================================================"
echo "BraTS Fusion+PE — full pipeline"
echo "Experiments : ${EXPERIMENTS[*]}"
echo "Epochs      : ${EPOCHS}"
echo "Start time  : $(date)"
echo "================================================================"

# ── Generate patient splits once ─────────────────────────────────────────────
if [[ ! -f "data_utils/splits.json" ]]; then
    echo ""
    echo ">>> Generating patient splits …"
    python data_utils/split_dataset.py \
        --data_root dataset/BraTS2021_Training_Data \
        --output    data_utils/splits.json \
        --seed      42
fi

# ── Per-experiment train → evaluate ──────────────────────────────────────────
PASSED=()
FAILED=()

for EXP in "${EXPERIMENTS[@]}"; do
    CFG="configs/${EXP}.yaml"
    CKPT="checkpoints/${EXP}_best.pth"
    RESULTS="results/${EXP}_test_results.json"
    LOG="experiments/logs/${EXP}.log"

    echo ""
    echo "================================================================"
    echo ">>> Experiment: ${EXP}  (max ${EPOCHS} epochs)"
    echo "================================================================"

    if [[ ! -f "$CFG" ]]; then
        echo "[SKIP] config not found: $CFG"
        FAILED+=("${EXP} (no config)")
        continue
    fi

    # ── Training ──────────────────────────────────────────────────────────────
    if [[ -f "$CKPT" ]]; then
        echo "  [SKIP TRAINING] checkpoint already exists: $CKPT"
    else
        echo "  Training … (log: $LOG)"
        set +e
        python train.py \
            --config          "$CFG"  \
            --experiment_name "$EXP"  \
            --max_epochs      "$EPOCHS" \
            --seed            42 \
            2>&1 | tee "$LOG"
        TRAIN_STATUS=${PIPESTATUS[0]}
        set -e

        if [[ $TRAIN_STATUS -ne 0 ]]; then
            echo "  [FAILED] training exited with status $TRAIN_STATUS"
            FAILED+=("${EXP} (training failed)")
            continue
        fi

        if [[ ! -f "$CKPT" ]]; then
            echo "  [FAILED] training finished but checkpoint not found: $CKPT"
            FAILED+=("${EXP} (no checkpoint after training)")
            continue
        fi
        echo "  Training complete. Checkpoint: $CKPT"
    fi

    # ── Evaluation ────────────────────────────────────────────────────────────
    if [[ -f "$RESULTS" ]]; then
        echo "  [SKIP EVALUATION] results already exist: $RESULTS"
        PASSED+=("$EXP")
        continue
    fi

    echo "  Evaluating on test set …"
    set +e
    python evaluate_all.py --configs "$EXP" 2>&1 | tee -a "$LOG"
    EVAL_STATUS=${PIPESTATUS[0]}
    set -e

    if [[ $EVAL_STATUS -ne 0 ]]; then
        echo "  [FAILED] evaluation exited with status $EVAL_STATUS"
        FAILED+=("${EXP} (evaluation failed)")
        continue
    fi

    PASSED+=("$EXP")
    echo "  Done. Results: $RESULTS"
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo ">>> Generating summary table and plots …"
echo "================================================================"
set +e
python results_analysis.py
set -e

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "PIPELINE COMPLETE  —  $(date)"
echo "  Passed (${#PASSED[@]}) : ${PASSED[*]:-none}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED (${#FAILED[@]}) : ${FAILED[*]}"
fi
echo "Results  : results/"
echo "Logs     : experiments/logs/"
echo "================================================================"

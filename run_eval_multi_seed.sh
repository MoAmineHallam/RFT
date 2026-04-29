#!/bin/bash
# run_eval_multi_seed.sh — RULER S-NIAH eval across all multi-seed retrains.
#
# Evaluates every (seed, variant) combination from run_multi_seed_v4b.sh
# and produces paper-ready results with Wilson confidence intervals.
#
# For NIAH-trained variants (rft_lm, rft_lm_disable_ocr), uses Phase 2 checkpoint.
# For non-NIAH variants (rft_lm_disable_memory, baseline), uses Phase 1 checkpoint.
#
# Output: one JSON per (seed, variant) in $OUTDIR, plus a summary CSV.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_multi_seed.sh
#   # Or with custom run root:
#   RUNROOT=/path/to/multi_seed_v4b_YYYYMMDD bash run_eval_multi_seed.sh

set -euo pipefail

export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

# Find most recent multi_seed run if not specified
RUNROOT="${RUNROOT:-$(ls -d "$ROOT"/multi_seed_v4b_* 2>/dev/null | sort | tail -1)}"
if [ -z "$RUNROOT" ] || [ ! -d "$RUNROOT" ]; then
    echo "ERROR: No multi_seed_v4b_* directory found under $ROOT"
    echo "  Set RUNROOT=/path/to/multi_seed_v4b_YYYYMMDD"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$RUNROOT/niah_eval"
mkdir -p "$OUTDIR"

SEEDS=(42 43 44)
# Variants that went through Phase 2 (NIAH curriculum)
NIAH_VARIANTS=(rft_lm rft_lm_disable_ocr)
# Variants that stopped after Phase 1
PHASE1_ONLY_VARIANTS=(rft_lm_disable_memory baseline)

SUMMARY="$OUTDIR/summary.csv"
echo "seed,variant,seq_len,depth,accuracy,ci95_low,ci95_high,n_trials" > "$SUMMARY"

echo "[EVAL_START] $(date -Is)" | tee "$OUTDIR/runbook.txt"
echo "[CONFIG] runroot=$RUNROOT" | tee -a "$OUTDIR/runbook.txt"

# ── Eval function ─────────────────────────────────────────────────────

eval_variant() {
    local SEED=$1
    local VARIANT=$2
    local RFT_CKPT=$3
    local BL_CKPT=$4
    local ABLATION=$5
    local TAG="${SEED}_${VARIANT}"

    echo "[EVAL] $TAG start=$(date -Is)" | tee -a "$OUTDIR/runbook.txt"

    if [ ! -f "$RFT_CKPT" ]; then
        echo "  SKIP $TAG: checkpoint not found ($RFT_CKPT)" | tee -a "$OUTDIR/runbook.txt"
        return 0
    fi

    python "$SCRIPT_DIR/eval_ruler_niah.py" \
        --rft_ckpt "$RFT_CKPT" \
        --baseline_ckpt "$BL_CKPT" \
        --tokenizer_path "$TOK" \
        --distractor_path "$DISTRACTOR" \
        --seq_lens 2048 4096 8192 \
        --needle_depth_grid 0.0 0.25 0.5 0.75 1.0 \
        --n_trials 100 \
        --max_new_tokens 5 \
        --chunk_size 512 \
        --seed "$SEED" \
        --prompt_style continuation \
        --value_type digit2 \
        --distractor_docs 2000 \
        --mk_num_keys 3 \
        --mk_num_values 1 \
        --mk_num_queries 1 \
        --rft_ablation "$ABLATION" \
        --outfile "niah_${TAG}.json" \
        2>&1 | tee "$OUTDIR/niah_${TAG}.log"

    echo "[EVAL_DONE] $TAG end=$(date -Is)" | tee -a "$OUTDIR/runbook.txt"
}

# ── Main loop ─────────────────────────────────────────────────────────

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  SEED $SEED"
    echo "============================================================"

    SDIR="$RUNROOT/seed_${SEED}"

    # Determine baseline checkpoint for this seed
    BL_CKPT="$SDIR/phase1/baseline/best_model.pt"
    if [ ! -f "$BL_CKPT" ]; then
        echo "  WARNING: baseline checkpoint not found for seed $SEED, skipping"
        continue
    fi

    # NIAH-trained variants (use Phase 2 checkpoint)
    for VARIANT in "${NIAH_VARIANTS[@]}"; do
        RFT_CKPT="$SDIR/phase2/${VARIANT}/best_model.pt"
        ABLATION="none"
        if [ "$VARIANT" = "rft_lm_disable_ocr" ]; then
            ABLATION="disable_ocr"
        fi
        eval_variant "$SEED" "$VARIANT" "$RFT_CKPT" "$BL_CKPT" "$ABLATION"
    done

    # Phase 1 only variants
    for VARIANT in "${PHASE1_ONLY_VARIANTS[@]}"; do
        RFT_CKPT="$SDIR/phase1/${VARIANT}/best_model.pt"
        ABLATION="none"
        if [ "$VARIANT" = "rft_lm_disable_memory" ]; then
            ABLATION="disable_memory"
        fi
        eval_variant "$SEED" "$VARIANT" "$RFT_CKPT" "$BL_CKPT" "$ABLATION"
    done
done

echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "============================================================"
echo "  Results: $OUTDIR/"
echo "  Summary: $SUMMARY"
echo ""
echo "  Next steps:"
echo "    1. Parse JSON results into paper tables (aggregate across seeds)"
echo "    2. Compute mean ± std across 3 seeds for each (variant, seq_len, depth)"
echo "    3. Generate heatmap figures"
echo ""
echo "[ALL_DONE] $(date -Is)" | tee -a "$OUTDIR/runbook.txt"
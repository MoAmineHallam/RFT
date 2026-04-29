#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_eval_ruler_paper.sh — Paper-grade eval with RULER-faithful
# value types across all 3 seeds.
#
# Run this AFTER run_eval_ruler_faithful.sh tells you which
# config gives honest numbers. Default: CONFIG B (7-digit numbers,
# continuation, 1 needle) which is the correct setup for a
# base model evaluated on an adapted RULER S-NIAH.
#
# Produces: one JSON per (seed, config) → aggregate with mean±std
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_ruler_paper.sh
#
#   # Or override value type if diagnostic showed something else works:
#   VALUE_TYPE=short_int CUDA_VISIBLE_DEVICES=0 bash run_eval_ruler_paper.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Configuration (override via env vars) ─────────────────────
VALUE_TYPE="${VALUE_TYPE:-numbers}"        # numbers | short_int | digit2
PROMPT_STYLE="${PROMPT_STYLE:-continuation}"
NUM_KEYS="${NUM_KEYS:-1}"                  # 1 = S-NIAH, 3 = MK-NIAH
N_TRIALS="${N_TRIALS:-200}"                # RULER standard is 500; 200 is practical
MAX_NEW="${MAX_NEW:-20}"                   # enough for 7-digit number

# ── Paths ─────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"

TAG="${VALUE_TYPE}_${PROMPT_STYLE}_${NUM_KEYS}n"
OUTDIR="$ROOT/eval_ruler_paper_${TAG}"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=(42 43 44)
SEQ_LENS="2048 4096 8192"
DEPTHS="0.0 0.25 0.5 0.75 1.0"

echo "════════════════════════════════════════════════════════════"
echo "  RULER Paper Eval — $TAG"
echo "════════════════════════════════════════════════════════════"
echo "  value_type:   $VALUE_TYPE"
echo "  prompt_style: $PROMPT_STYLE"
echo "  num_keys:     $NUM_KEYS"
echo "  n_trials:     $N_TRIALS"
echo "  seq_lens:     $SEQ_LENS"
echo "  depths:       $DEPTHS"
echo "  seeds:        ${SEEDS[*]}"
echo "  output:       $OUTDIR"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Main loop ─────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    RFT_CKPT="$ROOT/mixed_pilot_v6_seed${SEED}/rft_lm/best_model.pt"

    if [ ! -f "$RFT_CKPT" ]; then
        echo "[SKIP] seed $SEED: checkpoint not found at $RFT_CKPT"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  SEED $SEED — Full model"
    echo "============================================================"

    python "$SCRIPT_DIR/eval_ruler_niah.py" \
        --rft_ckpt "$RFT_CKPT" \
        --baseline_ckpt "$BASELINE_CKPT" \
        --tokenizer_path "$TOK" \
        --distractor_path "$DISTRACTOR" \
        --seq_lens $SEQ_LENS \
        --needle_depth_grid $DEPTHS \
        --n_trials "$N_TRIALS" \
        --max_new_tokens "$MAX_NEW" \
        --chunk_size 512 \
        --seed "$SEED" \
        --prompt_style "$PROMPT_STYLE" \
        --value_type "$VALUE_TYPE" \
        --distractor_docs 2000 \
        --mk_num_keys "$NUM_KEYS" --mk_num_values 1 --mk_num_queries 1 \
        --rft_ablation none \
        --tail_chunk_len 8 \
        --outfile "ruler_${TAG}_seed${SEED}.json" \
        2>&1 | tee "$OUTDIR/seed${SEED}.log"

    echo "[DONE] seed $SEED"

    # ── Ablation: disable memory (only for seed 42) ──────────
    if [ "$SEED" -eq 42 ]; then
        echo ""
        echo "  --- Ablation: memory disabled (seed 42 only) ---"
        python "$SCRIPT_DIR/eval_ruler_niah.py" \
            --rft_ckpt "$RFT_CKPT" \
            --tokenizer_path "$TOK" \
            --distractor_path "$DISTRACTOR" \
            --seq_lens 2048 4096 \
            --needle_depth_grid 0.0 0.5 1.0 \
            --n_trials 50 \
            --max_new_tokens "$MAX_NEW" \
            --chunk_size 512 \
            --seed "$SEED" \
            --prompt_style "$PROMPT_STYLE" \
            --value_type "$VALUE_TYPE" \
            --distractor_docs 2000 \
            --mk_num_keys "$NUM_KEYS" --mk_num_values 1 --mk_num_queries 1 \
            --rft_ablation disable_memory \
            --tail_chunk_len 8 \
            --outfile "ruler_${TAG}_seed${SEED}_no_memory.json" \
            2>&1 | tee "$OUTDIR/seed${SEED}_no_memory.log"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL SEEDS COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "  Results: $OUTDIR/"
echo ""
echo "  Aggregate with:"
echo "    python aggregate_results.py --results_dir $OUTDIR"
echo ""
echo "  For paper, report as:"
echo "    'Adapted RULER S-NIAH (${VALUE_TYPE} values,"
echo "     ${PROMPT_STYLE} prompting, ${NUM_KEYS} needle(s),"
echo "     C4 distractor text, ${N_TRIALS} trials per cell)'"
echo ""
echo "  DO NOT call this 'RULER S-NIAH' without the qualifiers."
echo "════════════════════════════════════════════════════════════"

#!/bin/bash
# run_ruler_niah.sh — RULER S-NIAH sweep over matched retrain checkpoints.
#
# Workflow:
#   1. First run sanity test (short ctx, depth=1.0) on seed 42 checkpoints.
#   2. If baseline gets >= 3/10 hits, proceed to full sweep.
#   3. Full sweep: 3 seeds x 4 seq_lens x 5 depths x 200 trials.
#
# Decision rules after full sweep:
#   - RFT >= 50% and baseline < 10% at 4K+: strong paper signal
#   - Both near 0% at all lengths: task OOD, need retrieval pretraining
#   - RFT modestly beats baseline (30 vs 5): weak signal, needs more work

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1

ROOT=/data3/adam_transfer/AmineHL
RUNROOT=$ROOT/runs_lm/matched_retrains_20260404
DISTRACTOR=$ROOT/data/c4_val.jsonl
TOK=$ROOT/gpt2_tokenizer

NIAH_OUT=$RUNROOT/niah_ruler
mkdir -p "$NIAH_OUT"
RUNBOOK=$NIAH_OUT/runbook.txt
ERRORS=$NIAH_OUT/error_log_summary.txt
: > "$ERRORS"

echo "[NIAH_START] $(date -Is)" | tee -a "$RUNBOOK"

# ── Step 1: sanity test on seed 42 ────────────────────────────────────────
SEED=42
SDIR=$RUNROOT/seed_${SEED}
RFT_CKPT=$SDIR/rft_lm/best_model.pt
BL_CKPT=$SDIR/baseline/best_model.pt

if [ ! -f "$RFT_CKPT" ] || [ ! -f "$BL_CKPT" ]; then
  echo "FAIL: missing checkpoints at $SDIR" | tee -a "$ERRORS" "$RUNBOOK"
  exit 1
fi

echo "[SANITY] running with short ctx, depth=1.0" | tee -a "$RUNBOOK"
conda run -n qwen python "$ROOT/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BL_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 256 \
    --depth 1.0 \
    --n_trials 10 \
    --max_new_tokens 20 \
    --chunk_size 256 \
    2>&1 | tee "$NIAH_OUT/sanity_seed${SEED}.log"

echo ""
echo "[SANITY] Inspect $NIAH_OUT/sanity_seed${SEED}.log BEFORE continuing."
echo "         If baseline hits >= 3/10, run with FULL_SWEEP=1 to continue."
echo "         If 0/10, investigate prompt/decoding before burning GPU-days."

if [ "${FULL_SWEEP:-0}" != "1" ]; then
  echo "[STOP] sanity run complete. Re-run with FULL_SWEEP=1 to continue." | tee -a "$RUNBOOK"
  exit 0
fi

# ── Step 2: full sweep across seeds ───────────────────────────────────────
for SEED in 42 43 44; do
  SDIR=$RUNROOT/seed_${SEED}
  RFT_CKPT=$SDIR/rft_lm/best_model.pt
  BL_CKPT=$SDIR/baseline/best_model.pt
  OUT=$NIAH_OUT/seed_${SEED}
  mkdir -p "$OUT"

  echo "[NIAH] seed=$SEED start=$(date -Is)" | tee -a "$RUNBOOK"
  if ! conda run -n qwen python "$ROOT/eval_ruler_niah.py" \
      --rft_ckpt "$RFT_CKPT" \
      --baseline_ckpt "$BL_CKPT" \
      --tokenizer_path "$TOK" \
      --distractor_path "$DISTRACTOR" \
      --seq_lens 1024 2048 4096 8192 \
      --needle_depth_grid 0.1 0.3 0.5 0.7 0.9 \
      --n_trials 200 \
      --max_new_tokens 3 \
      --chunk_size 512 \
      --seed "$SEED" \
      --distractor_docs 4000 \
      --prompt_style continuation \
      --value_type digit2 \
      --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
      --outfile "niah_seed${SEED}.json" \
      2>&1 | tee "$OUT/niah_seed${SEED}.log"; then
    echo "FAIL niah seed=$SEED" | tee -a "$ERRORS" "$RUNBOOK"
  fi
  # eval_ruler_niah writes to parent.parent of rft_ckpt = $SDIR, so move it
  if [ -f "$SDIR/niah_seed${SEED}.json" ]; then
    mv "$SDIR/niah_seed${SEED}.json" "$OUT/"
    [ -f "$SDIR/niah_seed${SEED}_details.json" ] && mv "$SDIR/niah_seed${SEED}_details.json" "$OUT/"
  fi
  echo "[NIAH_DONE] seed=$SEED end=$(date -Is)" | tee -a "$RUNBOOK"
done

if [ ! -s "$ERRORS" ]; then echo "none" > "$ERRORS"; fi
echo "[NIAH_ALL_DONE] $(date -Is)" | tee -a "$RUNBOOK"

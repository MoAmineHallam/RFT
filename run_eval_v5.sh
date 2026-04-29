#!/bin/bash
# run_eval_v5.sh — RULER S-NIAH evaluation for pilot v5 checkpoint.
#
# v5 fixed the embedding alignment loss: now trains on post-transform mem_ctx
# (gate → LN → out_proj → tanh(α) scaling) instead of raw mem_v.
#
# Sanity check result: 11/20 (55%) at seq=1024, depth=0.5
# vs v4b sanity: ~2/10 (20%) at same setting
# vs baseline: 0/20
#
# This full eval sweeps:
#   - seq_lens: 2048 (training length), 4096, 8192 (generalization)
#   - depths: 0.0, 0.25, 0.5, 0.75, 1.0
#   - 100 trials per cell
#   - Ablations: disable_memory, disable_ocr
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_v5.sh
#   SKIP_GATE=1 CUDA_VISIBLE_DEVICES=0 bash run_eval_v5.sh  # skip sanity pause

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"

RFT_CKPT="$ROOT/mixed_pilot_v5_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

OUTDIR="$ROOT/eval_v5_niah"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verify checkpoints exist ──────────────────────────────────────────
for f in "$RFT_CKPT" "$BASELINE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint not found: $f"
    exit 1
  fi
done
echo "[EVAL_V5] Checkpoints verified."
echo "[EVAL_V5] RFT:      $RFT_CKPT"
echo "[EVAL_V5] Baseline:  $BASELINE_CKPT"

# ── Step 1: Quick sanity (20 trials, short context) ──────────────────
echo ""
echo "================================================================"
echo "  STEP 1: Sanity check (seq=1024, depth=0.5, 20 trials)"
echo "================================================================"

python "$SCRIPT_DIR/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 1024 \
    --depth 0.5 \
    --n_trials 20 \
    --max_new_tokens 5 \
    --chunk_size 512 \
    --prompt_style continuation \
    --value_type digit2 \
    --mk_num_keys 3 \
    --mk_num_queries 1 \
    --distractor_docs 200 \
    --seed 42 \
    2>&1 | tee "$OUTDIR/sanity_v5.log"

echo ""
echo "[SANITY DONE] Check $OUTDIR/sanity_v5.log"
echo ""

# Gate: stop if user wants to inspect sanity first
if [ "${SKIP_GATE:-0}" != "1" ]; then
  echo "[GATE] Set SKIP_GATE=1 to skip this pause, or press Enter to continue."
  read -r -p "Continue to full eval? [Y/n] " response
  if [[ "$response" =~ ^[Nn] ]]; then
    echo "[STOP] Halted after sanity check."
    exit 0
  fi
fi

# ── Step 2: Full depth sweep ─────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 2: Full NIAH depth sweep (v5)"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 8192 \
    --needle_depth_grid 0.0 0.25 0.5 0.75 1.0 \
    --n_trials 100 \
    --max_new_tokens 5 \
    --chunk_size 512 \
    --seed 42 \
    --prompt_style continuation \
    --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 \
    --mk_num_values 1 \
    --mk_num_queries 1 \
    --rft_ablation none \
    --outfile "niah_v5_full.json" \
    2>&1 | tee "$OUTDIR/niah_v5_full.log"

echo ""
echo "[EVAL DONE] Results: $OUTDIR/niah_v5_full.log"

# ── Step 3: Ablation — disable memory ────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 3: Ablation — v5 with memory disabled"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 \
    --needle_depth_grid 0.0 0.5 1.0 \
    --n_trials 50 \
    --max_new_tokens 5 \
    --chunk_size 512 \
    --seed 42 \
    --prompt_style continuation \
    --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 \
    --mk_num_values 1 \
    --mk_num_queries 1 \
    --rft_ablation disable_memory \
    --outfile "niah_v5_no_memory.json" \
    2>&1 | tee "$OUTDIR/niah_v5_no_memory.log"

# ── Step 4: Ablation — disable OCR ───────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 4: Ablation — v5 with OCR disabled"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 \
    --needle_depth_grid 0.0 0.5 1.0 \
    --n_trials 50 \
    --max_new_tokens 5 \
    --chunk_size 512 \
    --seed 42 \
    --prompt_style continuation \
    --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 \
    --mk_num_values 1 \
    --mk_num_queries 1 \
    --rft_ablation disable_ocr \
    --outfile "niah_v5_no_ocr.json" \
    2>&1 | tee "$OUTDIR/niah_v5_no_ocr.log"

echo ""
echo "================================================================"
echo "  ALL V5 EVALUATIONS COMPLETE"
echo "================================================================"
echo "  Results directory: $OUTDIR/"
echo ""
echo "  Compare with v4b results in: $ROOT/eval_v4b_niah/"
echo ""
echo "  Key questions:"
echo "    v5 > v4b at all depths?    → embed fix helped"
echo "    v5 OCR > v5 no-OCR?        → OCR now helps (was harmful in v4b)"
echo "    v5 > 70% at 2K depths 0-0.5? → on track for paper"
echo "    depth=1.0 still ~0%?       → known limitation (same-chunk needle)"
echo ""
echo "  Aggregate results:"
echo "    python aggregate_results.py --results_dir $OUTDIR"
echo "================================================================"
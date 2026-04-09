#!/bin/bash
# run_eval_v4b.sh — RULER S-NIAH evaluation for pilot v4b checkpoint.
#
# This is the CRITICAL first eval: does the v4b checkpoint's 75% training
# lm_acc translate to actual generation accuracy on RULER S-NIAH?
#
# v4b was trained with:
#   - Embedding alignment loss (decode_w=5.0)
#   - 3 needles per context, digit2 values, continuation prompt style
#   - 2-chunk layout: context (512 tok) + query (512 tok)
#   - Resumed from v3 (C4 + synth only) checkpoint
#
# This eval matches training config:
#   - digit2 values (1 BPE token → first generated token IS the answer)
#   - continuation prompt style ("The magic number K is" → model completes)
#   - 3 needles placed in context, 1 queried
#   - chunk_size=512 (same as training)
#
# Decision rules:
#   - v4b >= 50% at 2K: training translates, proceed to multi-seed retrains
#   - v4b 10-50% at 2K: partial transfer, consider training longer or fixing
#   - v4b ~0% at 2K: generation gap persists, needs architectural fix
#   - v4b >> baseline at any length: memory retrieval working in practice

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"

RFT_CKPT="$ROOT/mixed_pilot_v4b_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

OUTDIR="$ROOT/eval_v4b_niah"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verify checkpoints exist ──────────────────────────────────────────
for f in "$RFT_CKPT" "$BASELINE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint not found: $f"
    exit 1
  fi
done
echo "[EVAL_V4B] Checkpoints verified."
echo "[EVAL_V4B] RFT:      $RFT_CKPT"
echo "[EVAL_V4B] Baseline:  $BASELINE_CKPT"

# ── Step 1: Quick sanity (10 trials, short context) ──────────────────
echo ""
echo "================================================================"
echo "  STEP 1: Sanity check (seq=1024, depth=0.5, 10 trials)"
echo "  NOTE: seq_len MUST be >= 2*chunk_size so that chunk 0 populates"
echo "  memory and chunk 1 retrieves from it. With seq=512=chunk_size,"
echo "  memory is empty and retrieval is impossible."
echo "================================================================"

python "$SCRIPT_DIR/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 1024 \
    --depth 0.5 \
    --n_trials 10 \
    --max_new_tokens 5 \
    --chunk_size 512 \
    --prompt_style continuation \
    --value_type digit2 \
    --mk_num_keys 3 \
    --mk_num_queries 1 \
    --distractor_docs 200 \
    --seed 42 \
    2>&1 | tee "$OUTDIR/sanity_v4b.log"

echo ""
echo "[SANITY DONE] Check $OUTDIR/sanity_v4b.log"
echo "  If v4b gets >= 3/10 hits, we have signal. Proceeding to full eval..."
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
# Sweep config matching training:
#   - seq_lens: 2048 (training length), 4096, 8192 (generalization)
#   - depths: 0.0, 0.25, 0.5, 0.75, 1.0 (paper-standard grid)
#   - n_trials: 100 (good CI width, ~30 min per seq_len on 1 GPU)
#   - max_new_tokens: 5 (digit2 = 1 token, but allow a few extra)
#   - continuation prompt style (matches training)

echo ""
echo "================================================================"
echo "  STEP 2: Full NIAH depth sweep"
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
    --outfile "niah_v4b_full.json" \
    2>&1 | tee "$OUTDIR/niah_v4b_full.log"

echo ""
echo "[EVAL DONE] Results: $OUTDIR/niah_v4b_full.log"

# ── Step 3: Ablation — disable memory ────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 3: Ablation — RFT-LM with memory disabled"
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
    --outfile "niah_v4b_no_memory.json" \
    2>&1 | tee "$OUTDIR/niah_v4b_no_memory.log"

# ── Step 4: Ablation — disable OCR ───────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 4: Ablation — RFT-LM with OCR disabled"
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
    --outfile "niah_v4b_no_ocr.json" \
    2>&1 | tee "$OUTDIR/niah_v4b_no_ocr.log"

echo ""
echo "================================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "================================================================"
echo "  Results directory: $OUTDIR/"
echo ""
echo "  Decision guide:"
echo "    v4b >= 50% at 2K  → proceed to multi-seed retrains"
echo "    v4b 10-50% at 2K  → train longer (v4c), then re-eval"
echo "    v4b ~0% at 2K     → generation gap persists, investigate"
echo "    v4b > v4b_no_ocr  → OCR helping at eval time too"
echo "    v4b > v4b_no_mem  → memory is the key differentiator"
echo ""

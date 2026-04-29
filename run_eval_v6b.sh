#!/bin/bash
# run_eval_v6b.sh — RULER S-NIAH evaluation for pilot v6b checkpoint (M=256).
#
# v6b is the parallel variant with mem_top_m=256 instead of 64.
# Testing whether OCR disambiguation matters when there are 4x more candidates.
#
# Usage (run on GPU 1 while v6 eval runs on GPU 0):
#   SKIP_GATE=1 CUDA_VISIBLE_DEVICES=1 bash run_eval_v6b.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"

RFT_CKPT="$ROOT/mixed_pilot_v6b_m256_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

OUTDIR="$ROOT/eval_v6b_niah"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for f in "$RFT_CKPT" "$BASELINE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint not found: $f"
    exit 1
  fi
done
echo "[EVAL_V6B] Checkpoints verified."

# NOTE: eval_ruler_niah.py loads the model with the checkpoint's mem_top_m config
# automatically via the "config" key in the .pt file — no CLI override needed.

# ── Step 1: Sanity ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 1: Sanity check (v6b, M=256)"
echo "================================================================"

python "$SCRIPT_DIR/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 1024 --depth 0.5 --n_trials 20 \
    --max_new_tokens 5 --chunk_size 512 \
    --prompt_style continuation --value_type digit2 \
    --mk_num_keys 3 --mk_num_queries 1 \
    --distractor_docs 200 --seed 42 \
    2>&1 | tee "$OUTDIR/sanity_v6b.log"

if [ "${SKIP_GATE:-0}" != "1" ]; then
  read -r -p "Continue to full eval? [Y/n] " response
  if [[ "$response" =~ ^[Nn] ]]; then exit 0; fi
fi

# ── Step 2: Full depth sweep ─────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 2: Full NIAH depth sweep (v6b)"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 8192 \
    --needle_depth_grid 0.0 0.25 0.5 0.75 1.0 \
    --n_trials 100 --max_new_tokens 5 --chunk_size 512 \
    --seed 42 --prompt_style continuation --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
    --rft_ablation none \
    --outfile "niah_v6b_full.json" \
    2>&1 | tee "$OUTDIR/niah_v6b_full.log"

# ── Step 3: Ablation — disable OCR (the critical comparison) ─────────
echo ""
echo "================================================================"
echo "  STEP 3: Ablation — v6b with OCR disabled"
echo "  This is THE critical test: does OCR help when M=256?"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 8192 \
    --needle_depth_grid 0.0 0.25 0.5 0.75 \
    --n_trials 100 --max_new_tokens 5 --chunk_size 512 \
    --seed 42 --prompt_style continuation --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
    --rft_ablation disable_ocr \
    --outfile "niah_v6b_no_ocr.json" \
    2>&1 | tee "$OUTDIR/niah_v6b_no_ocr.log"

echo ""
echo "================================================================"
echo "  V6B EVALUATIONS COMPLETE"
echo "================================================================"
echo "  Results: $OUTDIR/"
echo ""
echo "  KEY COMPARISON:"
echo "    v6b full vs v6b no-ocr → does OCR help at M=256?"
echo "    v6b vs v6 → does larger M help retrieval overall?"
echo "================================================================"
#!/bin/bash
# run_eval_v6.sh — RULER S-NIAH evaluation for pilot v6 checkpoint.
#
# v6 achieved 17/20 (85%) on sanity at seq=1024, depth=0.5.
# Expected full eval: 80-90% at 2K-8K depths 0.0-0.5.
#
# Usage:
#   SKIP_GATE=1 CUDA_VISIBLE_DEVICES=0 bash run_eval_v6.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"

RFT_CKPT="$ROOT/mixed_pilot_v6_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

OUTDIR="$ROOT/eval_v6_niah"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for f in "$RFT_CKPT" "$BASELINE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint not found: $f"
    exit 1
  fi
done
echo "[EVAL_V6] Checkpoints verified."

# ── Step 1: Sanity ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 1: Sanity check (seq=1024, depth=0.5, 20 trials)"
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
    2>&1 | tee "$OUTDIR/sanity_v6.log"

if [ "${SKIP_GATE:-0}" != "1" ]; then
  read -r -p "Continue to full eval? [Y/n] " response
  if [[ "$response" =~ ^[Nn] ]]; then exit 0; fi
fi

# ── Step 2: Full depth sweep ─────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 2: Full NIAH depth sweep (v6)"
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
    --outfile "niah_v6_full.json" \
    2>&1 | tee "$OUTDIR/niah_v6_full.log"

# ── Step 3: Ablation — disable memory ────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 3: Ablation — v6 with memory disabled"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 \
    --needle_depth_grid 0.0 0.5 1.0 \
    --n_trials 50 --max_new_tokens 5 --chunk_size 512 \
    --seed 42 --prompt_style continuation --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
    --rft_ablation disable_memory \
    --outfile "niah_v6_no_memory.json" \
    2>&1 | tee "$OUTDIR/niah_v6_no_memory.log"

# ── Step 4: Ablation — disable OCR ───────────────────────────────────
echo ""
echo "================================================================"
echo "  STEP 4: Ablation — v6 with OCR disabled"
echo "================================================================"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_lens 2048 4096 \
    --needle_depth_grid 0.0 0.5 1.0 \
    --n_trials 50 --max_new_tokens 5 --chunk_size 512 \
    --seed 42 --prompt_style continuation --value_type digit2 \
    --distractor_docs 2000 \
    --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
    --rft_ablation disable_ocr \
    --outfile "niah_v6_no_ocr.json" \
    2>&1 | tee "$OUTDIR/niah_v6_no_ocr.log"

echo ""
echo "================================================================"
echo "  V6 EVALUATIONS COMPLETE"
echo "================================================================"
echo "  Results: $OUTDIR/"
echo "================================================================"
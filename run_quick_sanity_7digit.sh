#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_quick_sanity_7digit.sh — 5-minute reality check
#
# Runs 20 trials each of digit2 vs 7-digit numbers at seq=1024,
# depth=0.5. Prints raw predictions so you can eyeball what the
# model actually generates for multi-token values.
#
# Run this FIRST before the full diagnostic.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_quick_sanity_7digit.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RFT_CKPT="$ROOT/mixed_pilot_v6_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"
OUTDIR="$ROOT/eval_ruler_faithful"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════"
echo "  QUICK SANITY: digit2 vs 7-digit numbers"
echo "  20 trials each, seq=1024, depth=0.5"
echo "  Look at the raw predictions to understand what's happening"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Test 1: digit2 (your current easy setup) ─────────────────
echo "══════════════════════════════════════"
echo "  TEST 1: digit2 values (current)"
echo "══════════════════════════════════════"

python "$SCRIPT_DIR/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 1024 --depth 0.5 --n_trials 20 \
    --max_new_tokens 10 --chunk_size 512 \
    --prompt_style continuation --value_type digit2 \
    --mk_num_keys 3 --mk_num_queries 1 \
    --distractor_docs 200 --seed 42 \
    --tail_chunk_len 8 \
    2>&1 | tee "$OUTDIR/quick_sanity_digit2.log"

echo ""
echo "══════════════════════════════════════"
echo "  TEST 2: 7-digit numbers (RULER standard)"
echo "══════════════════════════════════════"

python "$SCRIPT_DIR/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 1024 --depth 0.5 --n_trials 20 \
    --max_new_tokens 20 --chunk_size 512 \
    --prompt_style continuation --value_type numbers \
    --mk_num_keys 1 --mk_num_queries 1 \
    --distractor_docs 200 --seed 42 \
    --tail_chunk_len 8 \
    2>&1 | tee "$OUTDIR/quick_sanity_numbers.log"

echo ""
echo "══════════════════════════════════════"
echo "  TEST 3: 3-digit numbers (middle ground)"
echo "══════════════════════════════════════"

python "$SCRIPT_DIR/niah_sanity.py" \
    --rft_ckpt "$RFT_CKPT" \
    --baseline_ckpt "$BASELINE_CKPT" \
    --tokenizer_path "$TOK" \
    --distractor_path "$DISTRACTOR" \
    --seq_len 1024 --depth 0.5 --n_trials 20 \
    --max_new_tokens 10 --chunk_size 512 \
    --prompt_style continuation --value_type short_int \
    --mk_num_keys 1 --mk_num_queries 1 \
    --distractor_docs 200 --seed 42 \
    --tail_chunk_len 8 \
    2>&1 | tee "$OUTDIR/quick_sanity_short_int.log"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DONE — Check the pred= lines in the logs above"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  KEY THINGS TO LOOK FOR:"
echo "    digit2:    pred should show the 2-digit number → ~90% hit"
echo "    numbers:   pred should show 7 digits → likely much lower"
echo "    short_int: pred should show 3 digits → somewhere in between"
echo ""
echo "  LIKELY OUTCOMES:"
echo "    1. numbers gets ~0%: model can retrieve but can't generate"
echo "       multi-token values → need to retrain with numbers"
echo "    2. numbers gets 30-60%: model partially handles it"
echo "       → paper reports this honestly, still publishable"
echo "    3. numbers gets 70%+: digit2 shortcut was unnecessary"
echo "       → great, switch paper to numbers immediately"
echo ""
echo "  If outcome 1: short_int (3-digit) may be a good middle ground."
echo "  GPT-2 tokenizer encodes ' 123' as 1-2 tokens for 3-digit ints,"
echo "  so it's harder than digit2 but not as hard as 7-digit."
echo ""
echo "  Next: run run_eval_ruler_faithful.sh for full diagnostic"
echo "════════════════════════════════════════════════════════════"

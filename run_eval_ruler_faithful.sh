#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_eval_ruler_faithful.sh — Reality check: How does RFT-LM
# perform under RULER-faithful settings vs your current easy setup?
#
# PROBLEM: Your current eval uses digit2 (1 BPE token answer) +
# continuation prompting + 3 needles. Official RULER S-NIAH uses
# 7-digit numbers/words/UUIDs + instruct template + 1 needle +
# 500 trials. Titans reports against the official setup.
#
# This script runs 4 configs on the v6 seed42 checkpoint:
#
#   CONFIG A: Your current setup (digit2, continuation, 3 needles)
#             → baseline comparison, should match ~92% mean
#
#   CONFIG B: RULER-faithful for base models
#             (numbers 7-digit, continuation, 1 needle)
#             → the honest "adapted RULER" number for the paper
#
#   CONFIG C: RULER-faithful full
#             (numbers 7-digit, instruct, 1 needle)
#             → true RULER comparison (will likely be low for base model)
#
#   CONFIG D: Intermediate difficulty
#             (numbers 7-digit, continuation, 3 needles)
#             → isolates value_type effect from needle count effect
#
# Each config runs 100 trials at depths 0.0, 0.5, 1.0 and
# seq_lens 2048, 4096. Takes ~2-4 hours total on 1 GPU.
#
# After this, you'll know:
#   - How much of your 92% comes from the digit2 shortcut
#   - Whether continuation vs instruct matters for your base model
#   - Whether 1 vs 3 needles matters
#   - What number to actually put in the paper
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_ruler_faithful.sh
#   # Or skip the easy config if you trust existing numbers:
#   SKIP_A=1 CUDA_VISIBLE_DEVICES=0 bash run_eval_ruler_faithful.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Paths ─────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"

# Use the v6 seed42 checkpoint (your best model)
RFT_CKPT="$ROOT/mixed_pilot_v6_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

OUTDIR="$ROOT/eval_ruler_faithful"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verify ────────────────────────────────────────────────────
for f in "$RFT_CKPT" "$BASELINE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint not found: $f"
    exit 1
  fi
done
echo "[RULER_FAITHFUL] Checkpoints verified."
echo "[RULER_FAITHFUL] Output: $OUTDIR"
echo ""

# ── Common params ─────────────────────────────────────────────
# Using tail_chunk_len=8 (your validated fix for d=1.0)
COMMON_ARGS=(
    --rft_ckpt "$RFT_CKPT"
    --baseline_ckpt "$BASELINE_CKPT"
    --tokenizer_path "$TOK"
    --distractor_path "$DISTRACTOR"
    --chunk_size 512
    --seed 42
    --distractor_docs 2000
    --rft_ablation none
    --tail_chunk_len 8
)

# Depths: 0.0, 0.5, 1.0 for quick diagnostic
# Seq lens: 2048 (training length) and 4096 (generalization)
# 100 trials per cell (enough for diagnostic, scale to 500 for paper)
DIAG_DEPTHS="0.0 0.5 1.0"
DIAG_LENS="2048 4096"
DIAG_TRIALS=100

# ══════════════════════════════════════════════════════════════
#  CONFIG A: Your current easy setup (baseline comparison)
# ══════════════════════════════════════════════════════════════
if [ "${SKIP_A:-0}" != "1" ]; then
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  CONFIG A: Current setup (digit2 + continuation + 3 needles)"
echo "  This should match your existing ~92% numbers."
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lens $DIAG_LENS \
    --needle_depth_grid $DIAG_DEPTHS \
    --n_trials $DIAG_TRIALS \
    --max_new_tokens 5 \
    --prompt_style continuation \
    --value_type digit2 \
    --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
    --outfile "faithful_A_digit2_cont_3needle.json" \
    2>&1 | tee "$OUTDIR/config_A.log"

echo "[CONFIG A DONE]"
fi

# ══════════════════════════════════════════════════════════════
#  CONFIG B: RULER-adapted for base models (THE KEY TEST)
#  - 7-digit numbers (RULER standard)
#  - continuation prompt (appropriate for base model)
#  - 1 needle (RULER S-NIAH standard)
#  This is what you should report in the paper as
#  "adapted S-NIAH" if your model is not instruction-tuned.
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  CONFIG B: 7-digit numbers + continuation + 1 needle"
echo "  *** THIS IS THE CRITICAL TEST ***"
echo "  RULER-faithful values with base-model-appropriate prompting."
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lens $DIAG_LENS \
    --needle_depth_grid $DIAG_DEPTHS \
    --n_trials $DIAG_TRIALS \
    --max_new_tokens 20 \
    --prompt_style continuation \
    --value_type numbers \
    --mk_num_keys 1 --mk_num_values 1 --mk_num_queries 1 \
    --outfile "faithful_B_numbers_cont_1needle.json" \
    2>&1 | tee "$OUTDIR/config_B.log"

echo "[CONFIG B DONE]"

# ══════════════════════════════════════════════════════════════
#  CONFIG C: Full RULER-faithful (instruct template)
#  - 7-digit numbers
#  - RULER instruct template (model must follow instructions)
#  - 1 needle
#  Expect this to be LOW for a 125M base model. That's fine —
#  it tells you whether the instruct template is OOD.
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  CONFIG C: 7-digit numbers + instruct + 1 needle"
echo "  Full RULER protocol. Likely low for 125M base model."
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lens $DIAG_LENS \
    --needle_depth_grid $DIAG_DEPTHS \
    --n_trials $DIAG_TRIALS \
    --max_new_tokens 20 \
    --prompt_style instruct \
    --value_type numbers \
    --mk_num_keys 1 --mk_num_values 1 --mk_num_queries 1 \
    --outfile "faithful_C_numbers_inst_1needle.json" \
    2>&1 | tee "$OUTDIR/config_C.log"

echo "[CONFIG C DONE]"

# ══════════════════════════════════════════════════════════════
#  CONFIG D: Isolate value_type effect
#  - 7-digit numbers (harder values)
#  - continuation (same as your current)
#  - 3 needles (same as your current)
#  Comparing A vs D isolates: how much does digit2→numbers hurt?
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  CONFIG D: 7-digit numbers + continuation + 3 needles"
echo "  Isolates value_type effect (A vs D comparison)."
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/eval_ruler_niah.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lens $DIAG_LENS \
    --needle_depth_grid $DIAG_DEPTHS \
    --n_trials $DIAG_TRIALS \
    --max_new_tokens 20 \
    --prompt_style continuation \
    --value_type numbers \
    --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
    --outfile "faithful_D_numbers_cont_3needle.json" \
    2>&1 | tee "$OUTDIR/config_D.log"

echo "[CONFIG D DONE]"

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL CONFIGS COMPLETE — Results in $OUTDIR/"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Compare these configs to understand your real accuracy:"
echo ""
echo "  A (digit2+cont+3n)    → your current reported number (~92%)"
echo "  B (numbers+cont+1n)   → honest base-model RULER number"
echo "  C (numbers+inst+1n)   → full RULER protocol"
echo "  D (numbers+cont+3n)   → isolate value_type from needle count"
echo ""
echo "  A vs D → how much does digit2 vs 7-digit matter?"
echo "  D vs B → how much does 3 vs 1 needle matter?"
echo "  B vs C → how much does continuation vs instruct matter?"
echo ""
echo "  DECISION GUIDE:"
echo "    B ≥ 70%  → report B numbers as 'adapted S-NIAH' in paper"
echo "    B 30-70% → still publishable but reframe expectations"
echo "    B < 30%  → need to retrain with 7-digit values before paper"
echo ""
echo "  If B >> C: paper should say 'continuation-style evaluation'"
echo "             and explain why (base model, not instruction-tuned)"
echo ""
echo "  Next: if B looks good, run the full sweep:"
echo "    run_eval_ruler_paper.sh (all depths, all seeds, 500 trials)"
echo "════════════════════════════════════════════════════════════"

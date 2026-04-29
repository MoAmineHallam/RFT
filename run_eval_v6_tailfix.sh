#!/bin/bash
# run_eval_v6_tailfix.sh — Test the tail-chunk-split fix for depth=1.0 on v6.
#
# Phase 1: depth=1.0 only, 50 trials, seq=2048, sweep tail_chunk_len in {0,1,4,8,16}
#          to find the smallest value that unlocks depth=1.0 without hurting.
#
# Phase 2: Full RULER S-NIAH sweep (all depths) with the best tail_chunk_len,
#          to confirm d=0.0-0.75 is preserved.
#
# Usage:
#   SKIP_GATE=1 CUDA_VISIBLE_DEVICES=0 bash run_eval_v6_tailfix.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RFT_CKPT="$ROOT/mixed_pilot_v6_seed42/rft_lm/best_model.pt"
BASELINE_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
TOK="$ROOT/gpt2_tokenizer"
DISTRACTOR="$ROOT/data/c4_train.jsonl"

OUTDIR="$ROOT/eval_v6_tailfix"
mkdir -p "$OUTDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for f in "$RFT_CKPT" "$BASELINE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint not found: $f"
    exit 1
  fi
done
echo "[TAILFIX] Checkpoints verified."

# ── Phase 1: depth=1.0 sweep over tail_chunk_len ─────────────────────
echo ""
echo "================================================================"
echo "  PHASE 1: depth=1.0 only, sweep tail_chunk_len"
echo "  Hypothesis: small tail split unlocks depth=1.0"
echo "================================================================"

for K in 0 1 4 8 16; do
  echo ""
  echo "--- tail_chunk_len = $K ---"
  python "$SCRIPT_DIR/eval_ruler_niah.py" \
      --rft_ckpt "$RFT_CKPT" \
      --baseline_ckpt "$BASELINE_CKPT" \
      --tokenizer_path "$TOK" \
      --distractor_path "$DISTRACTOR" \
      --seq_lens 2048 \
      --needle_depth_grid 1.0 \
      --n_trials 50 --max_new_tokens 5 --chunk_size 512 \
      --seed 42 --prompt_style continuation --value_type digit2 \
      --distractor_docs 2000 \
      --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
      --rft_ablation none \
      --tail_chunk_len $K \
      --outfile "tailfix_phase1_K${K}.json" \
      2>&1 | tee "$OUTDIR/phase1_K${K}.log"
done

echo ""
echo "================================================================"
echo "  PHASE 1 COMPLETE"
echo "  Inspect the depth=1.0 accuracy for each K in $OUTDIR/"
echo "  Pick the smallest K that gives >50% at d=1.0 → use in Phase 2"
echo "================================================================"

if [ "${SKIP_GATE:-0}" != "1" ]; then
  read -r -p "Continue to Phase 2 (full sweep)? Enter K to use, or 'n' to abort: " response
  if [[ "$response" =~ ^[Nn] ]]; then exit 0; fi
  K_BEST="$response"
else
  K_BEST="${K_BEST:-8}"
fi

# ── Phase 2: full sweep with best K ──────────────────────────────────
echo ""
echo "================================================================"
echo "  PHASE 2: Full depth sweep with tail_chunk_len=$K_BEST"
echo "  Critical: d=0.0-0.75 must stay ≥ 90% (i.e. fix doesn't break things)"
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
    --tail_chunk_len "$K_BEST" \
    --outfile "niah_v6_tailfix_K${K_BEST}.json" \
    2>&1 | tee "$OUTDIR/phase2_full_K${K_BEST}.log"

echo ""
echo "================================================================"
echo "  TAILFIX EVAL COMPLETE"
echo "================================================================"
echo "  Results: $OUTDIR/"
echo ""
echo "  Compare to v6 baseline (no tail fix):"
echo "    Expected d=0.0-0.5: 96-97% (unchanged)"
echo "    Expected d=0.75:    68-74% (unchanged)"
echo "    Expected d=1.0:     0→?    (this is the test)"
echo "================================================================"
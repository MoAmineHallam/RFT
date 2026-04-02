#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# RFT-LM PILOT EXPERIMENT — 3-hour window on 2×L40S
# ═══════════════════════════════════════════════════════════════
#
# PURPOSE: Verify that RFT-LM architecture works before committing
# to overnight training. Tests:
#   1. Forward/backward pass works
#   2. Loss decreases (model is learning)
#   3. Memory bank grows correctly across chunks
#   4. Memory usage fits in 46GB L40S
#   5. Rough throughput estimate for planning overnight run
#
# RUN:
#   chmod +x pilot_3h.sh
#   ./pilot_3h.sh
#
# EXPECTED TIME: ~2-3 hours total
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/RFT_LM.py"
OUTDIR="$SCRIPT_DIR/runs_lm"

mkdir -p "$OUTDIR"

echo "════════════════════════════════════════════════════════════"
echo "  PILOT 1/4: Tiny model, short sequence (sanity check)"
echo "════════════════════════════════════════════════════════════"
# Should complete in <5 minutes
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
  --gpu 0 \
  --outdir "$OUTDIR/pilot_tiny" \
  --vocab_size 32000 \
  --d_model 256 \
  --n_layers 4 \
  --n_heads 4 \
  --ff_mult 4 \
  --window_size 256 \
  --memory_layer_idx 2 \
  --mem_top_m 32 \
  --ocr_dim 128 \
  --total_seq_len 2048 \
  --chunk_size 256 \
  --batch_size 4 \
  --lr 3e-4 \
  --pilot_steps 50

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PILOT 2/4: Target-scale model, short sequence"
echo "════════════════════════════════════════════════════════════"
# Tests the actual model size we'll use for the paper (~50M params)
# Should complete in ~15-20 minutes
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
  --gpu 0 \
  --outdir "$OUTDIR/pilot_target_scale" \
  --vocab_size 32000 \
  --d_model 512 \
  --n_layers 8 \
  --n_heads 8 \
  --ff_mult 4 \
  --window_size 512 \
  --memory_layer_idx 4 \
  --mem_top_m 64 \
  --ocr_dim 256 \
  --total_seq_len 4096 \
  --chunk_size 512 \
  --batch_size 4 \
  --lr 3e-4 \
  --pilot_steps 100

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PILOT 3/4: Target-scale model, LONG sequence (8K)"
echo "════════════════════════════════════════════════════════════"
# Tests memory scaling — baseline would OOM here
# Should complete in ~30 minutes
CUDA_VISIBLE_DEVICES=4 python "$SCRIPT" \
  --gpu 0 \
  --outdir "$OUTDIR/pilot_long_8k" \
  --vocab_size 32000 \
  --d_model 512 \
  --n_layers 8 \
  --n_heads 8 \
  --ff_mult 4 \
  --window_size 512 \
  --memory_layer_idx 4 \
  --mem_top_m 64 \
  --ocr_dim 256 \
  --total_seq_len 8192 \
  --chunk_size 512 \
  --batch_size 2 \
  --lr 3e-4 \
  --pilot_steps 50

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PILOT 4/4: Target-scale, VERY LONG sequence (16K)"
echo "════════════════════════════════════════════════════════════"
# The money test — 16K context. Baseline would definitely OOM.
# If this works, we can handle RULER/BABILong contexts.
# Should complete in ~45-60 minutes
CUDA_VISIBLE_DEVICES=4 python "$SCRIPT" \
  --gpu 0 \
  --outdir "$OUTDIR/pilot_long_16k" \
  --vocab_size 32000 \
  --d_model 512 \
  --n_layers 8 \
  --n_heads 8 \
  --ff_mult 4 \
  --window_size 512 \
  --memory_layer_idx 4 \
  --mem_top_m 64 \
  --ocr_dim 256 \
  --total_seq_len 16384 \
  --chunk_size 512 \
  --batch_size 1 \
  --lr 3e-4 \
  --pilot_steps 30

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL PILOTS COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Check results in:"
echo "  $OUTDIR/pilot_tiny/pilot_results.json"
echo "  $OUTDIR/pilot_target_scale/pilot_results.json"
echo "  $OUTDIR/pilot_long_8k/pilot_results.json"
echo "  $OUTDIR/pilot_long_16k/pilot_results.json"
echo ""
echo "KEY THINGS TO VERIFY:"
echo "  1. loss_decreasing = true for all pilots"
echo "  2. peak_mem_mb < 40000 (fits in L40S 46GB)"
echo "  3. memory_size grows correctly across chunks"
echo "  4. No OOM errors"
echo ""
echo "If all pass, you're ready for overnight training!"
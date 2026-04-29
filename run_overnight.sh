#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# OVERNIGHT TRAINING — RFT-LM + Baseline on C4
# ═══════════════════════════════════════════════════════════════
#
# Trains two models sequentially on GPUs 3,4:
#   1. RFT-LM (with memory + OCR)  ~5-6 hours
#   2. Baseline Transformer          ~4-5 hours
#
# Total: ~10-12 hours
#
# RUN:
#   tmux new -s overnight
#   conda activate qwen && bash run_overnight.sh 2>&1 | tee overnight_log.txt
#   # Ctrl+B, D to detach
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="$SCRIPT_DIR/data/c4_train.jsonl"
OUTDIR="/data/AmineHL_data/overnight"

echo "════════════════════════════════════════════════════════════"
echo "  OVERNIGHT TRAINING — $(date)"
echo "  Data: $DATA"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "════════════════════════════════════════════════════════════"
echo ""

# ─── Model 1: RFT-LM ─────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  [1/2] Training RFT-LM (with memory + OCR)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/train_overnight.py" \
  --data_path "$DATA" \
  --outdir "$OUTDIR" \
  --model rft_lm \
  --vocab_size 50257 \
  --tokenizer_path "$SCRIPT_DIR/gpt2_tokenizer" \
  --d_model 768 \
  --n_layers 12 \
  --n_heads 12 \
  --ff_mult 4 \
  --dropout 0.0 \
  --window_size 512 \
  --memory_layer_idx 6 \
  --mem_top_m 64 \
  --ocr_dim 256 \
  --total_seq_len 2048 \
  --chunk_size 512 \
  --batch_size 4 \
  --lr 3e-4 \
  --epochs 2 \
  --warmup_steps 500 \
  --save_every 1000 \
  --log_interval 20 \
  --num_workers 4 \
  --seed 42 \
  --n_gpus 4

echo ""
echo "  RFT-LM DONE: $(date)"
echo ""

# ─── Model 2: Baseline Transformer ───────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  [2/2] Training Baseline Transformer (no memory)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/train_overnight.py" \
  --data_path "$DATA" \
  --outdir "$OUTDIR" \
  --model baseline \
  --vocab_size 50257 \
  --tokenizer_path "$SCRIPT_DIR/gpt2_tokenizer" \
  --d_model 768 \
  --n_layers 12 \
  --n_heads 12 \
  --ff_mult 4 \
  --dropout 0.0 \
  --window_size 512 \
  --total_seq_len 2048 \
  --chunk_size 512 \
  --batch_size 4 \
  --lr 3e-4 \
  --epochs 2 \
  --warmup_steps 500 \
  --save_every 1000 \
  --log_interval 20 \
  --num_workers 4 \
  --seed 42 \
  --n_gpus 4

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL TRAINING COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Results saved in: $OUTDIR/"
echo "  $OUTDIR/rft_lm/train_results.json"
echo "  $OUTDIR/rft_lm/best_model.pt"
echo "  $OUTDIR/baseline/train_results.json"
echo "  $OUTDIR/baseline/best_model.pt"
echo ""
echo "Compare final losses to see if RFT memory helps!"

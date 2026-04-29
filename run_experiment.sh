#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# RFT-LM V2 EXPERIMENT — Retrain with OCR loss + RULER S-NIAH eval
# ═══════════════════════════════════════════════════════════════
#
# This script runs the definitive experiment:
#   1. Retrain RFT-LM WITH OCR contrastive loss (the fix)
#   2. Retrain Baseline (same budget, fair comparison)
#   3. Evaluate both on held-out C4 perplexity
#   4. Evaluate both on RULER S-NIAH (the paper benchmark)
#
# Total: ~12-14 hours on 4×4090
#
# RUN:
#   tmux new -s experiment
#   conda activate challenge
#   bash run_experiment.sh 2>&1 | tee experiment_log.txt
#   # Ctrl+B, D to detach
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="$SCRIPT_DIR/data/c4_train.jsonl"
VAL_DATA="$SCRIPT_DIR/data/c4_val.jsonl"
OUTDIR="/data/AmineHL_data/experiment_v2"
TOKENIZER="$SCRIPT_DIR/gpt2_tokenizer"

echo "════════════════════════════════════════════════════════════"
echo "  RFT-LM V2 EXPERIMENT — $(date)"
echo "  Data: $DATA"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Output: $OUTDIR"
echo "════════════════════════════════════════════════════════════"
echo ""

# ─── Phase 1: Train RFT-LM (with OCR contrastive loss) ──────
echo "════════════════════════════════════════════════════════════"
echo "  [1/4] Training RFT-LM v2 (memory + OCR + contrastive loss)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/train_overnight.py" \
  --data_path "$DATA" \
  --outdir "$OUTDIR" \
  --model rft_lm \
  --vocab_size 50257 \
  --tokenizer_path "$TOKENIZER" \
  --d_model 768 \
  --n_layers 12 \
  --n_heads 12 \
  --ff_mult 4 \
  --dropout 0.0 \
  --window_size 512 \
  --memory_layer_idx 6 \
  --mem_top_m 64 \
  --ocr_dim 256 \
  --ocr_loss_weight 0.05 \
  --ocr_margin 0.10 \
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
echo "  RFT-LM v2 DONE: $(date)"
echo ""

# ─── Phase 2: Train Baseline ────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  [2/4] Training Baseline Transformer (no memory)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/train_overnight.py" \
  --data_path "$DATA" \
  --outdir "$OUTDIR" \
  --model baseline \
  --vocab_size 50257 \
  --tokenizer_path "$TOKENIZER" \
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
echo "  Baseline DONE: $(date)"
echo ""

# ─── Phase 3: Held-out Perplexity ───────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  [3/4] Evaluating held-out C4 perplexity"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/eval_perplexity.py" \
  --val_path "$VAL_DATA" \
  --rft_ckpt "$OUTDIR/rft_lm/best_model.pt" \
  --baseline_ckpt "$OUTDIR/baseline/best_model.pt" \
  --tokenizer_path "$TOKENIZER" \
  --seq_lens 2048 4096 8192 \
  --max_docs 2000 \
  --max_seqs 100 \
  --batch_size 1

echo ""
echo "  Perplexity eval DONE: $(date)"
echo ""

# ─── Phase 4: RULER S-NIAH ──────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  [4/4] RULER S-NIAH evaluation"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/eval_ruler_niah.py" \
  --rft_ckpt "$OUTDIR/rft_lm/best_model.pt" \
  --baseline_ckpt "$OUTDIR/baseline/best_model.pt" \
  --tokenizer_path "$TOKENIZER" \
  --distractor_path "$VAL_DATA" \
  --seq_lens 2048 4096 8192 16384 \
  --chunk_size 512 \
  --n_trials 100 \
  --distractor_docs 500

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  Perplexity: $OUTDIR/eval_results.json"
echo "  RULER NIAH: $OUTDIR/niah_results.json"
echo ""
echo "Compare your RULER S-NIAH numbers against:"
echo "  Titans MAC (125M):    ~96-99% at 4K-16K"
echo "  Transformer+ (125M):  ~92% at 4K, drops to ~70% at 16K"
echo "  Mamba-2 (125M):       ~90% at 4K, drops to ~65% at 16K"
echo ""
echo "If RFT-LM > 90% and holds at 8K+, you have a paper."
echo "If RFT-LM > Baseline by 5%+, the memory layer clearly helps."
echo "If RFT-LM ~ Titans, you have a strong paper."

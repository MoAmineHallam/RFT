#!/bin/bash
# run_train_v5.sh — Train pilot v5 with corrected embedding alignment loss.
#
# v4b achieved 42-61% RULER S-NIAH accuracy despite a bug where the
# embedding alignment loss trained on raw mem_v (before gate/ln/proj/scale).
# The LM loss alone provided enough signal for ~50% accuracy.
#
# v5 fixes niah_batch.py: embed_loss now operates on post-transform mem_ctx
# at the query position. This should:
#   (a) Train gate, mem_ln, out_proj, mem_gate_alpha to preserve alignment
#   (b) Push mem_gate_alpha larger (currently tanh(0.1) ≈ 10% signal)
#   (c) OCR should learn to help (v4b OCR was slightly harmful)
#
# Expected improvement: 50% → 70-80%+ at 2K-8K depths 0.0-0.5.
#
# Resumes from v3 checkpoint (C4 + synth only), NOT from v4b, because
# v4b's gate/ln/proj were shaped by the buggy loss.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_train_v5.sh
#   # Monitor: tail -f $OUTDIR/rft_lm/train_metrics.jsonl | grep niah

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RESUME_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTDIR="$ROOT/mixed_pilot_v5_seed42"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verify prerequisites ─────────────────────────────────────────────
if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: v3 checkpoint not found: $RESUME_CKPT"
    exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "ERROR: training data not found: $DATA"
    exit 1
fi

echo "================================================================"
echo "  RFT-LM Pilot v5 — Corrected Embedding Alignment Loss"
echo "================================================================"
echo "  Resume from: $RESUME_CKPT"
echo "  Output:      $OUTDIR"
echo "  GPU:         $CUDA_VISIBLE_DEVICES"
echo ""
echo "  Key change: embed_loss on post-transform mem_ctx (not raw mem_v)"
echo "  Expected:   50% → 70-80%+ NIAH accuracy"
echo "================================================================"
echo ""

# ── Training ──────────────────────────────────────────────────────────
# Pre-create output dir so tee doesn't fail
mkdir -p "$OUTDIR/rft_lm"

# Match v4b hyperparams exactly, except using the fixed niah_batch.py
python "$SCRIPT_DIR/train_overnight.py" \
    --data_path "$DATA" \
    --outdir "$OUTDIR" \
    --model rft_lm \
    --tokenizer_path "$TOK" \
    --vocab_size 50257 \
    --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 \
    --window_size 512 \
    --memory_layer_idx 6 --mem_top_m 64 --ocr_dim 256 \
    --total_seq_len 2048 --chunk_size 512 \
    --batch_size 4 \
    --lr 1e-4 --warmup_steps 200 \
    --epochs 20 \
    --max_docs 5000 \
    --synth_ratio 0.10 --synth_ratio_end 0.05 \
    --niah_ratio 0.30 --niah_ratio_end 0.20 \
    --niah_lm_w 3.0 --niah_decode_w 5.0 \
    --niah_batch_size 4 \
    --niah_num_needles 3 \
    --niah_distractor_docs 200 \
    --save_every 1000 \
    --log_interval 20 \
    --resume_from "$RESUME_CKPT" \
    --seed 42 \
    --n_gpus 1 \
    2>&1 | tee "$OUTDIR/rft_lm/v5_train.log"

echo ""
echo "================================================================"
echo "  v5 TRAINING COMPLETE"
echo "================================================================"
echo "  Checkpoint: $OUTDIR/rft_lm/best_model.pt"
echo ""
echo "  Next: evaluate with run_eval_v4b.sh (update RFT_CKPT path)"
echo "  Or run quick sanity check:"
echo "    python niah_sanity.py \\"
echo "      --rft_ckpt $OUTDIR/rft_lm/best_model.pt \\"
echo "      --baseline_ckpt $ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt \\"
echo "      --tokenizer_path $TOK \\"
echo "      --distractor_path $DATA \\"
echo "      --seq_len 1024 --depth 0.5 --n_trials 20 --chunk_size 512"
echo "================================================================"

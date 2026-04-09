#!/bin/bash
# run_train_v6.sh — Push NIAH accuracy from 56% toward 80%+.
#
# v5 achieved 54-61% at depths 0.0-0.5 with the corrected embed alignment.
# v6 targets 80%+ via three changes:
#
#   (a) mem_gate_alpha init 1.0 → tanh(1.0) ≈ 0.76 memory signal
#       (v5 had tanh(0.1) ≈ 0.10 — memory was only 10% of residual)
#   (b) Longer training: 40 epochs (v5 was 20, lm_acc still climbing)
#   (c) Higher embed decode weight: 10.0 (v5 was 5.0)
#
# Resumes from v3 checkpoint (clean base, no buggy weights).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_train_v6.sh

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RESUME_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTDIR="$ROOT/mixed_pilot_v6_seed42"

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
echo "  RFT-LM Pilot v6 — Push to 80%+ NIAH"
echo "================================================================"
echo "  Resume from: $RESUME_CKPT"
echo "  Output:      $OUTDIR"
echo "  GPU:         $CUDA_VISIBLE_DEVICES"
echo ""
echo "  Changes from v5:"
echo "    mem_gate_alpha: 0.1 → 1.0 (tanh: 0.10 → 0.76)"
echo "    epochs:         20  → 40"
echo "    niah_decode_w:  5.0 → 10.0"
echo "  Target: 80%+ at depths 0.0-0.5"
echo "================================================================"
echo ""

# ── Training ──────────────────────────────────────────────────────────
mkdir -p "$OUTDIR/rft_lm"

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
    --epochs 40 \
    --max_docs 5000 \
    --synth_ratio 0.10 --synth_ratio_end 0.05 \
    --niah_ratio 0.30 --niah_ratio_end 0.20 \
    --niah_lm_w 3.0 --niah_decode_w 10.0 \
    --niah_batch_size 4 \
    --niah_num_needles 3 \
    --niah_distractor_docs 200 \
    --save_every 1000 \
    --log_interval 20 \
    --resume_from "$RESUME_CKPT" \
    --mem_gate_alpha_init 1.0 \
    --seed 42 \
    --n_gpus 1 \
    2>&1 | tee "$OUTDIR/rft_lm/v6_train.log"

echo ""
echo "================================================================"
echo "  v6 TRAINING COMPLETE"
echo "================================================================"
echo "  Checkpoint: $OUTDIR/rft_lm/best_model.pt"
echo ""
echo "  Quick sanity check:"
echo "    python niah_sanity.py \\"
echo "      --rft_ckpt $OUTDIR/rft_lm/best_model.pt \\"
echo "      --baseline_ckpt $ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt \\"
echo "      --tokenizer_path $TOK \\"
echo "      --distractor_path $DATA \\"
echo "      --seq_len 1024 --depth 0.5 --n_trials 20 --chunk_size 512"
echo ""
echo "  Full eval:"
echo "    # Update run_eval_v5.sh to point to v6 checkpoint, then run"
echo "================================================================"

#!/bin/bash
# run_train_v6_seed44.sh — Multi-seed Gate 1C, seed 44 of v6 recipe.
#
# Identical recipe to run_train_v6.sh except --seed 44 and a new outdir.
# Designed to run on L40S in parallel with seed 43 on V100.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_train_v6_seed44.sh \
#       > /tmp/v6_seed44.out 2>&1 &

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RESUME_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTDIR="$ROOT/mixed_pilot_v6_seed44"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: v3 checkpoint not found: $RESUME_CKPT"
    exit 1
fi

echo "================================================================"
echo "  RFT-LM v6 — Multi-seed Gate 1C, SEED 44"
echo "================================================================"
echo "  Same recipe as v6 seed 42, only --seed differs."
echo "  Output: $OUTDIR"
echo "================================================================"

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
    --seed 44 \
    --n_gpus 1 \
    2>&1 | tee "$OUTDIR/rft_lm/v6_seed44_train.log"

echo ""
echo "================================================================"
echo "  v6 seed 44 training complete"
echo "  Checkpoint: $OUTDIR/rft_lm/best_model.pt"
echo "================================================================"

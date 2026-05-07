#!/bin/bash
# run_graft_full.sh — Full training run for the Qwen2.5-0.5B graft.
# ~6-10 hours on 1x V100.

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
DATA="$ROOT/data/c4_train.jsonl"
OUT="$ROOT/graft_qwen05b_v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"

echo "================================================================"
echo "  GRAFT v1 — Qwen2.5-0.5B + RFT memory operator (8000 steps)"
echo "================================================================"

python "$SCRIPT_DIR/train_graft.py" \
    --base Qwen/Qwen2.5-0.5B \
    --data_path "$DATA" \
    --outdir "$OUT" \
    --memory_layer_idx 12 \
    --mem_top_m 64 --ocr_dim 256 \
    --mem_gate_alpha_init 1.0 \
    --chunk_size 512 --batch_size 2 \
    --steps 8000 --warmup 300 --lr 1e-4 \
    --niah_value_type numbers --niah_max_val_tokens 8 \
    --niah_num_needles 3 \
    --niah_distractor_docs 300 \
    --niah_lm_w 3.0 --niah_decode_w 10.0 \
    --niah_router_w 1.0 --niah_topm_w 0.25 \
    --niah_pointer_w 0.5 --niah_ocr_w 0.2 \
    --save_every 1000 --log_every 50 \
    --seed 42 \
    2>&1 | tee "$OUT/train.log"

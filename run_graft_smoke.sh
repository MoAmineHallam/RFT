#!/bin/bash
# run_graft_smoke.sh — 1000-step smoke test for the Qwen graft.
# Confirms pipeline correctness in ~30 minutes on 1× V100.
#
# Pass criterion: niah_lm_full_match should climb from 0 toward 0.3+ by step 1000.

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
DATA="$ROOT/data/c4_train.jsonl"
OUT="$ROOT/graft_qwen05b_smoke"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"

echo "================================================================"
echo "  GRAFT SMOKE — Qwen2.5-0.5B + RFT memory operator (1000 steps)"
echo "  Value type: digit2 (1-2 BPE tokens) — achievable full_match in 1k steps"
echo "  Pass criterion: niah_lm_full_match > 0.10 by step 1000"
echo "================================================================"

python "$SCRIPT_DIR/train_graft.py" \
    --base Qwen/Qwen2.5-0.5B \
    --data_path "$DATA" \
    --outdir "$OUT" \
    --memory_layer_idx 23 --unfreeze_from_layer 24 \
    --mem_top_m 64 --ocr_dim 256 \
    --mem_gate_alpha_init 1.0 \
    --chunk_size 512 --batch_size 2 \
    --steps 1000 --warmup 100 --lr 1e-4 \
    --niah_value_type digit2 --niah_max_val_tokens 4 \
    --niah_num_needles 1 \
    --niah_distractor_docs 100 \
    --niah_lm_w 5.0 --niah_decode_w 0.0 \
    --niah_router_w 1.0 --niah_topm_w 0.25 \
    --niah_pointer_w 0.5 --niah_ocr_w 0.2 \
    --save_every 500 --log_every 20 \
    --seed 42 \
    2>&1 | tee "$OUT/smoke.log"

echo ""
echo "Smoke training done. Run quick eval:"
echo "  bash run_graft_eval_smoke.sh"

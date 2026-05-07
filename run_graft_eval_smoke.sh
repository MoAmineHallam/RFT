#!/bin/bash
# run_graft_eval_smoke.sh — 50-trial eval of the smoke checkpoint.

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
CKPT="$ROOT/graft_qwen05b_smoke/best_model.pt"
DIST="$ROOT/data/pg_essays.jsonl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Run run_graft_smoke.sh first."
    exit 1
fi

for VAL in numbers short_int digit2; do
    echo ""
    echo "=== graft smoke eval, value_type=$VAL ==="
    python "$SCRIPT_DIR/eval_ruler_niah.py" \
        --graft_ckpt "$CKPT" \
        --tokenizer_path Qwen/Qwen2.5-0.5B \
        --distractor_path "$DIST" \
        --seq_lens 1024 --needle_depth 0.5 \
        --value_type "$VAL" \
        --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
        --prompt_style instruct --tail_chunk_len 8 \
        --n_trials 50 --max_new_tokens 16 \
        --outfile "graft_smoke_${VAL}_L1024.json" \
        2>&1 | tee "$ROOT/graft_qwen05b_smoke/eval_${VAL}.log"
done

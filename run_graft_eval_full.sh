#!/bin/bash
# run_graft_eval_full.sh — Full eval grid: 3 value types x 3 lengths x depth grid.
# Plus vanilla Qwen baseline for the same cells.

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
CKPT="$ROOT/graft_qwen05b_v1/best_model.pt"
DIST="$ROOT/data/pg_essays.jsonl"
OUT="$ROOT/graft_qwen05b_v1/eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Run run_graft_full.sh first."
    exit 1
fi

# 1) Graft eval grid
for VAL in numbers short_int digit2; do
    for SEQ in 1024 2048 4096; do
        echo ""
        echo "=== graft eval val=$VAL seq=$SEQ ==="
        python "$SCRIPT_DIR/eval_ruler_niah.py" \
            --graft_ckpt "$CKPT" \
            --tokenizer_path Qwen/Qwen2.5-0.5B \
            --distractor_path "$DIST" \
            --seq_lens "$SEQ" \
            --needle_depth_grid 0.0 0.25 0.5 0.75 1.0 \
            --value_type "$VAL" \
            --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
            --prompt_style instruct --tail_chunk_len 8 \
            --n_trials 100 --max_new_tokens 16 \
            --outfile "$OUT/graft_${VAL}_L${SEQ}.json" \
            2>&1 | tee "$OUT/graft_${VAL}_L${SEQ}.log"
    done
done

# 2) Vanilla Qwen baseline (same cells)
for VAL in numbers short_int digit2; do
    for SEQ in 1024 2048 4096; do
        echo ""
        echo "=== vanilla Qwen2.5-0.5B val=$VAL seq=$SEQ ==="
        python "$SCRIPT_DIR/eval_vanilla_hf.py" \
            --base Qwen/Qwen2.5-0.5B \
            --distractor_path "$DIST" \
            --seq_lens "$SEQ" \
            --needle_depth_grid 0.0 0.25 0.5 0.75 1.0 \
            --value_type "$VAL" \
            --mk_num_keys 3 --mk_num_values 1 --mk_num_queries 1 \
            --prompt_style instruct \
            --n_trials 100 --max_new_tokens 16 \
            --outfile "$OUT/vanilla_${VAL}_L${SEQ}.json" \
            2>&1 | tee "$OUT/vanilla_${VAL}_L${SEQ}.log"
    done
done

echo ""
echo "Done. Compare graft_*.json vs vanilla_*.json files in:"
echo "  $OUT"

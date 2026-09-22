#!/bin/bash
# run_graft_attempt2.sh — THE time-boxed graft attempt (G0 retry, one shot).
#
# Why the first smoke failed (diagnosis):
#   Memory was injected AFTER the last layer (idx=23), so the only transform
#   between mem_ctx and the logits was RMSNorm + tied lm_head. The frozen
#   base's hidden state dominates the residual sum, and nothing trainable
#   exists downstream to route/amplify the memory signal. Retrieval was
#   perfect (r@M=1.0) and mem_ctx was decodable in isolation (emb=1.0),
#   but token-0 at the probe never cracked — the exact memory-to-output gap.
#
# This attempt mirrors the geometry of the 125M recipe that reached 96%:
#   - inject at 75% depth (idx=17 of 24) → SIX transformer blocks after it
#   - unfreeze those six blocks (18-23) so they can learn to read mem_ctx
#   - alignment loss ON with v6 weights (lm_w=3, decode_w=10, gate=1.0)
#   - digit2 values, 1 needle → token-0/full-match observable within 2k steps
#
# Pass criterion: tok0 > 0.5 AND full > 0.3 by step 2000.
# On fail: PARK the graft (limitation/future-work), 125M paper proceeds.

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
DATA="$ROOT/data/c4_train.jsonl"
DIST="$ROOT/data/pg_essays.jsonl"
OUT="$ROOT/graft_qwen05b_attempt2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"

echo "================================================================"
echo "  GRAFT ATTEMPT 2 — inject idx=17, unfreeze 18-23, alignment ON"
echo "  Pass: tok0 > 0.5 and full > 0.3 by step 2000. On fail: park."
echo "================================================================"

python "$SCRIPT_DIR/train_graft.py" \
    --base Qwen/Qwen2.5-0.5B \
    --data_path "$DATA" \
    --outdir "$OUT" \
    --memory_layer_idx 17 --unfreeze_from_layer 18 \
    --mem_top_m 64 --ocr_dim 256 \
    --mem_gate_alpha_init 1.0 \
    --chunk_size 512 --batch_size 2 \
    --steps 2000 --warmup 100 --lr 1e-4 \
    --niah_value_type digit2 --niah_max_val_tokens 4 \
    --niah_num_needles 1 \
    --niah_distractor_docs 100 \
    --niah_lm_w 3.0 --niah_decode_w 10.0 \
    --niah_router_w 1.0 --niah_topm_w 0.25 \
    --niah_pointer_w 0.5 --niah_ocr_w 0.2 \
    --save_every 1000 --log_every 20 \
    --seed 42 \
    2>&1 | tee "$OUT/attempt2.log"

echo ""
echo "=== attempt2 eval: digit2 @ L=1024 d=0.5 (the one meaningful cell) ==="
python "$SCRIPT_DIR/eval_ruler_niah.py" \
    --graft_ckpt "$OUT/best_model.pt" \
    --tokenizer_path Qwen/Qwen2.5-0.5B \
    --distractor_path "$DIST" \
    --seq_lens 1024 \
    --needle_depth_grid 0.5 \
    --value_type digit2 \
    --mk_num_keys 1 --mk_num_values 1 --mk_num_queries 1 \
    --prompt_style continuation --tail_chunk_len 8 \
    --n_trials 50 --max_new_tokens 8 \
    --outfile "$OUT/attempt2_digit2_L1024.json" \
    2>&1 | tee "$OUT/attempt2_eval.log"

echo ""
echo "================================================================"
echo "  ATTEMPT 2 DONE. Decision rule:"
echo "  - eval digit2 >= 30%  → graft is ALIVE → scale up (full run)"
echo "  - eval digit2 <  30%  → PARK the graft, finish the 125M paper"
echo "================================================================"

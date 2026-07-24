#!/bin/bash
# run_train_v6_noalign.sh — THE linchpin ablation.
#
# Identical to the v6 recipe (run_train_v6_seed43.sh) in every respect
# EXCEPT --niah_decode_w 0.0 (alignment loss OFF), seed 42, new outdir.
#
# Isolates what the embedding-alignment loss contributes ON TOP of already
# supervised, correct retrieval (router/topm/pointer give r@M -> 1.0 here,
# unlike the MT baseline which has no retrieval supervision).
#
# Decision this run settles:
#   - eval collapses (<< v6's 96%)  -> alignment is the unlock given retrieval
#   - eval stays ~96%               -> supervised retrieval is the unlock,
#                                       alignment is optional
#
# Compare against v6 seed 42 (the 96% headline), same seed for a clean pair.

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RESUME_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTDIR="$ROOT/mixed_pilot_v6_noalign_seed42"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: v3 checkpoint not found: $RESUME_CKPT"
    exit 1
fi

echo "================================================================"
echo "  RFT-LM v6 — ALIGNMENT ABLATION (decode_w=0), SEED 42"
echo "  Same recipe as v6 seed 42, only --niah_decode_w differs (10 -> 0)."
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
    --niah_lm_w 3.0 --niah_decode_w 0.0 \
    --niah_batch_size 4 \
    --niah_num_needles 3 \
    --niah_distractor_docs 200 \
    --save_every 1000 \
    --log_interval 20 \
    --resume_from "$RESUME_CKPT" \
    --mem_gate_alpha_init 1.0 \
    --seed 42 \
    --n_gpus 1 \
    2>&1 | tee "$OUTDIR/rft_lm/v6_noalign_train.log"

echo ""
echo "================================================================"
echo "  v6 no-align training complete"
echo "  Checkpoint: $OUTDIR/rft_lm/best_model.pt"
echo "  Next: eval on RULER S-NIAH and compare to v6 seed 42 (96%)."
echo "================================================================"

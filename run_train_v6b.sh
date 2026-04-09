#!/bin/bash
# run_train_v6b.sh — Parallel experiment: make OCR earn its keep.
#
# Same as v6 (high gate alpha, longer training) but with:
#   - mem_top_m = 256 (v6: 64) — more candidates = harder disambiguation
#   - OCR should matter when there are 256 candidates to sort through
#
# Run on GPU 1 while v6 runs on GPU 0:
#   CUDA_VISIBLE_DEVICES=0 bash run_train_v6.sh &
#   CUDA_VISIBLE_DEVICES=1 bash run_train_v6b.sh &
#   wait

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
RESUME_CKPT="$ROOT/mixed_pilot_v3_seed42/rft_lm/best_model.pt"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTDIR="$ROOT/mixed_pilot_v6b_m256_seed42"

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
echo "  RFT-LM Pilot v6b — M=256 (OCR disambiguation test)"
echo "================================================================"
echo "  Resume from: $RESUME_CKPT"
echo "  Output:      $OUTDIR"
echo "  GPU:         $CUDA_VISIBLE_DEVICES"
echo ""
echo "  Same as v6 except:"
echo "    mem_top_m:  64  → 256 (4x more candidates to disambiguate)"
echo "  Hypothesis: OCR will help when M is large enough"
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
    --memory_layer_idx 6 --mem_top_m 256 --ocr_dim 256 \
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
    2>&1 | tee "$OUTDIR/rft_lm/v6b_train.log"

echo ""
echo "================================================================"
echo "  v6b TRAINING COMPLETE"
echo "================================================================"
echo "  Checkpoint: $OUTDIR/rft_lm/best_model.pt"
echo ""
echo "  Compare v6 (M=64) vs v6b (M=256):"
echo "    - If v6b_full > v6b_no_ocr: OCR helps with more candidates"
echo "    - If v6b ≈ v6: M=64 was enough, larger M adds noise"
echo "    - If v6b > v6: more candidates = better retrieval overall"
echo "================================================================"

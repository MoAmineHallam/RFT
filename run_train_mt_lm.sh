#!/bin/bash
# run_train_mt_lm.sh — Gate 1A: Memorizing-Transformers kNN baseline,
#                     with and without embedding alignment loss.
#
# Two variants, identical except for --niah_decode_w:
#   VARIANT=align    → niah_decode_w=10.0  (MT + alignment loss)
#   VARIANT=noalign  → niah_decode_w=0.0   (vanilla MT baseline)
#
# Trains from SCRATCH (no resume) so the alignment loss is the only
# training-time variable separating the two runs.
#
# Pass criterion (from PAPER_PLAN.md Gate 1A):
#   align variant lifts kNN by ≥20pp at d=0.0-0.5 over the noalign baseline.
#
# Usage:
#   VARIANT=noalign CUDA_VISIBLE_DEVICES=1 bash run_train_mt_lm.sh
#   VARIANT=align   CUDA_VISIBLE_DEVICES=0 bash run_train_mt_lm.sh
#
# Run both in parallel on two GPUs.

set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

VARIANT="${VARIANT:?set VARIANT=align or VARIANT=noalign}"

case "$VARIANT" in
    align)
        NIAH_DECODE_W=10.0
        TAG="mt_lm_align"
        ;;
    noalign)
        NIAH_DECODE_W=0.0
        TAG="mt_lm_noalign"
        ;;
    *)
        echo "ERROR: VARIANT must be 'align' or 'noalign' (got $VARIANT)"; exit 1
        ;;
esac

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTDIR="$ROOT/${TAG}_seed42"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$DATA" ]; then
    echo "ERROR: training data not found: $DATA"; exit 1
fi

echo "================================================================"
echo "  Memorizing Transformers (MTLM) — Gate 1A"
echo "================================================================"
echo "  Variant:       $VARIANT"
echo "  niah_decode_w: $NIAH_DECODE_W"
echo "  Output:        $OUTDIR"
echo "  GPU:           $CUDA_VISIBLE_DEVICES"
echo ""
echo "  MT design (minimal, Wu et al. 2022):"
echo "    - Top-K kNN over past (k,v) hidden states"
echo "    - No recency bias, no OCR, no supervised router"
echo "    - Gated residual add via tanh(mem_gate_alpha)"
echo ""
echo "  Gate 1A pass criterion:"
echo "    align variant must lift d=0.0-0.5 accuracy by ≥20pp over noalign."
echo "================================================================"
echo ""

mkdir -p "$OUTDIR/mt_lm"

python "$SCRIPT_DIR/train_overnight.py" \
    --data_path "$DATA" \
    --outdir "$OUTDIR" \
    --model mt_lm \
    --tokenizer_path "$TOK" \
    --vocab_size 50257 \
    --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 \
    --window_size 512 \
    --memory_layer_idx 6 --mem_top_m 64 \
    --total_seq_len 2048 --chunk_size 512 \
    --batch_size 4 \
    --lr 1e-4 --warmup_steps 200 \
    --epochs 40 \
    --max_docs 5000 \
    --synth_ratio 0.00 --synth_ratio_end 0.00 \
    --niah_ratio 0.30 --niah_ratio_end 0.20 \
    --niah_lm_w 3.0 --niah_decode_w "$NIAH_DECODE_W" \
    --niah_batch_size 4 \
    --niah_num_needles 3 \
    --niah_distractor_docs 200 \
    --save_every 1000 \
    --log_interval 20 \
    --mem_gate_alpha_init 1.0 \
    --seed 42 \
    --n_gpus 1 \
    2>&1 | tee "$OUTDIR/mt_lm/${TAG}_train.log"

echo ""
echo "================================================================"
echo "  $TAG TRAINING COMPLETE"
echo "================================================================"
echo "  Checkpoint: $OUTDIR/mt_lm/best_model.pt"
echo ""
echo "  After BOTH variants are done, eval each on RULER S-NIAH and"
echo "  compare d=0.0-0.5 accuracy. Gate 1A passes if align - noalign ≥ 20pp."
echo "================================================================"

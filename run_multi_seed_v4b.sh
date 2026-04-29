#!/bin/bash
# run_multi_seed_v4b.sh — Paper-quality multi-seed matched retraining.
#
# Produces: 3 seeds × 4 variants = 12 runs, each with identical hyperparams.
# Each run has TWO phases:
#   Phase 1 (v3-style): C4 + synthetic retrieval, ~5800 steps
#   Phase 2 (v4b-style): Resume with NIAH curriculum, ~6000 steps
#
# Variants:
#   rft_lm              — full model (memory + OCR + emb alignment)
#   rft_lm_disable_ocr  — memory but no OCR disambiguation
#   rft_lm_disable_memory — architecture present but memory disabled
#   baseline            — vanilla transformer, no memory at all
#
# The disable_memory and baseline variants skip Phase 2 (no NIAH training)
# since they can't do cross-chunk retrieval anyway.
#
# Resource estimate per run:
#   Phase 1: ~5800 steps × ~0.5s/step ≈ 50 min (1 GPU)
#   Phase 2: ~6000 steps × ~0.6s/step ≈ 60 min (1 GPU)
#   Total: ~22 hours for all 12 runs on 1 GPU, or ~7.5h on 3 GPUs in parallel.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 bash run_multi_seed_v4b.sh
#   # Seeds run sequentially; variants within a seed run in parallel across GPUs.

set -euo pipefail

export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────
ROOT="/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
DATA="$ROOT/data/c4_train.jsonl"
TOK="$ROOT/gpt2_tokenizer"
OUTROOT="$ROOT/multi_seed_v4b_$(date +%Y%m%d)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=(42 43 44)
VARIANTS=(rft_lm rft_lm_disable_ocr rft_lm_disable_memory baseline)
# Variants that get Phase 2 (NIAH curriculum)
NIAH_VARIANTS=(rft_lm rft_lm_disable_ocr)

# ── Common hyperparams (match pilot v3 → v4b exactly) ────────────────
PHASE1_EPOCHS=20
PHASE1_MAX_DOCS=5000
PHASE1_LR=3e-4
PHASE1_BATCH=4
PHASE1_SEQ=2048
PHASE1_CHUNK=512
PHASE1_WARMUP=200
PHASE1_SYNTH=0.30
PHASE1_SYNTH_END=0.10
PHASE1_SAVE_EVERY=2000

PHASE2_EPOCHS=20
PHASE2_LR=1e-4
PHASE2_BATCH=4
PHASE2_WARMUP=200
PHASE2_SYNTH=0.10
PHASE2_SYNTH_END=0.05
PHASE2_NIAH=0.30
PHASE2_NIAH_END=0.20
PHASE2_NIAH_LM_W=3.0
PHASE2_NIAH_DECODE_W=5.0
PHASE2_NIAH_NEEDLES=3
PHASE2_SAVE_EVERY=1000

mkdir -p "$OUTROOT"
RUNBOOK="$OUTROOT/runbook.txt"
ERRORS="$OUTROOT/errors.txt"
: > "$ERRORS"

echo "[START] $(date -Is) multi-seed v4b retraining" | tee "$RUNBOOK"
echo "[CONFIG] seeds=${SEEDS[*]}, variants=${VARIANTS[*]}" | tee -a "$RUNBOOK"
echo "[CONFIG] outroot=$OUTROOT" | tee -a "$RUNBOOK"

# ── Training functions ────────────────────────────────────────────────

run_phase1() {
    local SEED=$1
    local VARIANT=$2
    local GPU=$3
    local OUTDIR="$OUTROOT/seed_${SEED}/phase1"

    echo "[PHASE1] seed=$SEED variant=$VARIANT gpu=$GPU start=$(date -Is)" | tee -a "$RUNBOOK"

    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/train_overnight.py" \
        --data_path "$DATA" \
        --outdir "$OUTDIR" \
        --model "$VARIANT" \
        --tokenizer_path "$TOK" \
        --vocab_size 50257 \
        --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 \
        --window_size 512 \
        --memory_layer_idx 6 --mem_top_m 64 --ocr_dim 256 \
        --total_seq_len $PHASE1_SEQ --chunk_size $PHASE1_CHUNK \
        --batch_size $PHASE1_BATCH \
        --lr $PHASE1_LR --warmup_steps $PHASE1_WARMUP \
        --epochs $PHASE1_EPOCHS \
        --max_docs $PHASE1_MAX_DOCS \
        --synth_ratio $PHASE1_SYNTH --synth_ratio_end $PHASE1_SYNTH_END \
        --niah_ratio 0.0 --niah_ratio_end 0.0 \
        --save_every $PHASE1_SAVE_EVERY \
        --seed "$SEED" \
        --n_gpus 1 \
        2>&1 | tee "$OUTDIR/${VARIANT}_phase1.log"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "FAIL phase1 seed=$SEED variant=$VARIANT exit=$EXIT_CODE" | tee -a "$ERRORS" "$RUNBOOK"
    else
        echo "[PHASE1_DONE] seed=$SEED variant=$VARIANT end=$(date -Is)" | tee -a "$RUNBOOK"
    fi
    return $EXIT_CODE
}

run_phase2() {
    local SEED=$1
    local VARIANT=$2
    local GPU=$3
    local PHASE1_CKPT="$OUTROOT/seed_${SEED}/phase1/${VARIANT}/best_model.pt"
    local OUTDIR="$OUTROOT/seed_${SEED}/phase2"

    if [ ! -f "$PHASE1_CKPT" ]; then
        echo "FAIL phase2 seed=$SEED variant=$VARIANT: phase1 checkpoint not found" | tee -a "$ERRORS" "$RUNBOOK"
        return 1
    fi

    echo "[PHASE2] seed=$SEED variant=$VARIANT gpu=$GPU start=$(date -Is)" | tee -a "$RUNBOOK"

    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/train_overnight.py" \
        --data_path "$DATA" \
        --outdir "$OUTDIR" \
        --model "$VARIANT" \
        --tokenizer_path "$TOK" \
        --vocab_size 50257 \
        --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 \
        --window_size 512 \
        --memory_layer_idx 6 --mem_top_m 64 --ocr_dim 256 \
        --total_seq_len $PHASE1_SEQ --chunk_size $PHASE1_CHUNK \
        --batch_size $PHASE2_BATCH \
        --lr $PHASE2_LR --warmup_steps $PHASE2_WARMUP \
        --epochs $PHASE2_EPOCHS \
        --max_docs $PHASE1_MAX_DOCS \
        --synth_ratio $PHASE2_SYNTH --synth_ratio_end $PHASE2_SYNTH_END \
        --niah_ratio $PHASE2_NIAH --niah_ratio_end $PHASE2_NIAH_END \
        --niah_lm_w $PHASE2_NIAH_LM_W --niah_decode_w $PHASE2_NIAH_DECODE_W \
        --niah_num_needles $PHASE2_NIAH_NEEDLES \
        --niah_batch_size 4 \
        --save_every $PHASE2_SAVE_EVERY \
        --resume_from "$PHASE1_CKPT" \
        --seed "$SEED" \
        --n_gpus 1 \
        2>&1 | tee "$OUTDIR/${VARIANT}_phase2.log"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "FAIL phase2 seed=$SEED variant=$VARIANT exit=$EXIT_CODE" | tee -a "$ERRORS" "$RUNBOOK"
    else
        echo "[PHASE2_DONE] seed=$SEED variant=$VARIANT end=$(date -Is)" | tee -a "$RUNBOOK"
    fi
    return $EXIT_CODE
}

# ── Main loop ─────────────────────────────────────────────────────────
# Assign GPUs: get comma-separated list from CUDA_VISIBLE_DEVICES
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=${#GPUS[@]}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  SEED $SEED — Phase 1 (C4 + synthetic)"
    echo "============================================================"

    # Run all 4 variants for this seed
    # If we have multiple GPUs, run variants in parallel
    GPU_IDX=0
    PIDS=()
    for VARIANT in "${VARIANTS[@]}"; do
        GPU=${GPUS[$((GPU_IDX % NUM_GPUS))]}
        run_phase1 "$SEED" "$VARIANT" "$GPU" &
        PIDS+=($!)
        GPU_IDX=$((GPU_IDX + 1))

        # If we've filled all GPUs, wait for current batch
        if [ $((GPU_IDX % NUM_GPUS)) -eq 0 ] && [ $GPU_IDX -lt ${#VARIANTS[@]} ]; then
            for PID in "${PIDS[@]}"; do wait "$PID" || true; done
            PIDS=()
        fi
    done
    # Wait for remaining
    for PID in "${PIDS[@]}"; do wait "$PID" || true; done

    echo ""
    echo "============================================================"
    echo "  SEED $SEED — Phase 2 (NIAH curriculum, rft_lm variants only)"
    echo "============================================================"

    GPU_IDX=0
    PIDS=()
    for VARIANT in "${NIAH_VARIANTS[@]}"; do
        GPU=${GPUS[$((GPU_IDX % NUM_GPUS))]}
        run_phase2 "$SEED" "$VARIANT" "$GPU" &
        PIDS+=($!)
        GPU_IDX=$((GPU_IDX + 1))
    done
    for PID in "${PIDS[@]}"; do wait "$PID" || true; done

    echo "[SEED_DONE] seed=$SEED $(date -Is)" | tee -a "$RUNBOOK"
done

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE"
echo "============================================================"
if [ -s "$ERRORS" ]; then
    echo "  ERRORS:"
    cat "$ERRORS"
else
    echo "  No errors."
fi
echo ""
echo "  Results: $OUTROOT/"
echo "  Next: run RULER S-NIAH eval on all seeds (run_eval_multi_seed.sh)"
echo ""
echo "[ALL_DONE] $(date -Is)" | tee -a "$RUNBOOK"
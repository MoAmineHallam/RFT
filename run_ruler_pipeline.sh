#!/bin/bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_ruler_pipeline.sh \
#     --train_data ./data/c4_train.jsonl \
#     --ruler_data ./data/ruler_sniah.jsonl \
#     --tokenizer ./gpt2_tokenizer \
#     --outdir ./runs_lm/ruler_run

TRAIN_DATA="./data/c4_train.jsonl"
RULER_DATA="./data/ruler_sniah.jsonl"
TOKENIZER="./gpt2_tokenizer"
OUTDIR="./runs_lm/ruler_run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --ruler_data) RULER_DATA="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"

echo "[1/3] Train RFT-LM with OCR-loss + non-zero memory gate init"
python train_overnight.py \
  --data_path "$TRAIN_DATA" \
  --outdir "$OUTDIR" \
  --model rft_lm \
  --tokenizer_path "$TOKENIZER" \
  --vocab_size 50257 \
  --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 --dropout 0.0 \
  --window_size 512 --memory_layer_idx 6 --mem_top_m 64 --ocr_dim 256 \
  --ocr_alpha_init 1e-2 --ocr_margin 0.10 --ocr_loss_weight 0.05 --mem_gate_alpha_init 1.0 \
  --total_seq_len 2048 --chunk_size 512 --batch_size 4 --lr 3e-4 --epochs 2 \
  --warmup_steps 500 --save_every 1000 --log_interval 20 --num_workers 4 --seed 42 --n_gpus 4

echo "[2/3] Train baseline"
python train_overnight.py \
  --data_path "$TRAIN_DATA" \
  --outdir "$OUTDIR" \
  --model baseline \
  --tokenizer_path "$TOKENIZER" \
  --vocab_size 50257 \
  --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 --dropout 0.0 \
  --window_size 512 \
  --total_seq_len 2048 --chunk_size 512 --batch_size 4 --lr 3e-4 --epochs 2 \
  --warmup_steps 500 --save_every 1000 --log_interval 20 --num_workers 4 --seed 42 --n_gpus 4

echo "[3/3] Evaluate on RULER S-NIAH"
python eval_ruler_sniah.py \
  --data_path "$RULER_DATA" \
  --tokenizer_path "$TOKENIZER" \
  --rft_ckpt "$OUTDIR/rft_lm/best_model.pt" \
  --baseline_ckpt "$OUTDIR/baseline/best_model.pt" \
  --chunk_size 512 --max_new_tokens 16 \
  --out_path "$OUTDIR/ruler_sniah_results.json"

echo "Done. Results: $OUTDIR/ruler_sniah_results.json"

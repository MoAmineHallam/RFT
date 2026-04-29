#!/bin/bash
cd /zeng_gk/Amine/Huawei\ Challenge/RFT/AmineHL
export CUDA_VISIBLE_DEVICES=0
exec python train_overnight.py \
    --data_path data/c4_train.jsonl \
    --outdir mixed_pilot_v12_ruler_full_seed42 \
    --model rft_lm \
    --tokenizer_path gpt2_tokenizer \
    --vocab_size 50257 \
    --d_model 768 --n_layers 12 --n_heads 12 --ff_mult 4 \
    --window_size 512 \
    --memory_layer_idx 6 --mem_top_m 64 --ocr_dim 256 \
    --total_seq_len 4096 --chunk_size 512 \
    --batch_size 2 \
    --lr 3e-4 --warmup_steps 1000 \
    --epochs 6 \
    --max_docs 214057 \
    --synth_ratio 0.05 --synth_ratio_end 0.02 \
    --niah_ratio 0.40 --niah_ratio_end 0.30 \
    --niah_lm_w 5.0 --niah_decode_w 20.0 \
    --niah_batch_size 2 \
    --niah_num_needles 3 \
    --niah_distractor_path PaulGrahamEssays.json \
    --niah_distractor_docs 2000 \
    --niah_value_type numbers \
    --resume_from mixed_pilot_v11_ruler_full_seed42/rft_lm/best_model.pt \
    --save_every 5000 \
    --log_interval 100 \
    --mem_gate_alpha_init 1.0 \
    --seed 42 \
    --n_gpus 1

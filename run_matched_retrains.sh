#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/data3/adam_transfer/AmineHL
OUTROOT=/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404
DATA=$ROOT/data/c4_train.jsonl
VAL=$ROOT/data/c4_val.jsonl
TOK=$ROOT/gpt2_tokenizer

mkdir -p "$OUTROOT"
RUNBOOK=$OUTROOT/runbook.txt
ERRORS=$OUTROOT/error_log_summary.txt
: > "$ERRORS"

echo "[START] $(date -Is)" | tee -a "$RUNBOOK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$RUNBOOK"

COMMON_TRAIN=(
  --data_path "$DATA"
  --vocab_size 50257
  --tokenizer_path "$TOK"
  --d_model 768
  --n_layers 12
  --n_heads 12
  --ff_mult 4
  --dropout 0.0
  --window_size 512
  --memory_layer_idx 6
  --mem_top_m 64
  --ocr_dim 256
  --ocr_margin 0.10
  --total_seq_len 2048
  --chunk_size 512
  --batch_size 4
  --lr 3e-4
  --epochs 2
  --warmup_steps 500
  --save_every 1000
  --log_interval 20
  --num_workers 4
  --n_gpus 3
)

run_train() {
  local model=$1
  local seed=$2
  local extra_ocr_weight=$3
  local outdir="$OUTROOT/seed_${seed}"
  mkdir -p "$outdir"
  echo "[TRAIN] model=$model seed=$seed ocr_loss_weight=$extra_ocr_weight outdir=$outdir start=$(date -Is)" | tee -a "$RUNBOOK"
  if ! conda run -n qwen python "$ROOT/train_overnight.py" \
      --outdir "$outdir" \
      --model "$model" \
      --seed "$seed" \
      --ocr_loss_weight "$extra_ocr_weight" \
      "${COMMON_TRAIN[@]}" \
      2>&1 | tee "$outdir/train_${model}_seed${seed}.log"; then
    echo "FAIL train model=$model seed=$seed" | tee -a "$ERRORS" "$RUNBOOK"
    return 1
  fi
  echo "[TRAIN_DONE] model=$model seed=$seed end=$(date -Is)" | tee -a "$RUNBOOK"
}

# Matched seeds and variants
for seed in 42 43 44; do
  run_train baseline "$seed" 0.0
  run_train rft_lm "$seed" 0.05
  run_train rft_lm_disable_memory "$seed" 0.0
  run_train rft_lm_disable_ocr "$seed" 0.0
done

# Post-train eval using existing rigorous evaluator
for seed in 42 43 44; do
  SDIR="$OUTROOT/seed_${seed}"
  EDIR="$SDIR/eval"
  mkdir -p "$EDIR"
  echo "[EVAL] seed=$seed start=$(date -Is)" | tee -a "$RUNBOOK"
  if ! CUDA_VISIBLE_DEVICES=1,2,3 conda run -n qwen python "$ROOT/eval_lm_rigor.py" \
      --val_path "$VAL" \
      --tokenizer_path "$TOK" \
      --rft_ckpt "$SDIR/rft_lm/best_model.pt" \
      --baseline_ckpt "$SDIR/baseline/best_model.pt" \
      --outdir "$EDIR" \
      --seq_lens 2048 4096 8192 16384 \
      --seeds "$seed" \
      --chunk_size 512 \
      --max_docs 2000 \
      --max_seqs 64 \
      2>&1 | tee "$EDIR/eval_seed${seed}.log"; then
    echo "FAIL eval seed=$seed" | tee -a "$ERRORS" "$RUNBOOK"
  fi
  echo "[EVAL_DONE] seed=$seed end=$(date -Is)" | tee -a "$RUNBOOK"
done

# Merge seed eval summaries into one paper summary/table
python - <<'PY'
import csv, json, glob, os, math, statistics
OUTROOT='/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404'
files=sorted(glob.glob(f"{OUTROOT}/seed_*/eval/results_summary.json"))
merged={'source_files':files,'aggregates':{},'per_seed':{}}
for fp in files:
    with open(fp) as f:
        js=json.load(f)
    seed=os.path.basename(os.path.dirname(os.path.dirname(fp))).split('_')[1]
    merged['per_seed'][seed]=js
seqs=['2048','4096','8192','16384']
variants=['baseline','rft_none','rft_disable_memory','rft_disable_ocr']
for s in seqs:
    merged['aggregates'][s]={}
    for v in variants:
        losses=[]; ppls=[]; tps=[]; wall=[]; vram=[]
        for seed,js in merged['per_seed'].items():
            rec=js.get('aggregates',{}).get(s,{}).get(v)
            if rec:
                losses.append(rec['loss_mean']); ppls.append(rec['ppl_mean'])
                tps.append(rec['tokens_per_s_mean']); wall.append(rec['wall_s_mean']); vram.append(rec['peak_vram_alloc_gb_mean'])
        if not losses:
            continue
        def ci(a):
            if len(a)==1: return (a[0],a[0],a[0])
            m=sum(a)/len(a); sd=statistics.stdev(a); d=1.96*sd/math.sqrt(len(a)); return (m,m-d,m+d)
        lm,ll,lh=ci(losses); pm,pl,ph=ci(ppls); tm,tl,th=ci(tps); wm,wl,wh=ci(wall); vm,vl,vh=ci(vram)
        merged['aggregates'][s][v]={
            'n_seeds':len(losses),
            'loss_mean':lm,'loss_ci95':[ll,lh],
            'ppl_mean':pm,'ppl_ci95':[pl,ph],
            'tokens_per_s_mean':tm,'tokens_per_s_ci95':[tl,th],
            'wall_s_mean':wm,'wall_s_ci95':[wl,wh],
            'peak_vram_alloc_gb_mean':vm,'peak_vram_alloc_gb_ci95':[vl,vh],
        }
    b=merged['aggregates'][s].get('baseline')
    if b:
        for v in ['rft_none','rft_disable_memory','rft_disable_ocr']:
            r=merged['aggregates'][s].get(v)
            if r:
                d=r['loss_mean']-b['loss_mean']
                r['delta_vs_baseline_loss_mean']=d
                r['delta_significant_ci_excludes_zero']= (r['loss_ci95'][0] > b['loss_ci95'][1]) or (r['loss_ci95'][1] < b['loss_ci95'][0])
out_json=f"{OUTROOT}/results_summary.json"
with open(out_json,'w') as f: json.dump(merged,f,indent=2)
out_csv=f"{OUTROOT}/results_table.csv"
with open(out_csv,'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['seq_len','variant','n_seeds','loss_mean','loss_ci95_low','loss_ci95_high','ppl_mean','ppl_ci95_low','ppl_ci95_high','delta_loss_vs_baseline_mean','delta_significant','tokens_per_s_mean','wall_s_mean','peak_vram_alloc_gb_mean'])
    for s in seqs:
        for v in variants:
            r=merged['aggregates'].get(s,{}).get(v)
            if not r: continue
            w.writerow([s,v,r['n_seeds'],r['loss_mean'],r['loss_ci95'][0],r['loss_ci95'][1],r['ppl_mean'],r['ppl_ci95'][0],r['ppl_ci95'][1],r.get('delta_vs_baseline_loss_mean'),r.get('delta_significant_ci_excludes_zero'),r['tokens_per_s_mean'],r['wall_s_mean'],r['peak_vram_alloc_gb_mean']])
print(out_json)
print(out_csv)
PY

if [ ! -s "$ERRORS" ]; then echo "none" > "$ERRORS"; fi

echo "[DONE] $(date -Is)" | tee -a "$RUNBOOK"

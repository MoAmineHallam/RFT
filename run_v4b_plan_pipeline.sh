#!/bin/bash
set -euo pipefail

# Implements:
# 1) v4b NIAH sanity (n_trials=20, one seq_len)
# 2) v4b full NIAH (n_trials=200, seq_lens 2048/4096/8192)
# 3) If promising, matched retrains (seeds 42/43/44; baseline + 3 RFT variants)
# 4) Matched post-train evals (NIAH + perplexity) + CI aggregation + go/no-go verdict

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"

V4B_RFT_CKPT="${V4B_RFT_CKPT:-$ROOT/runs_lm/overnight/rft_lm/best_model.pt}"
V4B_BASELINE_CKPT="${V4B_BASELINE_CKPT:-$ROOT/runs_lm/overnight/baseline/best_model.pt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/gpt2_tokenizer}"
TRAIN_DATA="${TRAIN_DATA:-$ROOT/data/c4_train.jsonl}"
VAL_DATA="${VAL_DATA:-$ROOT/data/c4_val.jsonl}"
DISTRACTOR_PATH="${DISTRACTOR_PATH:-$VAL_DATA}"
OUTROOT="${OUTROOT:-$ROOT/runs_lm/v4b_plan_$(date +%Y%m%d_%H%M%S)}"

SANITY_SEQ_LEN="${SANITY_SEQ_LEN:-2048}"
SANITY_N_TRIALS="${SANITY_N_TRIALS:-20}"
FULL_N_TRIALS="${FULL_N_TRIALS:-200}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

SEEDS=(42 43 44)
RFT_VARIANTS=(rft_lm rft_lm_disable_memory rft_lm_disable_ocr)
TRAIN_VARIANTS=(baseline rft_lm rft_lm_disable_memory rft_lm_disable_ocr)
DEPTH_GRID=(0.00 0.25 0.50 0.75 1.00)
SEQ_LENS=(2048 4096 8192)

mkdir -p "$OUTROOT"
RUNBOOK="$OUTROOT/runbook.txt"
ERRORS="$OUTROOT/error_log_summary.txt"
: > "$ERRORS"

run_py() {
  if [ -n "$CONDA_ENV" ]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" | tee -a "$ERRORS" "$RUNBOOK"
    exit 1
  fi
}

move_eval_artifacts() {
  local ckpt="$1"
  local outfile="$2"
  local dest_dir="$3"
  local src_dir
  src_dir="$(run_py - <<PY
from pathlib import Path
print(str(Path("$ckpt").resolve().parent.parent))
PY
)"
  mkdir -p "$dest_dir"
  if [ -f "$src_dir/$outfile" ]; then
    mv "$src_dir/$outfile" "$dest_dir/"
  fi
  local detail="${outfile%.json}_details.json"
  if [ -f "$src_dir/$detail" ]; then
    mv "$src_dir/$detail" "$dest_dir/"
  fi
}

require_file "$V4B_RFT_CKPT"
require_file "$V4B_BASELINE_CKPT"
require_file "$TRAIN_DATA"
require_file "$VAL_DATA"

echo "[START] $(date -Is)" | tee -a "$RUNBOOK"
echo "[ROOT] $ROOT" | tee -a "$RUNBOOK"
echo "[OUTROOT] $OUTROOT" | tee -a "$RUNBOOK"

# ────────────────────────────────────────────────────────────────────────────────
# Step 1/2: v4b sanity then full NIAH
# ────────────────────────────────────────────────────────────────────────────────
PRECHECK_DIR="$OUTROOT/precheck_v4b"
mkdir -p "$PRECHECK_DIR"

echo "[STEP1] v4b sanity NIAH (n_trials=$SANITY_N_TRIALS, seq_len=$SANITY_SEQ_LEN)" | tee -a "$RUNBOOK"
if ! run_py "$ROOT/eval_ruler_niah.py" \
  --rft_ckpt "$V4B_RFT_CKPT" \
  --baseline_ckpt "$V4B_BASELINE_CKPT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --distractor_path "$DISTRACTOR_PATH" \
  --seq_lens "$SANITY_SEQ_LEN" \
  --needle_depth_grid "${DEPTH_GRID[@]}" \
  --n_trials "$SANITY_N_TRIALS" \
  --max_new_tokens 3 \
  --chunk_size 512 \
  --seed 42 \
  --distractor_docs 2000 \
  --prompt_style continuation \
  --value_type digit2 \
  --outfile niah_v4b_sanity.json \
  2>&1 | tee "$PRECHECK_DIR/niah_v4b_sanity.log"; then
  echo "FAIL: v4b sanity niah eval" | tee -a "$ERRORS" "$RUNBOOK"
  exit 1
fi
move_eval_artifacts "$V4B_RFT_CKPT" "niah_v4b_sanity.json" "$PRECHECK_DIR"

echo "[STEP2] v4b full NIAH (n_trials=$FULL_N_TRIALS, seq_lens=${SEQ_LENS[*]})" | tee -a "$RUNBOOK"
if ! run_py "$ROOT/eval_ruler_niah.py" \
  --rft_ckpt "$V4B_RFT_CKPT" \
  --baseline_ckpt "$V4B_BASELINE_CKPT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --distractor_path "$DISTRACTOR_PATH" \
  --seq_lens "${SEQ_LENS[@]}" \
  --needle_depth_grid "${DEPTH_GRID[@]}" \
  --n_trials "$FULL_N_TRIALS" \
  --max_new_tokens 3 \
  --chunk_size 512 \
  --seed 42 \
  --distractor_docs 4000 \
  --prompt_style continuation \
  --value_type digit2 \
  --outfile niah_v4b_full.json \
  2>&1 | tee "$PRECHECK_DIR/niah_v4b_full.log"; then
  echo "FAIL: v4b full niah eval" | tee -a "$ERRORS" "$RUNBOOK"
  exit 1
fi
move_eval_artifacts "$V4B_RFT_CKPT" "niah_v4b_full.json" "$PRECHECK_DIR"

PROMISING="$(run_py - <<PY
import json, os
p = os.path.join("$PRECHECK_DIR", "niah_v4b_full.json")
with open(p, "r", encoding="utf-8") as f:
    js = json.load(f)
acc_r, acc_b = [], []
for seq_map in js.get("results", {}).values():
    for drec in seq_map.values():
        rr = drec.get("rft_lm", {}).get("accuracy")
        bb = drec.get("baseline", {}).get("accuracy")
        if rr is not None and bb is not None:
            acc_r.append(rr); acc_b.append(bb)
if not acc_r:
    print("0")
else:
    mean_r = sum(acc_r) / len(acc_r)
    mean_b = sum(acc_b) / len(acc_b)
    win_rate = sum(1 for r, b in zip(acc_r, acc_b) if r > b) / len(acc_r)
    # "promising" gate: RFT better on mean and wins most cells
    print("1" if (mean_r > mean_b and win_rate >= 0.6) else "0")
PY
)"

if [ "$PROMISING" != "1" ] && [ "$FORCE_RETRAIN" != "1" ]; then
  echo "[STOP] v4b full eval not promising; set FORCE_RETRAIN=1 to continue." | tee -a "$RUNBOOK"
  if [ ! -s "$ERRORS" ]; then echo "none" > "$ERRORS"; fi
  exit 0
fi

TRAIN_ROOT="$OUTROOT/matched_retrains"
mkdir -p "$TRAIN_ROOT"

COMMON_TRAIN=(
  --data_path "$TRAIN_DATA"
  --vocab_size 50257
  --tokenizer_path "$TOKENIZER_PATH"
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
)
if [ -n "${N_GPUS:-}" ]; then
  COMMON_TRAIN+=(--n_gpus "$N_GPUS")
fi
if [ -n "${MAX_DOCS:-}" ]; then
  COMMON_TRAIN+=(--max_docs "$MAX_DOCS")
fi

# ────────────────────────────────────────────────────────────────────────────────
# Step 3/4: matched multi-seed retrains
# ────────────────────────────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
  sdir="$TRAIN_ROOT/seed_${seed}"
  mkdir -p "$sdir"
  for model in "${TRAIN_VARIANTS[@]}"; do
    ocr_w="0.0"
    if [ "$model" = "rft_lm" ]; then
      ocr_w="0.05"
    fi
    echo "[TRAIN] seed=$seed model=$model ocr_loss_weight=$ocr_w" | tee -a "$RUNBOOK"
    if ! run_py "$ROOT/train_overnight.py" \
      --outdir "$sdir" \
      --model "$model" \
      --seed "$seed" \
      --ocr_loss_weight "$ocr_w" \
      "${COMMON_TRAIN[@]}" \
      2>&1 | tee "$sdir/train_${model}_seed${seed}.log"; then
      echo "FAIL train seed=$seed model=$model" | tee -a "$ERRORS" "$RUNBOOK"
    fi
  done
done

# ────────────────────────────────────────────────────────────────────────────────
# Step 5: post-train NIAH + PPL eval on matched runs
# ────────────────────────────────────────────────────────────────────────────────
EVAL_ROOT="$OUTROOT/posttrain_eval"
mkdir -p "$EVAL_ROOT"

for seed in "${SEEDS[@]}"; do
  sdir="$TRAIN_ROOT/seed_${seed}"
  bckpt="$sdir/baseline/best_model.pt"
  if [ ! -f "$bckpt" ]; then
    echo "FAIL missing baseline ckpt seed=$seed: $bckpt" | tee -a "$ERRORS" "$RUNBOOK"
    continue
  fi
  for variant in "${RFT_VARIANTS[@]}"; do
    rckpt="$sdir/$variant/best_model.pt"
    edir="$EVAL_ROOT/seed_${seed}/$variant"
    mkdir -p "$edir"
    if [ ! -f "$rckpt" ]; then
      echo "FAIL missing ckpt seed=$seed variant=$variant: $rckpt" | tee -a "$ERRORS" "$RUNBOOK"
      continue
    fi

    echo "[EVAL_PPL] seed=$seed variant=$variant" | tee -a "$RUNBOOK"
    if ! run_py "$ROOT/eval_perplexity.py" \
      --val_path "$VAL_DATA" \
      --rft_ckpt "$rckpt" \
      --baseline_ckpt "$bckpt" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --seq_lens "${SEQ_LENS[@]}" \
      --chunk_size 512 \
      --batch_size 1 \
      --max_docs 2000 \
      --max_seqs 64 \
      --outfile "ppl_${variant}_seed${seed}.json" \
      2>&1 | tee "$edir/ppl_${variant}_seed${seed}.log"; then
      echo "FAIL ppl eval seed=$seed variant=$variant" | tee -a "$ERRORS" "$RUNBOOK"
    fi
    # eval_perplexity writes to rft_ckpt_dir/../outfile
    src_ppl="$sdir/$variant/../ppl_${variant}_seed${seed}.json"
    if [ -f "$src_ppl" ]; then mv "$src_ppl" "$edir/"; fi

    echo "[EVAL_NIAH] seed=$seed variant=$variant" | tee -a "$RUNBOOK"
    if ! run_py "$ROOT/eval_ruler_niah.py" \
      --rft_ckpt "$rckpt" \
      --baseline_ckpt "$bckpt" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --distractor_path "$DISTRACTOR_PATH" \
      --seq_lens "${SEQ_LENS[@]}" \
      --needle_depth_grid "${DEPTH_GRID[@]}" \
      --n_trials "$FULL_N_TRIALS" \
      --max_new_tokens 3 \
      --chunk_size 512 \
      --seed "$seed" \
      --distractor_docs 4000 \
      --prompt_style continuation \
      --value_type digit2 \
      --outfile "niah_${variant}_seed${seed}.json" \
      2>&1 | tee "$edir/niah_${variant}_seed${seed}.log"; then
      echo "FAIL niah eval seed=$seed variant=$variant" | tee -a "$ERRORS" "$RUNBOOK"
    fi
    move_eval_artifacts "$rckpt" "niah_${variant}_seed${seed}.json" "$edir"
  done
done

# Aggregate with confidence intervals and go/no-go verdict
run_py - <<PY
import glob, json, math, os, statistics
from pathlib import Path

eval_root = Path("$EVAL_ROOT")
out_json = eval_root / "aggregated_summary.json"
out_md = eval_root / "go_no_go.md"
seq_lens = ["2048", "4096", "8192"]
variants = ["baseline", "rft_lm", "rft_lm_disable_memory", "rft_lm_disable_ocr"]
rft_variants = ["rft_lm", "rft_lm_disable_memory", "rft_lm_disable_ocr"]

def ci95(vals):
    if not vals:
        return (None, None, None)
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    m = sum(vals) / len(vals)
    sd = statistics.stdev(vals)
    d = 1.96 * sd / math.sqrt(len(vals))
    return (m, m - d, m + d)

summary = {
    "config": {
        "eval_root": str(eval_root),
        "seq_lens": [int(s) for s in seq_lens],
        "variants": variants,
    },
    "perplexity": {},
    "niah": {},
    "go_no_go": {},
}

# seed-level storage
ppl_seed_variant = {s: {v: [] for v in variants} for s in seq_lens}
niah_seed_variant = {s: {v: [] for v in variants} for s in seq_lens}
delta_seed = {s: [] for s in seq_lens}

# Parse each variant run and collect baseline once per file row
for p in glob.glob(str(eval_root / "seed_*" / "*" / "ppl_*.json")):
    sp = Path(p)
    seed = sp.parent.parent.name.split("_")[1]
    variant = sp.parent.name
    with open(p, "r", encoding="utf-8") as f:
        js = json.load(f)
    for s in seq_lens:
        rec = js.get(s)
        if not rec:
            continue
        rv = rec.get("rft_lm", {}).get("perplexity")
        bv = rec.get("baseline", {}).get("perplexity")
        if rv is not None and variant in ppl_seed_variant[s]:
            ppl_seed_variant[s][variant].append((seed, rv))
        if bv is not None:
            ppl_seed_variant[s]["baseline"].append((seed, bv))

for p in glob.glob(str(eval_root / "seed_*" / "*" / "niah_*.json")):
    sp = Path(p)
    seed = sp.parent.parent.name.split("_")[1]
    variant = sp.parent.name
    with open(p, "r", encoding="utf-8") as f:
        js = json.load(f)
    for s in seq_lens:
        depth_map = js.get("results", {}).get(s, {})
        for _, drec in depth_map.items():
            rv = drec.get("rft_lm", {}).get("accuracy")
            bv = drec.get("baseline", {}).get("accuracy")
            if rv is not None and variant in niah_seed_variant[s]:
                niah_seed_variant[s][variant].append((seed, rv))
            if bv is not None:
                niah_seed_variant[s]["baseline"].append((seed, bv))
            if variant == "rft_lm" and rv is not None and bv is not None:
                delta_seed[s].append(rv - bv)

def dedupe_seed_pairs(pairs):
    by_seed = {}
    for seed, v in pairs:
        by_seed.setdefault(seed, []).append(v)
    return [sum(vs) / len(vs) for _, vs in sorted(by_seed.items())]

for s in seq_lens:
    summary["perplexity"][s] = {}
    summary["niah"][s] = {}
    for v in variants:
        ppl_vals = dedupe_seed_pairs(ppl_seed_variant[s][v])
        niah_vals = dedupe_seed_pairs(niah_seed_variant[s][v])
        pm, pl, ph = ci95(ppl_vals)
        nm, nl, nh = ci95(niah_vals)
        summary["perplexity"][s][v] = {
            "n": len(ppl_vals),
            "mean": pm,
            "ci95": [pl, ph] if pm is not None else None,
        }
        summary["niah"][s][v] = {
            "n": len(niah_vals),
            "mean": nm,
            "ci95": [nl, nh] if nm is not None else None,
        }

robust = True
per_len = {}
for s in seq_lens:
    dm, dl, dh = ci95(delta_seed[s])
    per_len[s] = {"delta_acc_mean": dm, "delta_acc_ci95": [dl, dh] if dm is not None else None}
    if dm is None or dl is None or dl <= 0:
        robust = False

decision = "GO (prioritize paper writing)" if robust else "NO-GO (collect stronger evidence first)"
summary["go_no_go"] = {
    "decision": decision,
    "criterion": "RFT beats baseline robustly across lengths if delta-accuracy CI95 lower bound > 0 for each seq_len",
    "per_seq_len": per_len,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(out_md, "w", encoding="utf-8") as f:
    f.write("# Go/No-Go Decision\n\n")
    f.write(f"**Decision:** {decision}\n\n")
    f.write("## Criterion\n")
    f.write("- RFT vs baseline NIAH delta-accuracy CI95 lower bound must be > 0 at each length (2048/4096/8192).\n\n")
    f.write("## Per-length delta (RFT - baseline)\n")
    for s in seq_lens:
        rec = per_len[s]
        f.write(f"- seq_len={s}: mean={rec['delta_acc_mean']}, ci95={rec['delta_acc_ci95']}\n")

print(out_json)
print(out_md)
PY

if [ ! -s "$ERRORS" ]; then echo "none" > "$ERRORS"; fi
echo "[DONE] $(date -Is)" | tee -a "$RUNBOOK"

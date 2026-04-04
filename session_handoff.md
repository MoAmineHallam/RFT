# Session Handoff — RFT-LM (Current, April 4, 2026)

## TL;DR
We migrated the workflow to this server, fixed paths, added rigorous reproducible evaluation tooling, started matched multi-seed retraining, and pushed code updates to GitHub branch `claude/analyze-repo-improvements-4sYcD`.

Current training is in progress under:
`/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404`

---

## 1) What we are doing right now

We are running a **matched retrain matrix** to test whether RFT-LM is truly better than baseline (architecture-level claim, not checkpoint luck).

### Active run configuration
- GPUs: `CUDA_VISIBLE_DEVICES=1,2,3`
- Seeds: `42, 43, 44`
- Variants per seed:
  - `baseline`
  - `rft_lm` (full, OCR enabled)
  - `rft_lm_disable_memory`
  - `rft_lm_disable_ocr`
- Train script: `train_overnight.py`
- Eval script: `eval_lm_rigor.py`
- Driver: `run_matched_retrains.sh`

### Current progress snapshot
From runbook:
- baseline seed 42 finished
- rft_lm seed 42 running

Artifacts:
- `/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404/runbook.txt`
- `/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404/error_log_summary.txt`

---

## 2) What we changed in this session

### Code changes
1. `train_overnight.py`
- Added model variants:
  - `rft_lm_disable_memory`
  - `rft_lm_disable_ocr`
- Kept `rft_lm` as full OCR-enabled variant.
- Variant-specific behavior:
  - disable memory when requested
  - neutralize OCR contribution + OCR loss where requested

2. `eval_lm_rigor.py` (new)
- Reproducible LM evaluation harness with:
  - fixed seed protocol
  - seq-lens `{2048, 4096, 8192, 16384}`
  - per-variant comparisons
  - efficiency metrics (tokens/s, wall-time, peak VRAM)
  - CIs and delta vs baseline
  - failure analysis extraction (RFT underperform samples)
- Writes standard outputs:
  - `results_summary.json`
  - `results_table.csv`
  - `runbook.txt`
  - `error_log_summary.txt`
  - `verdict.md`

3. `run_matched_retrains.sh` (new)
- End-to-end train+eval orchestrator across seeds and variants
- Appends command ledger and error log
- Merges per-seed eval outputs into aggregate summary/table

4. `eval_ruler_niah.py`
- Includes depth sweep, MK-NIAH, CI, and RFT ablation hooks

5. `eval_perplexity.py`
- Updated in branch sync; kept for direct perplexity workflows

6. `.gitignore`
- Hardened to avoid pushing run artifacts/results logs

### Documentation updates
- Replaced `session_handoff.md` with this up-to-date operational handoff.
- Removed stale `hand_off.md` in repo (deleted intentionally).

---

## 3) What is already validated

### Checkpoint-only rigorous eval (pre-retrain)
Location:
`/data3/adam_transfer/AmineHL/runs_lm/evals_20260404/`

Contained files:
- `results_summary.json`
- `results_table.csv`
- `runbook.txt`
- `error_log_summary.txt`
- `verdict.md`

High-level finding from that checkpoint-only pass:
- RFT-LM did **not** show robust LM advantage over baseline for the existing checkpoints.
- This motivated the matched retrain matrix now running.

---

## 4) Git status / branch

Remote repo: `git@github.com:MoAmineHallam/RFT.git`
Branch: `claude/analyze-repo-improvements-4sYcD`

A push was completed in this session for core tooling/runner updates (commit already on branch). This file supersedes older handoff narratives.

---

## 5) Next actions (operational)

1. Let `run_matched_retrains.sh` finish all seeds/variants.
2. Verify final aggregate outputs under:
- `/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404/results_summary.json`
- `/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404/results_table.csv`
- `/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404/runbook.txt`
- `/data3/adam_transfer/AmineHL/runs_lm/matched_retrains_20260404/error_log_summary.txt`
3. Draft final scientific verdict from retrain-based evidence (not checkpoint-only).

---

## 6) Request: analyze CCM idea

Please analyze the **CCM idea** you proposed, specifically:
- exact mechanism definition in LM context,
- how CCM differs from current RFT memory + OCR,
- complexity/latency impact vs current pipeline,
- failure modes it could fix (and introduce),
- minimal ablation plan to test CCM fairly under same budget,
- whether CCM should replace OCR scoring, augment it, or gate it.

Goal: decide if CCM is a publishable extension or a distraction before implementing.


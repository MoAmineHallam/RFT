# Session Handoff — RFT / RFT-OCR / RFT-LM Repository
## Date: April 2, 2026

## Status Update (April 4, 2026)

This handoff captured the project state before the latest rigorous checkpoint-only LM reevaluation.

Newer reproducible evaluation artifacts now exist under:

- `runs_lm/evals_20260404/results_summary.json`
- `runs_lm/evals_20260404/results_table.csv`
- `runs_lm/evals_20260404/verdict.md`

Under that protocol (fixed seeds, matched tokenizer/data slices, long-context seq-len sweep, ablations, CI), **RFT-LM did not show a robust LM advantage over baseline** for the currently available checkpoints.

Interpret this file as historical context; use the above artifacts for latest quantitative conclusions.
## Purpose
This document is a repository-level handoff for the RFT research project. It summarizes the research goal, architecture, completed work, exact current experimental status, the new 4×4090 server setup, and the strongest results currently in hand. It is written for continuity inside the repo and avoids tasking language aimed at a future assistant.

---

## 1. Project Goal

The project studies **RFT (Sparse Routed Memory / Sparse Routed Long-Context Retrieval)** and its extension **OCR (Occurrence-Contrastive Resolver)** as an explicit memory mechanism for long-context modeling.

The central scientific goal is:

> Build and validate an explicit sparse memory layer for long-context language models, show that the main bottleneck is **pointer disambiguation** rather than routing recall, and show that **OCR** improves retrieval quality and downstream language modeling behavior.

The intended paper framing is not “RFT solves long context generally.” The more defensible framing is:

> **Sparse routing + contrastive disambiguation for efficient, interpretable long-context retrieval**.

This project has two connected tracks:

1. **Synthetic / mechanistic track** — controlled experiments showing where routing fails and how OCR helps.
2. **Language-model track (RFT-LM)** — integration of RFT as a memory layer inside a Transformer LM, trained on real text and evaluated on held-out data and, ultimately, long-context benchmarks.

Earlier project planning explicitly targeted a top conference path through LM-scale evaluation against mechanisms such as Titans, Mamba-2, and DeltaNet on comparable long-context benchmarks. fileciteturn2file0 fileciteturn2file1

---

## 2. High-Level Scientific Story

### Core claim under development
RFT provides **explicit sparse routed retrieval** over stored hidden states. In this setting, routing itself can already achieve very high recall, but selecting the correct item among retrieved candidates remains difficult. OCR is designed to solve this second problem.

### Why OCR matters
OCR is intended to resolve ambiguity among retrieved candidates using contrastive disambiguation features rather than relying only on router scores.

### Why RFT-LM matters
Embedding RFT into a real LM makes the work more than a toy synthetic architecture. The research direction is to position RFT-LM as an interpretable alternative to implicit neural memory systems.

This framing is already reflected in the earlier project handoff and top-conference plan. fileciteturn2file0 fileciteturn2file1

---

## 3. Main Components in the Repository

The repo currently revolves around these core files:

- `RFT_LM.py` — architecture definitions:
  - `RFTLM`
  - `BaselineTransformerLM`
  - `MemoryBank`
  - `train_step_chunked`
- `train_overnight.py` — multi-GPU training script with C4 loading, DDP, checkpointing, and live metric logging.
- `run_overnight.sh` — launcher script that trains RFT-LM and then the baseline sequentially.
- `pilot_3h.sh` — pilot launcher used earlier for sanity checks.
- `eval_perplexity.py` — validation perplexity comparison between RFT-LM and the baseline across multiple sequence lengths. fileciteturn2file0 fileciteturn3file0

There are also earlier synthetic-track artifacts referenced in previous handoff notes, including synthetic experiment scripts and aggregated results. fileciteturn2file0

---

## 4. Architecture Summary

### RFT-LM
RFT-LM is a Transformer decoder with a memory layer inserted at a configurable depth.

The memory mechanism works as follows:
- hidden states from the designated memory layer are projected into memory keys and values,
- these states are stored in a `MemoryBank`,
- later tokens query the bank via sparse routing,
- the router selects top-M candidates,
- OCR re-scores candidates using contrastive features,
- the retrieved context is added back through a gated residual pathway.

### Memory and training design choices
From the current implementation and prior notes:
- gradients do **not** flow through the memory bank,
- chunked processing is used for long sequences,
- each chunk backpropagates immediately to avoid graph accumulation,
- memory gathering was made more efficient through per-batch advanced indexing instead of tensor expansion,
- OCR uses features including recency and relative router/rank signals. fileciteturn2file0 fileciteturn3file0

### Important implementation nuance
Earlier notes contained a mismatch between a byte-level LM description (`vocab_size=256`) and the actual launcher/training scripts. The **actual current LM setup** uses a **GPT-2 tokenizer** at `./gpt2_tokenizer` with `vocab=50257`, both in training and evaluation. This is reflected in the actual launcher and evaluation behavior. fileciteturn2file0 fileciteturn3file0

---

## 5. Synthetic Track Status (Research Context)

The synthetic track predates the new 4×4090 run and remains important because it provides the mechanistic evidence behind OCR.

Earlier verified project notes reported:
- Step 1 OCR ablation complete,
- frozen OCR configuration:
  - `ocr_loss_weight = 0.05`
  - `ocr_margin = 0.10`
  - `residual_dim = 256`
- Step 2 base-side grid complete,
- OCR-side runs largely complete,
- OCR outperforming the base variant across tested conditions,
- ARP negative result: pool residual and sparse attention residual did not help. fileciteturn2file0

Representative previously recorded synthetic gains included roughly +7% to +9% in harder settings, with OCR winning in every listed condition. These are historical notes from the project state before the new server migration and remain part of the project story. fileciteturn2file0

---

## 6. New 4×4090 Server Setup

### Machine context used for the current successful LM run
- **Server IP:** `10.249.42.10`
- **Username:** `Amine`
- **Project path:** `/home/Amine/AmineHL`
- **Conda environment used for the successful run:** `challenge`
- **Visible GPUs for the launcher:** `0,1,2,3`

The run initially failed under the base environment because `torch` was missing. The successful launch used the `challenge` environment instead. The live logs confirm this environment change and the successful 4-GPU DDP run. fileciteturn1file0

### Launcher configuration used on the new server
The corrected launcher used:
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `--batch_size 4`
- `--n_gpus 4`
- `--save_every 1000`
- `--log_interval 20`
- `--num_workers 4`

This configuration had been verified from the edited script before running. fileciteturn0file3

---

## 7. Training Run That Actually Completed on the New Server

The successful new-server LM training used:
- `d_model = 768`
- `n_layers = 12`
- `n_heads = 12`
- `window_size = 512`
- `memory_layer_idx = 6`
- `mem_top_m = 64`
- `ocr_dim = 256`
- `total_seq_len = 2048`
- `chunk_size = 512`
- `batch_size = 4` per GPU across 4 GPUs → global batch size 16
- `lr = 3e-4`
- `epochs = 2`
- `warmup_steps = 500`
- tokenizer: GPT-2 BPE from `./gpt2_tokenizer` with vocab size 50257.

The live run log on the new server showed:
- **steps per epoch:** 7310
- **total steps:** 14620
- **RFT-LM parameter count:** 128,359,685
- peak memory around **7.4 GB** on rank 0,
- throughput stabilizing near **23k tok/s**,
- regular checkpoint saving at 1000-step intervals. fileciteturn1file0

The current evidence also shows that both full training runs completed, because subsequent evaluation loaded:
- `./runs_lm/overnight/rft_lm/best_model.pt`
- `./runs_lm/overnight/baseline/best_model.pt`

and both checkpoints were reported at **step 14620**. The evaluation logs reported:
- `[LOAD] RFT-LM ... (step 14620, loss 4.3957)`
- `[LOAD] Baseline ... (step 14620, loss 4.4045)`

This means the overnight training sequence for both models completed successfully.

---

## 8. Best Available LM Result Right Now: Held-Out C4 Validation Perplexity

A validation perplexity comparison was run on the new server with the following command:

```bash
CUDA_VISIBLE_DEVICES=3 python eval_perplexity.py \
  --val_path ./data/c4_val.jsonl \
  --rft_ckpt ./runs_lm/overnight/rft_lm/best_model.pt \
  --baseline_ckpt ./runs_lm/overnight/baseline/best_model.pt \
  --tokenizer_path ./gpt2_tokenizer \
  --seq_lens 2048 4096 8192 \
  --max_docs 2000 \
  --max_seqs 100 \
  --batch_size 1
```

### Evaluation protocol actually used
From the evaluation script and run log:
- validation file: `./data/c4_val.jsonl`
- documents loaded: **2,000**
- total validation tokens: **944,198**
- sequence lengths evaluated: **2048, 4096, 8192**
- chunk size: **512**
- batch size: **1**
- up to **100 sequences** evaluated at each sequence length.

The evaluation script processes both models in chunked form; for RFT-LM it builds a memory bank across chunks, and for the baseline it processes each chunk without memory. fileciteturn3file0

### Results

| Seq len | RFT-LM loss | RFT-LM PPL | Baseline loss | Baseline PPL | Δ loss (Base - RFT) | Δ PPL (Base - RFT) | Winner |
|--------:|------------:|-----------:|--------------:|-------------:|--------------------:|-------------------:|:-------|
| 2048 | 4.3452 | 77.11 | 4.3538 | 77.77 | +0.0086 | +0.66 | RFT-LM |
| 4096 | 4.3064 | 74.18 | 4.3164 | 74.92 | +0.0100 | +0.75 | RFT-LM |
| 8192 | 4.2951 | 73.34 | 4.3061 | 74.15 | +0.0110 | +0.81 | RFT-LM |

### What this means
The strongest current LM result is:

> On held-out C4 validation text, RFT-LM achieved consistently lower loss and perplexity than the matched baseline at **2048, 4096, and 8192** sequence lengths, with the gap increasing modestly as sequence length increased.

This is more informative than pure training loss. It is still not a full long-context benchmark result, but it is real held-out evidence that the RFT memory layer is not merely fitting training data more aggressively.

### Interpretation boundaries
These results are encouraging but should be described carefully:
- they are **not** yet RULER or BABILong results,
- they are **not** yet matched against external published baselines like Titans on those same tasks,
- the current baseline in evaluation is the trained repo baseline, evaluated chunkwise without memory,
- the gains are consistent but modest.

Still, this is the strongest clean empirical claim currently available inside the repo.

---

## 9. Exact Current Project Claim That Is Now Defensible

Given the synthetic evidence already accumulated and the new held-out C4 validation results, the most defensible current claim is:

> In controlled synthetic settings, OCR consistently improves sparse routed retrieval by addressing pointer disambiguation, and when integrated into a 125M-scale Transformer memory architecture, RFT-LM trains stably and yields modest but consistent held-out perplexity gains over a matched baseline from 2K to 8K context.

That is materially stronger than the earlier purely engineering claim that the model “trains stably on real text.”

---

## 10. Current Limitations

The project still has important limitations.

### LM evaluation limitations
- No RULER results yet in this handoff.
- No BABILong results yet in this handoff.
- No external published-model comparison on matched public benchmarks yet.
- The current evaluation is held-out C4 validation only.

### Training-script limitations
- `train_overnight.py` saves checkpoints but does **not** implement true resume-from-checkpoint logic.
- DDP startup is inefficient because each spawned rank tokenizes the full dataset independently.

### Repo/publication limitations
- If this repo is pushed to GitHub, raw data, checkpoints, logs, and secrets should **not** be committed.
- The private/internal server password should never be included in the repo.

---

## 11. Operational Facts Worth Preserving

### Training outputs
Outputs are written under:

```text
/home/Amine/AmineHL/runs_lm/overnight/
```

Important subdirectories:
- `runs_lm/overnight/rft_lm/`
- `runs_lm/overnight/baseline/`

Important artifacts:
- `best_model.pt`
- `final_model.pt`
- `train_results.json`
- intermediate `checkpoint_step*.pt`
- `train_metrics.jsonl`
- `latest_metrics.json`

### Validation output
The evaluation run reported saving:

```text
./runs_lm/overnight/eval_results.json
```

### Monitoring commands used during active training
Examples used during the successful server session included:

```bash
tail -f ~/AmineHL/overnight_log.txt
```

```bash
watch -n 1 nvidia-smi
```

```bash
watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits && echo && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader'
```

---

## 12. Repo-Safe Reproduction Commands

### Train
```bash
conda activate challenge
cd ~/AmineHL
bash run_overnight.sh 2>&1 | tee overnight_log.txt
```

### Evaluate held-out perplexity
```bash
CUDA_VISIBLE_DEVICES=3 python eval_perplexity.py \
  --val_path ./data/c4_val.jsonl \
  --rft_ckpt ./runs_lm/overnight/rft_lm/best_model.pt \
  --baseline_ckpt ./runs_lm/overnight/baseline/best_model.pt \
  --tokenizer_path ./gpt2_tokenizer \
  --seq_lens 2048 4096 8192 \
  --max_docs 2000 \
  --max_seqs 100 \
  --batch_size 1
```

---

## 13. Research Position of the Project Right Now

The project is no longer only a mechanistic synthetic study. It now has:
- a defined architecture,
- a completed matched RFT-LM vs baseline training run on the new 4×4090 server,
- held-out validation evidence in favor of RFT-LM,
- a coherent long-context paper direction grounded in both mechanism and LM behavior.

The research goal remains to turn this into a strong long-context memory paper by combining:
- synthetic evidence for **why OCR helps**,
- LM evidence that RFT-LM is viable and helpful on held-out text,
- public-benchmark evidence on long-context tasks.

That broader plan was already articulated in the earlier project planning documents and remains the right overarching objective. fileciteturn2file0 fileciteturn2file1

---

## 14. Minimal Summary

RFT/OCR is a sparse long-context retrieval architecture whose key hypothesis is that routing recall is not the main bottleneck; **disambiguation among retrieved candidates is**. OCR addresses that problem. The project already had strong synthetic evidence in favor of OCR. On the new 4×4090 server, a full 125M-scale RFT-LM and matched baseline were successfully trained using GPT-2 tokenization. A held-out C4 validation evaluation then showed **consistent RFT-LM wins at 2K, 4K, and 8K sequence lengths**, with modest gains that increase slightly with context length. This is the strongest current empirical state of the project.

# RFT-LM: Sparse Routed Memory for Long-Context Retrieval at Small Scale

A 125M-parameter decoder-only language model augmented with an external memory
bank and a **supervised sparse router**, trained end-to-end. It solves
needle-in-a-haystack retrieval that an identically-sized baseline cannot do at
all, **without sacrificing language-modeling quality**.

> ⚠️ **Research code, active development.** Results below are from controlled
> runs on the `v7` recipe; multi-seed confidence intervals are still landing.

## Headline results

RULER S-NIAH, 2-digit values, 3 keys, 100 trials/cell, evaluated at
2K / 4K / 8K × 5 needle depths. Model trained at 2K context.

| Model | NIAH (mean) | C4 val perplexity |
|---|---|---|
| **RFT-LM (v7)** | **92.1%** | **120.4** |
| Same checkpoint, memory disabled | **0.0%** | 121.3 |
| Matched baseline (no memory) | 0.0% | 126.6 |

Two things worth noting:

- **Memory is the entire difference.** The same checkpoint with one flag
  toggled goes from 92.1% to exactly 0.0% in every cell.
- **No trade-off.** Perplexity is *better* than the matched baseline, so the
  retrieval capability is not bought with language-modeling quality.
- **Length generalization**: trained at 2K, accuracy holds (and at some depths
  improves) out to 8K.

## What we found that we did not expect

This project's most useful results are negative, and they are reported here
because they saved us weeks and may save others the same.

| Component | Believed | Measured |
|---|---|---|
| Embedding-alignment auxiliary loss | the key contribution | **inert** — ablating it entirely changes nothing (93.5 vs 92.7) |
| Memory gate initialization (0.1 vs 1.0) | critical (7.6× signal) | **inert** (93.7 vs 92.7) |
| Long training (40 epochs) | needed | saturates by epoch ~11 |
| Supervised routing | one ingredient | **necessary** — a kNN memory without it sits at floor (2–4%) |
| Trainable readout | assumed | **necessary** — a frozen decoder with *perfect* retrieval (recall@M = 1.000) still answers **0%** |

**What actually matters is the memory plus a trainable path from it to the
output.** The auxiliary machinery does not.

### A methodology warning

An earlier recipe (5,000 docs × 40 epochs) scored **92% on NIAH while its
validation perplexity was 6453** — a 50× degradation vs baseline, i.e. a
catastrophically broken language model that still aced the benchmark. NIAH
accuracy alone did not detect this. **Report perplexity alongside retrieval
benchmarks**; the fix here was 50,000 docs × 4 epochs at identical compute.

### A stable architectural limit

Needle depth 0.75 scores **71–75% in every configuration we have ever run**
(seven configs, three seeds, both ablations). This is a boundary effect where
the needle straddles a chunk edge — not variance.

## Architecture

- GPT-2 style decoder, 125M params (d_model 768, 12 layers, 12 heads)
- Sliding-window attention (512) for local context
- FIFO external memory bank of past hidden states
- **Sparse top-M router** (M=64) trained with explicit retrieval supervision
  (router CE + top-M hinge + pointer CE)
- Memory injected mid-stack via a gated residual add
- An occurrence-contrastive resolver head (helps +4.8% on synthetic KV
  retrieval; **neutral on natural language**, reported as an honest negative)

## Repository layout

| File | Purpose |
|---|---|
| `RFT_LM.py` | Core architecture: `RFTLM`, `RFTMemoryLayer`, `MemoryBank`, baselines |
| `train_overnight.py` | Mixed-objective training (C4 LM + synthetic retrieval + NIAH) |
| `niah_batch.py` | Natural-language NIAH batch generation and training step |
| `synth_batch.py` | Synthetic KV-retrieval batches |
| `eval_ruler_niah.py` | RULER S-NIAH evaluation: depth grid, multi-key, Wilson CIs, ablations |
| `eval_perplexity.py` | Held-out C4 perplexity vs baseline |
| `diag_ppl_memory.py` | Diagnostic: separates catastrophic forgetting from memory-injection noise |
| `rft_graft.py` / `train_graft.py` | Grafting the memory operator onto a pretrained HF model |
| `run_train_v7_*.sh` | Training recipes (the corrected 50k-doc × 4-epoch runs) |
| `run_graft_*.sh` | Graft training and evaluation |

## Reproducing

```bash
# Train (≈2.2h on one V100)
bash run_train_v7_moredata.sh

# Evaluate NIAH
python eval_ruler_niah.py \
  --rft_ckpt <ckpt> --tokenizer_path gpt2_tokenizer \
  --distractor_path data/pg_essays.jsonl \
  --seq_lens 2048 4096 8192 --needle_depth_grid 0.0 0.25 0.5 0.75 1.0 \
  --value_type digit2 --mk_num_keys 3 \
  --prompt_style continuation --tail_chunk_len 8 \
  --n_trials 100 --outfile results.json

# Check it is still a language model
python diag_ppl_memory.py \
  --val_path data/c4_val.jsonl --tokenizer_path gpt2_tokenizer \
  --ckpt "model=<ckpt>" --seq_len 2048
```

Requires `torch` (CUDA), `transformers`, `tokenizers`. Data: C4 (train/val) and
Paul Graham essays as the NIAH distractor corpus.

## Limitations

- 125M parameters; the approach does **not** transfer to a frozen pretrained
  decoder (see the graft results — perfect retrieval, 0% answers).
- Evaluated on synthetic retrieval (RULER S-NIAH family) only.
- Depth 0.75 remains at ~73%.
- Trained on ~24M tokens, far below compute-optimal for this size.

## Related work

Recent work shows sub-7B models fail to use retrieved **text** 85–100% of the
time even under oracle retrieval ([Pandey 2026](https://arxiv.org/abs/2603.11513)),
and that separating retrieval from utilization requires an explicit protocol
([Four-Condition Diagnostic Protocol](https://arxiv.org/abs/2606.06758)). Our
setting differs: memory is a *learned representation* retrieved by a trained
router, not text placed in context.

## License

Research code released as-is.

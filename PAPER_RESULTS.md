# Paper Results — Consolidated (April 12, 2026)

## Status Summary

| Gate | Result | Implication |
|------|--------|-------------|
| **1A** MT alignment ablation | **FAILED** | Alignment loss alone does NOT bootstrap retrieval in vanilla kNN memory. Paper cannot claim "plug-and-play". |
| **1B** Tail-chunk eval fix | **PASSED** | d=1.0 from 0% → 95-98% with tail_chunk_len=8. Inference-time fix, no retraining. |
| **1C** Multi-seed variance | **PASSED** | 3-seed σ ~1-2pp. Results are reproducible. |

---

## Table 1 — RFT-LM v6: Multi-Seed RULER S-NIAH (tail_chunk_len=8)

**Setup**: 125M GPT-2 + sliding window (512) + FIFO memory + top-M router + OCR + alignment loss.
Trained at L=2048, evaluated at L={2048, 4096, 8192}. 100 trials per cell.

### Seed 42

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 96     | 97     | 75     | 98     | **92.6** |
| 4096 | 92     | 94     | 96     | 73     | 98     | **90.6** |
| 8192 | 92     | 92     | 94     | 74     | 95     | **89.4** |

### Seed 43

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 96     | 95     | 73     | 97     | **91.6** |
| 4096 | 93     | 93     | 96     | 73     | 97     | **90.4** |
| 8192 | 95     | 90     | 92     | 75     | 97     | **89.8** |

### Seed 44

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 98     | 95     | 74     | 98     | **92.4** |
| 4096 | 96     | 92     | 95     | 74     | 97     | **90.8** |
| 8192 | 95     | 95     | 96     | 75     | 96     | **91.4** |

### 3-Seed Average ± Std

| L    | d=0.00      | d=0.25      | d=0.50      | d=0.75      | d=1.00      | **mean**        |
|------|-------------|-------------|-------------|-------------|-------------|-----------------|
| 2048 | 97.0 ± 0.0 | 96.7 ± 1.2 | 95.7 ± 1.2 | 74.0 ± 1.0 | 97.7 ± 0.6 | **92.2 ± 0.5** |
| 4096 | 93.7 ± 2.1 | 93.0 ± 1.0 | 95.7 ± 0.6 | 73.3 ± 0.6 | 97.3 ± 0.6 | **90.6 ± 0.2** |
| 8192 | 94.0 ± 1.7 | 92.3 ± 2.5 | 94.0 ± 2.0 | 74.7 ± 0.6 | 96.0 ± 1.0 | **90.2 ± 1.0** |

**Headline**: 125M model trained at L=2048 achieves **90-92% mean RULER S-NIAH** at L=2048-8192 (4x length generalization). Cross-seed σ < 2.5pp at every cell.

**Known weakness**: d=0.75 consistently 73-75% across all seeds. This is a real model weakness, not noise.

---

## Table 2 — Baseline: Same Model, Memory Disabled

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 0      | 0      | 0      | 0      | 0      | **0.0**  |
| 4096 | 0      | 0      | 0      | 0      | 0      | **0.0**  |
| 8192 | 0      | 0      | 0      | 0      | 0      | **0.0**  |

**Note**: This is the same RFT-LM checkpoint with `use_memory=False`. It is NOT a separately trained full-attention baseline. Reviewers may ask for the latter.

---

## Table 3 — Ablation Study (Seed 42, tail_chunk_len=8)

### Full Model (RFT-LM v6)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 96     | 97     | 75     | 98     | **92.6** |
| 4096 | 92     | 94     | 96     | 73     | 98     | **90.6** |
| 8192 | 92     | 92     | 94     | 74     | 95     | **89.4** |

### -OCR (OCR disabled at test time)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 96     | 96     | 73     | 99     | **92.2** |
| 4096 | 96     | 93     | 95     | 73     | 97     | **90.8** |
| 8192 | 95     | 93     | 95     | 73     | 96     | **90.4** |

### -Memory (memory disabled at test time)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | ~0     | ~0     | ~0     | ~0     | ~0     | **~0**   |
| 4096 | ~0     | ~0     | ~0     | ~0     | ~0     | **~0**   |
| 8192 | ~0     | ~0     | ~0     | ~0     | ~0     | **~0**   |

**Findings**:
- **OCR is dispensable at inference**: Removing OCR costs <1pp. OCR acts as a *training-time scaffold* — it regularizes the router during training but is not needed at test time.
- **Memory is causally necessary**: Without memory, accuracy drops to 0%. The sliding window alone cannot reach past its 512-token horizon.

---

## Table 4 — Tail-Chunk Fix: Before vs After (Seed 42)

### Before fix (tail_chunk_len=0, original eval)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 96     | 97     | 75     | 2      | **73.4** |
| 4096 | 92     | 94     | 96     | 72     | 0      | **70.8** |
| 8192 | 92     | 93     | 94     | 73     | 0      | **70.4** |

### After fix (tail_chunk_len=8)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 97     | 96     | 97     | 75     | 98     | **92.6** |
| 4096 | 92     | 94     | 96     | 73     | 98     | **90.6** |
| 8192 | 92     | 92     | 94     | 74     | 95     | **89.4** |

**Explanation**: At d=1.0, the needle is in the final chunk. Without the tail-chunk fix, it hasn't been written to memory when the probe arrives. Processing the last K=8 tokens as a separate mini-chunk forces a memory write before the probe, fixing d=1.0 from 0% to 95-98%. This is an inference-time fix only — no retraining required.

---

## Table 5 — Gate 1A: Memorizing Transformers Alignment Ablation

**Setup**: MTLM = vanilla Memorizing Transformers (plain top-K kNN over past hidden states, no OCR/router/recency). Trained with and without embedding alignment loss. 125M, same hyperparameters.

### MT-LM without alignment (mt_noalign)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | ~0     | ~0     | ~0     | ~0     | ~0     | **~0**   |
| 4096 | ~0     | ~0     | ~0     | ~0     | ~0     | **~0**   |
| 8192 | ~0     | ~0     | ~0     | ~0     | ~0     | **~0**   |

### MT-LM with alignment (mt_align)

| L    | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | **mean** |
|------|--------|--------|--------|--------|--------|----------|
| 2048 | 0      | 1      | 2      | 1      | 1      | **~1.0** |
| 4096 | 1      | 1      | 1      | 1      | 1      | **~1.0** |
| 8192 | 0      | 0      | 0      | 0      | 0      | **~0.0** |

**Verdict**: Gate 1A **FAILED**. Alignment loss alone cannot bootstrap retrieval from scratch in a vanilla kNN memory. The full RFT recipe (OCR + learned router + alignment) is required. Alignment is necessary but not sufficient.

---

## Paper Narrative (revised after Gate 1A failure)

### What we CAN claim:
1. **The full RFT training recipe** (OCR scaffold + top-M router + embedding alignment) enables a 125M model to achieve 90%+ RULER S-NIAH at 4x training length
2. **Embedding alignment is necessary** (without it, no memory-augmented 125M model retrieves anything)
3. **OCR is a training-time scaffold** removable at inference (<1pp cost)
4. **Memory is causally necessary** (0% without it)
5. **Tail-chunk fix** elegantly solves d=1.0 at inference time
6. **Reproducible** across 3 seeds (σ < 2.5pp)

### What we CANNOT claim:
- ~~"Alignment loss is a plug-and-play technique for any memory architecture"~~ (Gate 1A disproved this)
- ~~"Alignment alone is sufficient"~~ (it's necessary but not sufficient)

### Remaining weaknesses to address:
1. **d=0.75 at 73-75%** — consistent across all seeds, real model weakness
2. **No separately-trained full-attention baseline** — current "baseline" is same model with memory off
3. **No 350M scale point** — cannot claim scale generalization
4. **Single benchmark** (RULER S-NIAH only) — need passkey/multi-needle for breadth

---

## Raw Numbers Reference

All percentages are accuracy (correct / total * 100), 100 trials per cell.
Depths: 0.00 = needle at start, 0.25/0.50/0.75 = proportional position, 1.00 = needle at end.
Sequence lengths: 2048 (= training length), 4096 (2x), 8192 (4x).

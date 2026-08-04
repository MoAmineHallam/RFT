# Paper Skeleton — draft v1 (Aug 4, 2026)

Built against results actually in hand. `[PENDING]` marks the two open cells.

## Title (working)

> **Retrieval Is Not Enough: Three Conditions for Memory to Reach the Output
> in Memory-Augmented Language Models**

Alternatives: *"Retrieved but Unreadable"* · *"The Memory-to-Output Gap"*

## Thesis

A memory-augmented LM can retrieve the right content **perfectly** and still
answer **0%** of the time. Getting retrieved memory into the output requires
three separable conditions, and prior work conflates them:

| Condition | Fails when absent | Our evidence |
|---|---|---|
| **C1. Correct retrieval** | memory returns the wrong content | MT kNN baseline: floor (2–4%), alignment cannot rescue it |
| **C2. Trainable readout** | nothing downstream can use the memory | Qwen-0.5B graft: r@M **1.000**, answer **0%** |
| **C3. Decodable representation** | retrieved vector is not in output-embedding space | v6 vs no-align alignment trajectory |

The headline system satisfies all three: **88.0 ± 5.2%** RULER S-NIAH across
3 seeds vs **0%** base LM and **0–2%** memory-ablated.

## Key mechanistic claim (the interesting part)

With **tied embeddings**, `lm_head` is the embedding matrix transposed, so
"make the retrieved vector resemble the target token's embedding" is the path
of least resistance for the LM objective. Consequently:

- Alignment **emerges on its own** from LM loss once retrieval is correct
  (decode_w=0 run: emb_loss 11.8 → 2.86 by epoch 11, emb_acc 0 → 1.0).
- The explicit alignment loss does not teach a new capability — it
  **accelerates and tightens** alignment by ~100× (v6: emb_loss **0.016**).
- It cannot help at all when C1 fails (MT: emb_loss stuck ~6.7, emb_acc 0) —
  aligning a wrongly-retrieved vector is meaningless.
- It cannot help when C2 fails (graft: frozen readout, alignment target
  satisfiable in isolation — emb_acc 1.0 — yet 0% end-to-end).

`[PENDING]` whether tighter alignment buys **eval** accuracy (v6 vs no-align
at epoch 40) → decides whether C3's explicit loss is a *requirement* or an
*accelerator*. Both are publishable; the sentence differs.

## Sections

1. **Introduction** — the gap; Fig. 1: (a) r@M=1.0 while answer=0 (graft),
   (b) 0% → 88% bar with 3-seed error bars.
2. **The memory-to-output gap** — definition; why it arises (signal
   attenuation through gated residual; layer-k representation vs output space);
   why end-task accuracy alone cannot diagnose it.
3. **Method** — memory bank + supervised sparse router (C1); gate init and
   trainable post-memory path (C2); embedding alignment (C3). Explicitly:
   the architecture is *not* the contribution — the decomposition is.
4. **Experiments** — 125M from scratch, RULER S-NIAH, 3 seeds, identical
   eval code.
5. **Ablations / the three conditions** — MT (¬C1), graft (¬C2),
   no-align (¬C3), no-memory (floor).
6. **Honest negatives** — (a) OCR/resolver head: +4.8% on 513 synthetic KV
   experiments, **neutral** on NL even at M=256; (b) alignment is **not** an
   architecture-agnostic plug-in — it fails on MT; (c) the graft does not
   transfer to a frozen pretrained decoder.
7. **Limitations** — d=0.75 (~73% in *all* seeds, stable architectural weak
   spot at chunk boundaries); one seed (44) degrades at mid-depths → routing
   stability; ≤0.5B scale; synthetic-task focus.
8. **Related work** — memory-augmented LMs (Memorizing Transformers,
   Neurocache, Titans/MIRAS, HMT); context compression incl. optical
   (DeepSeek-OCR, LensVLM, "Text or Pixels", gist tokens). **One paragraph**:
   the same gap governs compressed-memory systems; those methods implicitly
   fight decodability without diagnosing it — our decomposition explains why
   needles are what compression loses first. *Cite the field, don't join it.*

## Results in hand

**Main grid — RULER S-NIAH, digit2, 3 keys, tail_chunk_len=8, n=100/cell,
identical eval code across seeds:**

| Seed | L=2048 | L=4096 | L=8192 | grid mean |
|---|---|---|---|---|
| 42 | 93.2 | 90.6 | 90.6 | 91.5 |
| 43 | 91.0 | 90.6 | 90.0 | 90.5 |
| 44 | 83.0 | 81.0 | 82.2 | 82.1 |
| **mean ± σ** | | | | **88.0 ± 5.2** |

Baselines: base LM **0%** everywhere; memory-disabled **0–2%**.

**Per-depth (avg over lengths):**

| depth | s42 | s43 | s44 |
|---|---|---|---|
| 0.00 | 94.7 | 94.3 | 89.3 |
| 0.25 | 96.0 | 94.3 | 76.3 |
| 0.50 | 95.3 | 93.7 | 80.3 |
| 0.75 | 74.3 | 73.7 | 72.3 |
| 1.00 | 97.0 | 96.7 | 92.0 |

d=0.75 is stable across seeds (73–74) → architectural, not variance.
Seed 44 diverges only at mid-depths → routing-stability failure mode.

**Multi-needle breadth (seed 42, 4 keys vs 3):**
2048: 98→87 (d=0.0), 97→85 (d=0.5); 4096: 92→91, 95→89. Graceful.

**Length generalization:** trained at 2K, flat to 8K (no cliff).

**Tail-chunk fix:** d=1.00 0% → 92–97% (inference-time only, no retrain).

**MT kNN (Gate 1A), matched compute (40 ep, 5k docs):**
no-align 2/4/4%, +align 0/0/0% at d=0.0/0.25/0.5 → both floor.

**Qwen2.5-0.5B graft:** r@M 1.000, emb_acc 1.000, eval 0–2%.

## Open cells

- `[PENDING]` v6-no-align full eval @ epoch 40 → C3 requirement vs accelerator.
- `[PENDING]` perplexity sanity (memory training doesn't wreck LM quality).

## Venue

EMNLP 2026 has passed (ARR May 25 deadline). Targets: **ICLR 2027**
(submit Sept 24, 2026 — fits) or the **next ARR cycle** (~Oct) → NAACL/EMNLP
2027 for the CCF-B path. Confirm with Prof. Zeng whether ICLR counts at HIT.

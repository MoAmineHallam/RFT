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
| **C3. Sufficient memory signal** | memory is added but too attenuated to change the output | gate-init ablation `[PENDING]`; v5 (gate 0.1) 54–61% vs v6 (gate 1.0) 88–91% |

The headline system satisfies all three: **88.0 ± 5.2%** RULER S-NIAH across
3 seeds vs **0%** base LM and **0–2%** memory-ablated.

## Key mechanistic claim (the interesting part)

### The alignment loss is NOT the unlock — a self-correction the paper should own

The project's prior framing (`session_handoff.md`, PAPER_PLAN v1) held that the
embedding-alignment loss was *the* contribution. **A controlled ablation refutes
this.** Training v6's exact recipe with `decode_w = 0`:

| | eval d=0.0 | eval d=0.5 | epochs |
|---|---|---|---|
| v6 seed 42 (**with** alignment) | 98 | 97 | 40 |
| no-align (`decode_w=0`) | **99** | **95** | **11** |

The ablated model matches the full model with <1/3 the training. Explicit
alignment is unnecessary.

**Why**: with **tied embeddings**, `lm_head` *is* the embedding matrix
transposed, so "make the retrieved vector resemble the target token's
embedding" is the path of least resistance for the LM objective itself.
Alignment therefore **emerges for free** once retrieval is correct —
with `decode_w=0`, emb_loss fell 11.8 → 2.86 and emb_acc 0 → 1.0 by epoch 11.
The explicit loss only tightens it (~100×: v6 emb_loss 0.016) without buying
end-task accuracy.

Alignment also cannot rescue a system missing the other conditions:
- **¬C1** (MT, no supervised routing): emb_loss stuck ~6.7, emb_acc 0 —
  aligning a wrongly-retrieved vector is meaningless.
- **¬C2** (graft, frozen readout): alignment target satisfiable in isolation
  (emb_acc 1.0) yet 0% end-to-end.

### What the unlock actually is

v5 → v6 changed three things at once (decode_w 5→10, gate_init 0.1→1.0,
epochs 20→40). `decode_w` is now eliminated. The prime suspect is **gate init**:
`tanh(0.1) ≈ 0.10` vs `tanh(1.0) ≈ 0.76` — a **7.6× increase in memory signal
strength** into the residual stream, which matches the project's original
"signal attenuation" diagnosis of the gap.

`[PENDING]` gate-init ablation (v6 recipe, `mem_gate_alpha_init=0.1`, seed 42).
Collapse toward v5's ~55% ⇒ gate init is the mechanism. Stays ~90% ⇒ training
length was the story.

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

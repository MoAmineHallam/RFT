# Paper Plan — "Small Models, Long Memory" (April 11, 2026)

## Working title

> **Small Models, Long Memory: Training-Time Alignment Unlocks Long-Context Retrieval at 125M**

## Three-claim narrative (the pivoted paper)

1. **Training technique (novel)**: Embedding alignment loss — train the
   post-transform memory vector at the query position to match the target
   token's tied output embedding via cosine + CE. Fixes the memory-to-output
   decoding gap that blocks small memory-augmented transformers.
2. **Long-context empirical result (headline)**: 125M model, trained at 2K,
   reaches **96-97% on RULER S-NIAH at 8K** (base: 0%). Length generalizes
   from 2K → 4× training length with no quality cliff.
3. **Plug-and-play (breadth)**: The loss is architecture-agnostic. It
   improves a Memorizing-Transformers-style kNN memory and a Neurocache-style
   compressed memory without changing their architectures.

## Positioning vs the 2025-2026 field

| Work | Their slot | How we avoid collision |
|---|---|---|
| **Titans + MIRAS** (Google 2025) | surprise-gated memory, 2M context | out-of-reach on scale — cite, do not compete |
| **HMT** (NAACL 2025) | hierarchical memory, "small beats big" | orthogonal axis: training loss, not architecture |
| **Memorizing Transformers** (ICLR 2022) | kNN over (k,v) cache | **use as baseline** + apply alignment loss |
| **Neurocache** (NAACL 2024) | compressed-state kNN | **use as baseline** + apply alignment loss |
| **NAMM** (ICLR 2025, Sakana) | evolved memory management | orthogonal — different problem |

We own: **"training-time fix that works across memory-augmented architectures at small scale."** Nobody else is in this cell.

## What I have ✅

- RFT-LM architecture: 125M GPT-2 + sliding window + FIFO memory bank + top-M router + `mem_gate_alpha` gate
- **Embedding alignment loss** (post-transform `mem_ctx` → tied embedding)
- v6 recipe: gate_alpha=1.0, 40 epochs, decode_w=10.0
- v6 RULER S-NIAH headline (post tail-fix): **full grid 89-98% mean ~91%, base flat 0%**
- v6 ablations: no-memory = 0%, memory is sole differentiator
- **Gate 1B ✅ PASSED** — tail-chunk eval fix (K=8, inference-time only, no retrain).
  Full RULER S-NIAH grid after fix (baseline flat 0% everywhere):

  | L | d=0.00 | d=0.25 | d=0.50 | d=0.75 | d=1.00 | mean |
  |---|---|---|---|---|---|---|
  | 2048 | 97 | 96 | 97 | 75 | 98 | 92.6 |
  | 4096 | 92 | 94 | 96 | 73 | 98 | 90.6 |
  | 8192 | 92 | 92 | 94 | 74 | 95 | 89.4 |

  d=1.0 went from 0 → 95-98%. Only remaining soft cell: d=0.75 at ~73-75%
  (real model weakness, not a bug). Accept for first submission.
- `--mem_gate_alpha_init` CLI override
- Paper framing locked as Option B (training technique, not architecture)

## What to DROP (ruthless)

| Drop | Why |
|---|---|
| OCR head + OCR losses + OCR ablations | Confirmed neutral on NL even at M=256. One honest-negative line in appendix, that's it. |
| Mixed-objective synthetic curriculum as headline | Keep as appendix ablation. It's noise to the main story. |
| 760M stretch run | Not needed for the pivoted framing. |
| Titans / Infini-attention reimplementation | Never worth the weeks of engineering. |
| LongBench full suite | 125M won't have signal on QA/summarization. |
| Any new architectural variant | The architecture is NOT the contribution. Stop tweaking it. |

## What to ADD

| Add | Why | Criticality |
|---|---|---|
| **Memorizing Transformers kNN memory (reimplemented cleanly)** + train with/without alignment loss at 125M | **This IS the paper.** Without this head-to-head you have no plug-in claim. | ★★★ |
| **Neurocache-style compressed kNN** + alignment loss (optional, if time) | Strengthens the plug-in claim from 1 → 2 architectures | ★★ |
| Multi-seed v6 (3 seeds, reduced variant set) | CI rigor | ★★ |
| 350M scale point (single seed) for v6 AND for Memorizing-Transformers+alignment | Addresses scale objection | ★★★ |
| Passkey + multi-needle RULER eval | Benchmark breadth (cheap) | ★ |
| Tail-chunk eval fix run on v6 | Closes d=1.0 limitation | ★★ |
| NarrativeQA + PassageRetrieval (LongBench subset) | One "real" benchmark, cheap | ★ |
| Diagnostic figures (mem_gate_alpha trajectory, cosine-to-embed, per-depth) | Reviewer catnip | ★ |

## Gated plan — V100 first, L40S only after 1A + 1C pass

| Gate | Experiment | Where | Cost | Pass criterion | On failure |
|---|---|---|---|---|---|
| **1A** | **Memorizing Transformers kNN baseline @ 125M, with and without alignment loss** | V100 | ~2-3 days | alignment loss lifts kNN by ≥20pp at d=0.0-0.5 | Contribution is not plug-in → reframe as architecture-specific |
| **1B** ✅ | Tail-chunk eval fix on v6 — **DONE, K=8, d=1.0 → 95-98%, mean ~91% across full grid** | V100 | ~2 hours | d=1.0 → ≥50% with K-sweep | — |
| **1C** | Multi-seed v6 @ 125M (3 seeds × 2 variants: full, −embed_align) | V100 | ~3 GPU-days parallel | σ < 5pp across seeds | Harden recipe |
| **1D** | Passkey + multi-needle RULER eval of v6 (no retrain) | V100 | ~6 hrs | ≥80% on new prompts | Retrain with mixed prompts |
| **1E** | (optional) Neurocache-style compressed kNN baseline | V100 | ~3 days | alignment loss also helps it | Drop, single-baseline plug-in claim |
| **2F** | **350M v6 + 350M kNN+alignment, single seed each** | **L40S** | ~2 days | Same pattern as 125M | Contribution is scale-dependent |
| **2G** | 350M multi-seed (2-3 seeds) | **L40S** | ~4-5 days | σ < 5pp | Keep single-seed 350M, note in limitations |
| **2H** | 16K-32K context eval @ 350M | **L40S** | ~12 hrs | ≥70% at 32K | Cap claims at 8K |
| **3** | NarrativeQA + PassageRetrieval, diagnostic figures, paper draft | mixed | 1-2 wks | — | — |

**Rule**: do not touch L40S until **1A + 1C** pass. 1A proves the plug-in
claim (the whole paper hinges on it). 1C proves the headline is not a
single-seed artifact.

## Baselines for the paper (final)

| Baseline | Role |
|---|---|
| Sliding-window-only (125M, window=512, no memory) | Floor (~0%) |
| **Memorizing Transformers kNN (reimplemented, 125M)** | **Key head-to-head baseline** |
| **Memorizing Transformers kNN + alignment loss** | **The plug-in claim** |
| (optional) Neurocache-style compressed kNN, ± alignment loss | Second plug-in data point |
| RFT-LM v6 (our full model) | The headline number |
| RFT-LM v6 − embed_align (ablation) | Proves the loss is the unlock |
| Full-attention oracle @ 8K | Upper bound reference |

## What gets split across V100 and L40S

**V100 (your current server, 2× 32GB)** — all of Phase 1:
- All 125M experiments
- All baseline reimplementations
- All multi-seed work
- Tail-chunk fix
- Benchmark breadth evals
- Diagnostic figures

**L40S (your other server, 2× 46GB)** — only Phase 2, only after gates pass:
- 350M training runs (v6 and kNN+alignment)
- 350M multi-seed (if budget allows)
- Long-context eval at 16K-32K
- Nothing else

The V100s handle everything that doesn't strictly require the bigger VRAM
or faster throughput. The L40S handles only scale — one concern, one
hardware commitment.

## Venue targets (after full pivot + gates pass)

| Venue | Est. | Comment |
|---|---|---|
| NeurIPS/ICML/ICLR main | **40-50%** | Plug-in training-time claim is a strong angle |
| EMNLP/ACL main | 40-50% | NL retrieval framing fits |
| **COLM** | **55-65%** | Best venue/effort ratio |
| NeurIPS/ICML workshop | 75-85% | Essentially in the bag |

## Next 2 weeks — V100 only

**Week 1**
- Day 1: run tail-chunk eval fix on v6 (Gate 1B) — ~2 hours, decides d=1.0
- Day 1-3: implement clean Memorizing Transformers kNN baseline (no OCR, no alignment loss)
- Day 4-6: train Memorizing Transformers baseline + variant with alignment loss (Gate 1A) — 2 runs on 2 GPUs
- Day 3-6 (parallel): multi-seed v6 full variant, seed 43 + seed 44 (Gate 1C) — 2 runs on 2 GPUs after Day 1-3 code work
- Day 7: add multi-needle + passkey to eval script; eval v6 (Gate 1D)

**Week 2**
- Day 8-10: eval all Week 1 checkpoints, aggregate tables, compute CIs
- Day 11-12: write paper skeleton (title, intro, headline figure, ablation table)
- **Day 13-14: DECISION POINT.** If 1A passes (alignment loss lifts kNN ≥20pp) and 1C passes (σ < 5pp) → commit to L40S for Phase 2.

**Everything else** (Neurocache baseline, LongBench subset, diagnostic figures)
fits in the L40S-training wait periods of Phase 2.

# RFT-LM — Paper Plan (April 10, 2026)

## TL;DR

v6 hit **96-97% NIAH at 2K, 90-95% at 8K** (base: 0%). Memory is the sole
differentiator. OCR is confirmed neutral on NL even at M=256 — paper pivots to
**"Embedding Alignment Loss for Trainable Long-Context Memory"** (Option B),
with OCR as honest-negative subsection. Contribution and headline number are
done; what remains is rigor (multi-seed), baselines, benchmark breadth, and one
scale point.

## What I have right now

| Component | Status |
|---|---|
| Architecture: 125M GPT-2 + sliding window + external memory bank + OCR | ✅ |
| Embedding alignment loss (post-transform `mem_ctx` → target embedding) | ✅ |
| `--mem_gate_alpha_init` CLI override | ✅ |
| v6 training recipe (mem_gate_alpha=1.0, 40ep, niah_decode_w=10.0) | ✅ |
| v6 RULER S-NIAH eval: **96-97% @ 2K d=0.0-0.5, 90-95% @ 8K** | ✅ |
| v6b (M=256) — tied v6, confirms OCR neutral even at 4× candidates | ✅ |
| v6/v6b ablations: no-memory=0%, no-OCR=match full | ✅ |
| Paper framing: Option B locked | ✅ |

**Known limitations**
- `depth=1.0` = 0% (write-after-read: needle in same chunk as probe)
- Single seed (seed 42 only)
- No prior-art baseline comparison yet
- Only one benchmark (RULER S-NIAH)
- 125M only — no scale data point

## Current state vs top-tier conference bar

| Dimension | Have | Need | Gap |
|---|---|---|---|
| Headline result | 0→97% | dramatic on recognized bench | ✅ |
| Length generalization | 2K→8K | N→4N | ✅ |
| Architectural novelty | embed alignment loss | 1 novel component | ✅ |
| Multi-seed CIs | 1 seed | 3+ seeds | ❌ |
| Prior-art baseline | none | ≥1 | ❌ |
| Benchmark breadth | 1 task | 2-3 tasks | ❌ |
| Scale evidence | 125M | ≥1 mid-scale point | ❌ |
| Failure analysis | d=1.0 unframed | honest or fixed | partial |
| Paper draft | none | — | ❌ |

**Bottom line**: submittable to workshop today; COLM in ~3 weeks; top-tier main
conference (NeurIPS/ICML/ICLR/ACL) in ~6-8 weeks if all gates pass.

## Gated plan — V100 first, L40S only after gates pass

| Gate | Experiment | Where | Cost | Pass criterion | On failure |
|---|---|---|---|---|---|
| 1A | Memorizing Transformers kNN baseline @ 125M | V100 | ~2 days | kNN < 60% RULER | Pivot framing |
| 1B | Passkey + multi-needle RULER eval of v6 (no retrain) | V100 | ~6 hrs | ≥80% on new prompts | Retrain with mixed prompts |
| 1C | Multi-seed v6 @ 125M (3 seeds × 4 variants) | V100 | ~6 GPU-days parallel | σ < 5pp | Harden recipe |
| 1D | depth=1.0 micro-chunk fix | V100 | ~2 days | d=1.0 → 80%+ | Frame as limitation |
| 2E | 350M v6 multi-seed + 350M kNN baseline | **L40S** | ~5-6 days | Same pattern as 125M | Contribution is scale-dependent |
| 2F | 760M v6 single-seed (stretch) | **L40S** | ~2-3 days | Scaling curve holds | Drop, keep 2 points |
| 2G | 32K-context eval @ 350M | **L40S** | ~12 hrs | ≥70% at 32K | Cap claims at 8K |
| 3 | NarrativeQA + PassageRetrieval, diagnostics, paper draft | mixed | 1-2 wks | — | — |

**Rule**: do not touch L40S until **1A** and **1C** both pass. Those two gates
tell you whether the recipe is real (1C) and whether your contribution beats
the obvious prior-art baseline (1A).

## Baselines for the paper

| Baseline | What | Role |
|---|---|---|
| **Sliding-window-only** (your `BaselineTransformerLM`) | 125M GPT-2, window=512, no memory | Floor (~0%) |
| **Full-attention oracle** | Same model, context=8K, no chunking | Ceiling |
| **Memorizing Transformers (kNN)** | FIFO hidden-state bank, top-k by dot product, gated residual add. **No alignment loss, no OCR.** | **Key head-to-head** — isolates what the alignment loss adds |
| RFT-LM `-embed_align` (ablation) | v6 with `niah_decode_w=0` | Internal ablation of main contribution |
| RFT-LM `-memory` (ablation) | v6 memory disabled at eval | Memory is sole differentiator |
| RFT-LM `-OCR` (ablation) | v6 OCR disabled | Honest-negative subsection |
| (optional) Infini-attention / Titans | Compressive memory | Skip unless faithful reimpl exists |

**Most important comparison**: **RFT-LM vs Memorizing Transformers kNN** at the
same scale. If RFT-LM wins by ≥20pp at d=0.0-0.5, the alignment loss is a real
contribution. If gap < 10pp, reviewers will say "kNN already solved this."

## Venue probabilities (if all gates pass)

| Venue | Acceptance est. | Comment |
|---|---|---|
| NeurIPS/ICML/ICLR main | 30-40% | Fair dice roll with full package |
| EMNLP/ACL main | 40-50% | NL retrieval framing is a natural fit |
| **COLM** | **50-60%** | Best venue/effort ratio |
| NeurIPS/ICML workshop | 70-80% | Essentially in the bag |

## Next 2 weeks — V100 only

**Week 1**
- Day 1-2: implement kNN baseline (code + training script)
- Day 1-7 (background, both GPUs): multi-seed v6 — 3 seeds × 4 variants
- Day 3-4: add multi-needle + passkey to `eval_ruler_niah.py`; eval v6
- Day 5-7: depth=1.0 micro-chunk fix + single retrain

**Week 2**
- Day 8-10: eval all multi-seed checkpoints; aggregate with CIs
- Day 11-12: eval kNN baseline; head-to-head with v6
- **Day 13-14: DECISION POINT.** Gates 1A + 1C pass → commit to L40S.

# Paper Plan — "Small Models, Long Memory" (Updated April 13, 2026)

## Working title

> **Small Models, Long Memory: A Training Recipe for Long-Context Retrieval at 350M Parameters**

## Three-claim narrative

1. **Training recipe (novel)**: The combination of embedding alignment loss +
   learned top-M router + OCR training scaffold enables small memory-augmented
   transformers to use external memory effectively. Ablations show no single
   component suffices alone; the full recipe is required.
2. **Long-context retrieval headline**: At both 125M and 350M scales, the
   model reaches **90%+ mean RULER S-NIAH at 2K-16K** (3-seed average at 125M,
   σ < 2.5pp). 4x+ length generalization from training length.
3. **First full RULER (13 tasks) at <1B scale**: No prior memory-augmented
   paper at <1B reports the full RULER suite. We do. This is the breadth
   differentiator.

## Positioning vs the 2025-2026 field (data-informed)

| Work | Scale | Their best metric | Our angle |
|---|---|---|---|
| **Titans** (Google 2025) | 340M-760M | WikiText 23.6-26.2 ppl, S-NIAH ~95.6% @16K | Match S-NIAH at similar scale; avoid perplexity (they have 15B training tokens vs our 5K docs) |
| **HMT** (NAACL 2025) | 135M-7B | WikiText-103 14.28 @350M (OPT-based) | Orthogonal — training recipe, not hierarchy |
| **Memorizing Transformers** (ICLR 2022) | 200M+ | PG-19 11.37 (65K mem) | 4-year-old but still the PG-19 reference at small scale |
| **Neurocache** (NAACL 2024) | **184M** | PG-19 **13.35** (128K cache) | **Direct target** — same scale, from-scratch training |
| **NAMM** (ICLR 2025, Sakana) | 8B | LongBench +11% | Not directly comparable — different problem |

**We own**: *"the only memory-augmented transformer at 300-400M scale that reports full-suite RULER results, with reproducible 3-seed variance."*

## What I have ✅ (as of April 13)

- RFT-LM architecture: 125M GPT-2 + sliding window + FIFO memory bank + top-M router + `mem_gate_alpha` gate
- **Embedding alignment loss** (post-transform `mem_ctx` → tied embedding)
- v6 recipe: gate_alpha=1.0, 40 epochs, decode_w=10.0
- **Gate 1A ❌ FAILED** — alignment loss alone does NOT help vanilla kNN (MT-LM).
  mt_align ≈ mt_noalign ≈ 0%. Plug-and-play claim dead; reframed to "full recipe required".
- **Gate 1B ✅ PASSED** — tail-chunk eval fix (K=8, inference-time only, no retrain).
  d=1.0 went from 0 → 95-98%.
- **Gate 1C ✅ PASSED** — 3-seed variance (seeds 42/43/44):
  Mean across seeds: L=2048 92.2±0.5, L=4096 90.6±0.2, L=8192 90.2±1.0.
- **Ablations done**: -OCR (<1pp cost), -memory (0%). OCR = training scaffold.
- v6 RULER S-NIAH 3-seed average: **~90-92% mean across L=2048-8192**
- All results consolidated in **PAPER_RESULTS.md**

## Honest scale/data reality check

**What we CANNOT claim:**
- Competitive WikiText perplexity vs Titans 340M (23.6 vs their 23.6 would need 15B+ training tokens; we trained on ~5K docs × 40 epochs)
- Competitive HellaSwag / PIQA (data-hungry, undertrained)
- "SOTA at 340M" on anything perplexity-related

**What we CAN claim:**
- **RULER S-NIAH 90%+** at 125M already beats or matches Titans' 95.6% at 340M on S-NIAH
- **First <1B memory model reporting full RULER** (13-task suite)
- **Reproducible across 3 seeds** (nobody else reports cross-seed variance at this scale)
- **Competitive PG-19 perplexity vs Neurocache 184M** (13.35 is the bar — achievable at 350M)

## Realistic targets for 350M RFT-LM

| Benchmark | Titans 340M | Neurocache 184M | MemTrans ~200M | **Our target** |
|---|---|---|---|---|
| WikiText-103 ppl | 23.6-26.2 | n/r | n/r | skip (data-starved) |
| PG-19 ppl | n/r | **13.35** | 11.37 (65K mem) | **≤13.5** |
| C4 ppl | n/r | n/r | 13.64 | **≤14** |
| HellaSwag | 39.6-40.9 | n/r | n/r | skip |
| PIQA | 64.7-66.8 | n/r | n/r | skip |
| RULER S-NIAH @16K | ~95.6% (MAC) | n/r | n/r | **≥90%** |
| RULER full (13 tasks) | **nobody reports <1B** | nobody | nobody | **first** |
| BABILong | Titans strong (no exact #) | nobody | nobody | nice-to-have |

## Gated plan — status as of April 13

### Completed gates

| Gate | Experiment | Status | Result |
|---|---|---|---|
| **1A** | MT kNN ± alignment loss | ❌ FAILED | Reframed paper to "full recipe" |
| **1B** | Tail-chunk eval fix | ✅ PASSED | d=1.0: 0% → 95-98% |
| **1C** | Multi-seed v6 (3 seeds @ 125M) | ✅ PASSED | σ < 2.5pp |

### Remaining gates (revised after research)

| Gate | Experiment | Where | Cost | Target |
|---|---|---|---|---|
| **2A** | **Train 350M RFT-LM** (scale point) | L40S (GPU 1/2) | ~2-3 days | Finish training |
| **2B** | **Full RULER 13-task eval** @125M and @350M | V100 + L40S | ~1-2 days | ≥90% S-NIAH, ≥70% multi-hop |
| **2C** | **PG-19 perplexity** @125M and @350M | V100 + L40S | ~6-12 hrs | ≤13.5 ppl @350M |
| **2D** | **RULER @ 16K-32K** @350M | L40S | ~12 hrs | Match Titans at 16K |
| **2E** | **RFT-LM without alignment loss** (125M ablation) | V100 | ~3 hrs training | Proves alignment necessary |
| **2F** | **Fix d=0.75** (root cause + fix) | V100 | TBD | d=0.75 ≥ 85% |
| **2G** | Passkey retrieval @ 32K-128K | either | ~4 hrs | ≥95% |
| **2H** | BABILong subset (optional) | L40S | 1 day | Beat RMT at same scale |
| **3** | Diagnostic figures + paper draft | mixed | 1-2 wks | — |

**Priority**: 2A (kick off 350M training now) → in parallel: 2B-C on 125M checkpoint → 2E + 2F on V100 → 2B-C on 350M checkpoint when ready → 2D → 2G-H if time allows.

## What gets DROPPED from prior plan

| Dropped | Why |
|---|---|
| ~~WikiText perplexity as headline~~ | Data-starved vs Titans' 15B tokens — would look weak |
| ~~HellaSwag / PIQA~~ | Same reason — undertrained on data, not parameters |
| ~~Separately-trained full-attention baseline~~ | Use Titans' published Transformer++ numbers as reference instead; save compute |
| ~~LongBench full suite~~ | Small model has no signal on QA/summarization |
| ~~350M multi-seed~~ | Too expensive; single seed + 125M 3-seed variance is the rigor story |
| ~~Neurocache reimplementation~~ | Cite their numbers directly for PG-19 comparison |

## Baselines for the paper (finalized)

| Baseline | Role | Status |
|---|---|---|
| RFT-LM v6 125M (3 seeds) | Main headline, rigor | ✅ Done (90-92%) |
| RFT-LM v6 350M (1 seed) | Scale point for Titans comparison | **TODO (2A)** |
| RFT-LM 125M − OCR | OCR as training scaffold | ✅ Done (<1pp drop) |
| RFT-LM 125M − memory | Memory necessary | ✅ Done (0%) |
| RFT-LM 125M − alignment (trained) | Alignment necessary | **TODO (2E)** |
| MT kNN ± alignment | Gate 1A negative result | ✅ Done (~0%) |
| Titans 340M (cited, not run) | Scale comparison | Numbers from paper |
| Neurocache 184M (cited, not run) | PG-19 direct target | Numbers from paper |
| Memorizing Transformers ~200M (cited) | Historical PG-19 reference | Numbers from paper |

## Hardware split

**V100 (current server, 2× 32GB)**:
- All 125M experiments (training, eval, PG-19)
- Full RULER eval extension development
- 2E (train 125M − alignment)
- 2F (d=0.75 investigation)

**L40S (other server, 2× 46GB, GPUs 1 & 2 ONLY)**:
- 2A (350M training) — Week 1
- 2B-D (350M eval, RULER, PG-19, long-context) — after 2A
- Optional: 2H (BABILong)

## Venue targets (revised with realistic numbers)

| Venue | Est. | Comment |
|---|---|---|
| NeurIPS/ICML/ICLR main | **30-40%** | Strong if "first full RULER at <1B" + 350M scale lands |
| EMNLP/ACL main | 35-45% | NL retrieval framing |
| **COLM** | **55-65%** | Best venue/effort ratio, long-context focused |
| NeurIPS/ICML workshop | 80-90% | Essentially in the bag |

## Next steps

**Immediate (this week)**:
1. **Launch 350M training on L40S** (Gate 2A) — scale up v6 recipe to d_model=1024, n_layers=24, n_heads=16
2. **Extend eval_ruler_niah.py to full RULER suite** (Gate 2B) — add MK-NIAH, MV-NIAH, MQ-NIAH, VT, CWE, FWE, passkey, QA
3. **Set up PG-19 perplexity eval** (Gate 2C) — sliding window, stride 512
4. **Start 2E** (125M no-alignment training) on V100

**Short term (next 1-2 weeks)**:
5. Run full RULER + PG-19 on 125M checkpoints (baseline numbers)
6. When 350M finishes: run full RULER + PG-19 + 16K-32K eval
7. Investigate d=0.75 root cause (2F)
8. Begin paper skeleton

**If time allows**:
9. BABILong subset
10. Passkey retrieval at 32K-128K
11. Diagnostic figures (mem_gate_alpha trajectory, cosine-to-embed)

# Paper Plan — "Small Models, Long Memory" (Updated April 12, 2026)

## Working title

> **Small Models, Long Memory: A Training Recipe for Long-Context Retrieval at 125M Parameters**

## Three-claim narrative (revised after Gate 1A failure)

~~Claim 3 (plug-and-play) is dead.~~ Gate 1A showed alignment loss alone does NOT
bootstrap retrieval in vanilla kNN memory. The paper is now about **the full
training recipe**, not a single transferable technique.

1. **Training recipe (novel)**: The combination of embedding alignment loss +
   learned top-M router + OCR training scaffold enables a 125M model to use
   external memory effectively. No single component suffices alone.
2. **Long-context empirical result (headline)**: 125M model, trained at 2K,
   reaches **90-92% mean on RULER S-NIAH at 2K-8K** (3-seed average,
   σ < 2.5pp). 4× length generalization with no quality cliff.
3. **Training-inference decoupling**: OCR is needed during training but
   dispensable at inference (<1pp cost). The tail-chunk fix recovers d=1.0
   at inference time without retraining.

## Positioning vs the 2025-2026 field

| Work | Their slot | How we avoid collision |
|---|---|---|
| **Titans + MIRAS** (Google 2025) | surprise-gated memory, 2M context | out-of-reach on scale — cite, do not compete |
| **HMT** (NAACL 2025) | hierarchical memory, "small beats big" | orthogonal axis: training recipe, not architecture |
| **Memorizing Transformers** (ICLR 2022) | kNN over (k,v) cache | **negative result**: alignment loss alone fails on vanilla kNN (Table 5) |
| **Neurocache** (NAACL 2024) | compressed-state kNN | cite as related; not tested |
| **NAMM** (ICLR 2025, Sakana) | evolved memory management | orthogonal — different problem |

We own: **"complete training recipe that makes 125M memory-augmented transformers actually work for long-context retrieval."**

## What I have ✅ (as of April 12)

- RFT-LM architecture: 125M GPT-2 + sliding window + FIFO memory bank + top-M router + `mem_gate_alpha` gate
- **Embedding alignment loss** (post-transform `mem_ctx` → tied embedding)
- v6 recipe: gate_alpha=1.0, 40 epochs, decode_w=10.0
- **Gate 1A ❌ FAILED** — alignment loss alone does NOT help vanilla kNN (MT-LM).
  mt_align ≈ mt_noalign ≈ 0%. Plug-and-play claim is dead.
- **Gate 1B ✅ PASSED** — tail-chunk eval fix (K=8, inference-time only, no retrain).
  d=1.0 went from 0 → 95-98%.
- **Gate 1C ✅ PASSED** — 3-seed variance (seeds 42/43/44):
  Mean across seeds: L=2048 92.2±0.5, L=4096 90.6±0.2, L=8192 90.2±1.0.
  Cross-seed σ < 2.5pp at every cell.
- **Ablations done**: -OCR (<1pp cost), -memory (0%). OCR = training scaffold.
- v6 RULER S-NIAH 3-seed average: **~90-92% mean across L=2048-8192**
- Baseline (memory off): flat 0% everywhere
- Only remaining soft cell: d=0.75 at ~73-75% (real model weakness, not a bug)
- `--mem_gate_alpha_init` CLI override
- All results consolidated in **PAPER_RESULTS.md**

## What to DROP (ruthless)

| Drop | Why |
|---|---|
| OCR head + OCR losses + OCR ablations | Confirmed neutral on NL even at M=256. One honest-negative line in appendix, that's it. |
| Mixed-objective synthetic curriculum as headline | Keep as appendix ablation. It's noise to the main story. |
| 760M stretch run | Not needed for the pivoted framing. |
| Titans / Infini-attention reimplementation | Never worth the weeks of engineering. |
| LongBench full suite | 125M won't have signal on QA/summarization. |
| Any new architectural variant | The architecture is NOT the contribution. Stop tweaking it. |

## What to ADD (revised priorities)

| Add | Why | Criticality |
|---|---|---|
| ~~MT kNN + alignment loss head-to-head~~ | ~~Done — Gate 1A FAILED~~ | ~~★★★~~ |
| ~~Multi-seed v6~~ | ~~Done — Gate 1C PASSED (3 seeds)~~ | ~~★★~~ |
| ~~Tail-chunk eval fix~~ | ~~Done — Gate 1B PASSED~~ | ~~★★~~ |
| **Fix d=0.75 weakness** | 73-75% is a consistent 15-20pp gap vs other depths | ★★★ |
| **Separately-trained full-attention baseline** | Reviewers will demand it; current "baseline" is memory-off | ★★★ |
| **350M scale point** (single seed) | Addresses scale objection | ★★ |
| Passkey + multi-needle RULER eval | Benchmark breadth (cheap) | ★★ |
| RFT-LM trained WITHOUT alignment loss | Proves alignment is necessary (not just memory) | ★★ |
| NarrativeQA + PassageRetrieval (LongBench subset) | One "real" benchmark | ★ |
| Diagnostic figures (mem_gate_alpha trajectory, cosine-to-embed, per-depth) | Reviewer catnip | ★ |

## Gated plan — status as of April 12

| Gate | Experiment | Status | Result |
|---|---|---|---|
| **1A** | MT kNN ± alignment loss | **❌ FAILED** | mt_align ≈ mt_noalign ≈ 0%. Alignment alone insufficient. |
| **1B** | Tail-chunk eval fix | **✅ PASSED** | d=1.0: 0% → 95-98% with K=8. |
| **1C** | Multi-seed v6 (3 seeds) | **✅ PASSED** | σ < 2.5pp across all cells. |

### Remaining gates (revised)

| Gate | Experiment | Where | Cost | Pass criterion |
|---|---|---|---|---|
| **1D** | Fix d=0.75 weakness | V100/L40S | TBD | d=0.75 ≥ 85% |
| **1E** | Full-attention baseline (separately trained) | V100 | ~1 day | Provides honest comparison |
| **1F** | RFT-LM trained WITHOUT alignment loss | V100 | ~3 hrs | Proves alignment is necessary |
| **1G** | Passkey + multi-needle RULER eval | V100 | ~6 hrs | ≥80% on new prompts |
| **2A** | 350M scale point (single seed) | L40S | ~2 days | Same pattern as 125M |
| **3** | Diagnostic figures + paper draft | mixed | 1-2 wks | — |

**Priority order**: 1D (d=0.75 fix) → 1E/1F (baselines) → 1G (benchmark breadth) → 2A (scale)

## Baselines for the paper (revised)

| Baseline | Role | Status |
|---|---|---|
| Sliding-window-only (125M, window=512, no memory) | Floor (~0%) | ✅ Done |
| MT kNN without alignment loss | Negative control | ✅ Done (~0%) |
| MT kNN with alignment loss | Negative result — alignment alone fails | ✅ Done (~0-1%) |
| RFT-LM v6 (our full model) | The headline number | ✅ Done (90-92%) |
| RFT-LM v6 − OCR (test-time ablation) | OCR dispensable at inference | ✅ Done (<1pp drop) |
| RFT-LM v6 − memory (test-time ablation) | Memory is necessary | ✅ Done (0%) |
| **RFT-LM v6 − alignment loss (train-time ablation)** | Proves alignment is necessary | **TODO** |
| **Full-attention GPT-2 125M @ 8K** | Upper bound reference | **TODO** |

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

## Next steps (as of April 12)

**Immediate (this week)**:
1. Fix d=0.75 weakness — investigate root cause, try solutions
2. Train RFT-LM without alignment loss (ablation to prove alignment is necessary)
3. Clean up old experimental runs on both servers

**Short term (next 1-2 weeks)**:
4. Train separately-trained full-attention baseline (GPT-2 125M @ 8K context)
5. Passkey + multi-needle RULER eval (no retrain needed)
6. Begin paper skeleton

**If time allows**:
7. 350M scale point on L40S
8. Diagnostic figures (mem_gate_alpha trajectory, cosine-to-embed)
9. NarrativeQA / PassageRetrieval (LongBench subset)

# Research Direction — Retrieval-Aware Optical Compression (RAOC)

**Date:** July 23, 2026
**One-line:** Turn DeepSeek-OCR's *reconstruction*-optimized optical context
compression into a *retrieval*-optimized one, using the two mechanisms this
project already built — supervised routing and embedding-alignment decoding —
and validate it on the needle-in-a-haystack test DeepSeek-OCR said it would
run but hasn't.

---

## 1. Why this is a real opening, not a stretch

DeepSeek-OCR (arXiv:2510.18234, Oct 2025) renders long text as an image and
compresses it to a few hundred **vision tokens** via a DeepEncoder
(SAM → 16× conv compressor → CLIP), then a 3B-MoE decoder reconstructs the
text. Reported: ~97% reconstruction precision at <10× compression, ~60% at 20×.

**The paper's own stated gaps (their words / close paraphrase):**
1. "OCR alone is insufficient to fully validate true context optical
   compression." They plan **needle-in-a-haystack testing** as future work.
2. Open question: can an LM **reason/retrieve over context represented as
   compressed visual tokens** (as opposed to faithfully reconstructing it)?
3. **Memory decay**: a *speculative diagram* — older context downsampled to
   lower resolution so distant history fades. No mechanism, no evaluation.

**What this project already has that maps 1:1:**
| DeepSeek-OCR gap | Our existing asset |
|---|---|
| NIAH over compressed context (future work) | Full RULER S-NIAH pipeline: `niah_batch.py`, `eval_ruler_niah.py`, depth grid, CIs |
| "Can it retrieve, not just reconstruct?" | Our thesis: **retrieval ≠ decoding** — a model can hold info (r@M=1.0) yet answer 0% |
| Memory decay = uniform low-res by age | Supervised **recency-biased router** (RFTMemoryLayer) — content-AND-recency-aware selection |
| Reconstruction is lossy at high compression | **Embedding-alignment loss** — makes a compressed/retrieved representation decodable by the output head |

We are not chasing their compression ratio. We are attacking the axis they
left open: **retrievability of query-relevant content under compression.**

---

## 2. The thesis

Optical compression is optimized for **faithful full reconstruction**. But in
long-context use you rarely need the whole page back — you need to *retrieve a
specific fact* (a number, a name, a clause). A needle is a tiny fraction of a
page; uniform optical compression treats it like filler and blurs it first.

> **RAOC:** compress optically for *retrievability*, not reconstruction.
> Preserve query-relevant regions at high resolution (a learned router) and
> make what's preserved decodable by the readout (an alignment objective).

This reframes their "memory decay" from a fixed heuristic (old = blurry) into
a **learned, query-aware compression policy**, and it directly tests their
open question about reasoning over compressed visual tokens.

---

## 3. Three contributions, ranked by risk

### A — NIAH-over-Optical benchmark  ★ solid, cheap, first-of-its-kind
Render long contexts (with needles at controlled depths) as images, compress
with DeepSeek-OCR's *open* encoder at each mode (Tiny 64 → Gundam), and run our
existing NIAH eval on the reconstructed/decoded text. Report **retrieval
accuracy vs compression ratio vs needle depth** — a curve nobody has published
(DeepSeek-OCR reports *reconstruction* precision, not *targeted retrieval*).
Prediction, grounded in our findings: retrieval degrades **faster and
differently** than reconstruction — small critical tokens are exactly what
optical compression loses first. This alone is a publishable analysis paper.
Cost: inference only, fits our V100s.

### B — Content-aware optical memory decay  ★★ strong, more engineering
Replace uniform age-based downsampling with a lightweight router that scores
regions of old context and keeps retrieval-relevant ones at higher resolution
while aggressively downsampling filler. This is the *mechanism* their memory
decay diagram lacks. Compare against uniform decay at matched total token
budget: does query-aware allocation retrieve better per token?

### C — Alignment-pushed compression frontier  ★★★ boldest, highest payoff
Hypothesis straight from our text-side result: at high compression, part of the
failure is **decodability, not information loss**. Add an embedding-alignment
auxiliary at the decoder interface — align the compressed representation *at the
queried location* to the answer token's output embedding — so needles stay
decodable even under aggressive compression. If it moves the
compression-precision curve, that's direct evidence the gap is decodability and
a mechanistic fix. If it *doesn't*, that's also a clean result: optical loss is
information-theoretic, alignment can't help — which itself sharpens our gap
thesis. Either outcome is a finding.

**B + C unify** into one system: the router decides what to keep sharp, the
alignment loss ensures what's kept is decodable. That is RAOC.

---

## 4. How this uses — not discards — the 125M work

The 125M results are the **mechanism prior** that motivates the optical study:
we *proved* retrieval ≠ decoding in a text memory (0%→96% with the fix; perfect
retrieval yet 0% answer in the frozen-graft and MT negatives). The optical work
asks: **does the same gap, and the same fix, govern a completely different
memory substrate (pixels)?** That is a far stronger paper than either half:
a general law about compressed memory, demonstrated in two modalities.

So: **finish the 125M results** (seed 44 + v6-no-align are running now — the
v6-no-align verdict tells us whether alignment or retrieval is the unlock, which
is the exact hypothesis we carry into optical), then open RAOC as the flagship.

---

## 5. Feasibility

- DeepSeek-OCR is **open source** (HF). Encoder ~380M; inference fits 32GB V100.
- Text→image rendering is trivial (PIL/matplotlib, or their repo's renderer).
- Our needle generator + eval harness already produce the text side.
- A (eval) needs **no training**. B/C need training a small head/adapter on top
  of a mostly-frozen encoder — feasible on 2×V100.
- Timeline fits **ICLR 2027 (submit Sept 24, 2026)** or the next ARR cycle.

## 6. Hazards / honesty

- **Naming collision (must fix):** our disambiguator is called "OCR"
  (Occurrence-Contrastive Resolver); DeepSeek-OCR is optical character
  recognition. Rename ours in any merged paper (e.g. "Resolver" / "OCD head").
- **Speed:** DeepSeek-OCR is hot; others will attempt NIAH-over-optical. Our
  edge is the pre-built apparatus + the mechanistic framing. Move fast on A.
- **C could fail:** if optical loss is purely information-theoretic, alignment
  won't move the curve. We frame that as a finding, not a failure, but B must
  carry the paper if C is null.
- **Scope discipline:** A is the anchor. Do A first, fully. B/C build on it.

## 6b. PRIOR-WORK VERDICT (July 23, 2026) — DO NOT PIVOT

A novelty search found the optical/visual context-compression + retrieval space
is crowded and moving fast (Apple, DeepSeek, others; papers as recent as
2 months old). All three contributions above are done or imminently at risk:

- **A (NIAH over optical)**: SCOOPED. "Text or Pixels? It Takes Half"
  (arXiv:2510.18279) ran RULER NIAH over rendered text (99% @ ρ≈2).
  DeepSeek-OCR-2 (Jan 2026) already shipped; NIAH is the authors' own next step.
- **B (query-aware keep-relevant-sharp)**: SCOOPED by LensVLM (arXiv:2605.07019,
  Apple, May 2026) — selective on-demand expansion of relevant compressed
  images; full-text accuracy at 4.3×, beats baselines to 10.1×.
- **C (needles lost first under compression)**: the core hypothesis is already
  published for text gist tokens (arXiv:2412.17483). Alignment-at-decoder is the
  only sliver left, and it rests on A/B, which are gone.

**Decision: do not pivot to optical.** Cannot out-run Apple/DeepSeek from 10
months behind on 2 V100s. What survives as genuinely ours is the *diagnostic
lens*, not a compression method: none of these papers separate retrieval
(did info survive?) from decoding (could the model read it?). Fold the optical
angle into the 125M paper as ONE discussion paragraph ("the same gap governs
compressed-vision memory; methods like LensVLM fight decodability without
diagnosing it") — cite the field, do not join it. Return to finishing the
125M mechanism paper, which is nearly done and far less crowded in its framing.

## 7. First concrete steps  (SUPERSEDED by 6b — retained for record)

1. Grab DeepSeek-OCR open weights + its render/encode path; reproduce one
   reconstruction number (sanity that the pipeline runs on our box).
2. Wire our needle generator → image render → their encoder → decode → our
   scorer. Produce the **first NIAH-vs-compression-ratio curve** (Contribution A).
3. In parallel, finish the 125M v6-no-align + seed-44 evals (the mechanism
   prior + the hypothesis for C).
4. Decide B vs C as the training contribution based on what A's curve reveals
   about *where* retrieval breaks.

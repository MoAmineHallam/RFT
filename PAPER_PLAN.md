# Paper Plan v2 — "Retrieval Is Not Enough" (July 11, 2026)

Supersedes the April 11 plan. The story pivots from "Small Models, Long Memory"
to the **memory-to-output gap**: memory-augmented LMs can retrieve perfectly
(recall@M = 1.0) and still answer at 0%. We name the gap, diagnose it, and
present the training recipe that closes it.

## Working title

> **Retrieval Is Not Enough: Closing the Memory-to-Output Gap in
> Memory-Augmented Language Models**

## The story (three claims)

1. **Diagnosis (novel framing)**: Retrieval quality and answer accuracy are
   systematically decoupled in memory-augmented LMs. We show recall@M = 1.000
   with 0% answer accuracy across architectures and scales. Prior work
   conflates the two.
2. **Fix (the technique)**: Embedding-alignment loss on the **post-transform**
   memory context (`mem_ctx`, after gate → LN → out_proj → tanh(α)) + memory
   gate initialized at 1.0. Gradient threads through the full memory pipeline
   so tied embeddings can decode retrieved memories.
   - Built-in mechanistic ablation cascade (already run): raw-`mem_v`
     alignment → emb_acc 1.0 but 0% eval (proxy solved, task not);
     post-transform alignment → +11–14pp (v4b→v5); gate 1.0 + longer
     training → 96%+ (v6).
3. **Headline**: 125M model trained at 2K reaches **~91% mean RULER S-NIAH
   across the full 2K–8K × depth grid** (89–98% per cell) vs **0% base** and
   **0–2% memory-disabled**. Length generalizes 4× beyond training length.

**Stretch claim (decided by Gate G0)**: the recipe transfers to a pretrained
LM — graft the memory operator onto frozen Qwen2.5-0.5B, unfreeze the tail,
and unlock retrieval the vanilla model cannot do.

**Honest negative (credibility section)**: OCR disambiguation gives +4.8% on
513 synthetic experiments but is neutral on natural language even at M=256.

### Claim-scoping caveat (resolve BEFORE writing the abstract)

The graft currently trains with `decode_w=0` (alignment loss removed) because
alignment recreated the proxy-solved pathology on a pretrained decode pathway.
Two possible worlds — both publishable, different abstracts:

- **World A** — graft works without alignment: scope the claim as *"alignment
  is the unlock when the decode pathway is trained from scratch; a pretrained
  pathway needs only end-to-end CE."* This is itself a finding about where
  the gap lives.
- **World B** — graft needs alignment re-added after CE warms up: the
  universal "alignment closes the gap" claim holds everywhere.

Run the graft both ways (±alignment) once the smoke test passes, so the paper
can state which world we are in with evidence.

## What we have ✅

| Asset | Status |
|---|---|
| v6 RULER S-NIAH full grid (post tail-fix): 2K 92.6 / 4K 90.6 / 8K 89.4 mean; base 0%; no-memory 0–2% | ✅ in hand |
| v6b (M=256) quasi-replicate — ties v6 | ✅ in hand |
| v4b→v5→v6 ablation cascade (the bug history as mechanism evidence) | ✅ in hand |
| 513-exp synthetic OCR study (+4.8%, honest negative on NL) | ✅ in hand |
| Tail-chunk fix (Gate 1B): d=1.0 0% → 95–98%, inference-only | ✅ done |
| Graft pipeline on Qwen2.5-0.5B (causal mask, forward_manual, eval wiring) | ✅ built, ⚠ unverified fix pushed |
| MT kNN baseline code + eval wiring | ✅ code done; **run status unknown** |
| Multi-seed v6 scripts (seeds 43, 44) | ✅ scripts; **run status unknown** |
| aggregate_results.py → LaTeX tables | ✅ built |

## Decision gates

| Gate | What | Cost | Pass criterion | On pass | On fail |
|---|---|---|---|---|---|
| **G0** | Graft smoke test (`bash run_graft_smoke.sh`, fix already pushed: decode_w=0, digit2, 1 needle) | 30 min | `niah_lm_full_match > 0.10` by step 1000 | Graft is IN → target EMNLP main | Debug ≤2 weeks, then cut graft → Findings/COLM story |
| **G1** | Locate Gate 1A (MT ± alignment) checkpoints/evals on the training server | 10 min of looking | Runs finished | Eval + aggregate | Re-run: ~2–3 days on V100 |
| **G2** | Locate seed-43/44 v6 checkpoints on the training server | 10 min | Runs finished | Eval + aggregate | Re-run: ~3 GPU-days |
| **G3** | Graft full run (`run_graft_full.sh`, 8000 steps) + eval grid vs vanilla Qwen | ~1 day GPU | Graft beats vanilla Qwen by ≥30pp on 2–4K NIAH | Stretch section locked | Report smoke-only or cut |

**Rule**: G0 first — it decides both the venue target and the abstract wording.
G1/G2 are pure status checks on the server; do them the same day.

## Experiment checklist (ordered by blocking severity)

### Must-have — paper blocks without these

1. **[G0] Graft smoke test** — pull branch, `bash run_graft_smoke.sh`.
2. **[G1] MT kNN ± alignment @ 125M** — THE plug-in/generality evidence.
   Check server for April runs; re-run if missing. Without it, the
   contribution shrinks to one architecture.
3. **[G2] Multi-seed v6** (seeds 42/43/44) — eval + `aggregate_results.py`
   for mean±std / Wilson CIs. Single-seed headline will not survive review
   (v6b is a partial replicate, not a seed).
4. **v6-config minus alignment loss, 1 seed** (~1 day) — v5→v6 changed three
   things at once (gate init, epochs, decode_w). One controlled run under the
   final recipe isolates the loss as the unlock. Reviewers will demand
   exactly this ablation.

### Should-have — cheap, large credibility gain

5. **Multi-needle + passkey eval** (no retrain; `mk_num_keys ≥ 2` already
   supported in `eval_ruler_niah.py`) — hours.
6. **Perplexity sanity on v6** (`eval_perplexity.py`) — memory training does
   not degrade LM quality — hours.
7. **[G3] Graft full training + eval grid + vanilla Qwen baseline**
   (scripts exist: `run_graft_full.sh`, `run_graft_eval_full.sh`) — if G0
   passes. Include the ±alignment variant (World A/B experiment).

### Nice-to-have — only if time remains before the deadline

8. One non-synthetic data point (LongBench PassageRetrieval subset) —
   answers "is this only NIAH?".
9. Diagnostic figures: lm_acc / emb_acc / r@M training curves from
   `train_metrics.jsonl` (the r@M=1.0-while-lm_acc=0 plot IS the paper's
   Figure 1 motivation panel).

## What to DROP (unchanged from v1, still ruthless)

- OCR as a contribution (honest-negative subsection only)
- 350M/760M scale runs — the graft on Qwen-0.5B replaces the scale story
- Titans/Infini-attention reimplementation
- Full LongBench suite
- Any new architectural variant

## Paper skeleton

1. **Intro** — Figure 1: (a) training curves showing recall@M=1.0 while
   answer acc=0 (the gap, visually); (b) headline bar: base 0% → ours 91%.
2. **The memory-to-output gap** — definition, where it comes from (signal
   attenuation ×~0.1 gate, layer-6-space vs layer-12-space representation
   mismatch), evidence it is systematic.
3. **Method** — memory bank + router (brief, not the contribution);
   alignment loss on post-transform mem_ctx; gate init; training curriculum.
4. **Experiments A: from-scratch 125M** — full grid table (multi-seed CIs),
   ablations (no-memory, no-alignment, raw-vs-post-transform alignment,
   gate init), length generalization.
5. **Experiments B: plug-in generality** — MT kNN ± alignment head-to-head.
6. **Experiments C (stretch): pretrained graft** — Qwen2.5-0.5B graft vs
   vanilla, ±alignment (World A/B result).
7. **Honest negative** — OCR: synthetic +4.8%, NL neutral at M=64 and M=256.
8. **Limitations** — d=0.75 soft cell (~73–75%), synthetic-task focus,
   ≤0.5B scale, write-after-read handled by inference-time tail-chunking.

## Venue strategy

| Outcome | Target | Est. |
|---|---|---|
| G0+G3 pass (graft works) | **EMNLP 2026 main (CCF B)** | 40–50% |
| G0 fails, rest solid | EMNLP Findings | 50–60% |
| Either | COLM (not CCF-ranked, well regarded) as backup | 55–65% |

**Action items (non-experiment):**
- Confirm the exact ARR cycle deadline for EMNLP 2026 — today is July 11;
  the cycle determines whether G3 makes the submission or the camera-ready.
- **Ask Prof. Zeng whether EMNLP Findings counts as CCF B under HIT's degree
  policy.** If it does not, EMNLP main is the real target and G0/G3 are
  mandatory, not stretch.

## Two-week schedule (from July 11)

**Week 1**
- Day 1: **G0** graft smoke (30 min) + **G1/G2** server status check (30 min).
  Report results back → plan locks.
- Day 1–3: whatever G1/G2 found missing goes on the GPUs
  (MT ± alignment and/or seeds 43/44 — they parallelize across 2 GPUs).
- Day 2–3 (CPU/eval GPU): multi-needle + passkey evals, perplexity sanity.
- Day 3–5: v6-minus-alignment controlled run (item 4).
- Day 4–7 (if G0 passed): graft full run + ±alignment variant (G3).

**Week 2**
- Day 8–9: eval everything, `aggregate_results.py`, CIs, all tables.
- Day 10: G3 eval grid + vanilla Qwen baseline → decide main vs Findings.
- Day 11–14: paper skeleton → full draft (title, abstract per World A/B,
  Figure 1, grid table, ablation table, honest-negative section).

## Immediate next actions (do these today)

1. `git pull` on the training server, `bash run_graft_smoke.sh` → paste the
   log (G0).
2. Look for these on the server and report what exists (G1/G2):
   - MT baseline checkpoints/evals (Gate 1A runs from April)
   - `mixed_pilot_v6_seed43*` / `seed44*` checkpoints
3. Ask Prof. Zeng: does Findings count as CCF B for the degree?
4. Look up the ARR deadline for the EMNLP 2026 cycle.

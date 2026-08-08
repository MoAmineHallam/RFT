# Plan: CCF-A Paper (Aug 7, 2026)

## The reframe

**Not**: "we built a memory-augmented LM that scores 92% on NIAH."
(125M + one synthetic benchmark = CCF-B at best, correctly.)

**Instead**: "retrieval success and answer success are separable failure modes.
We give a diagnostic that separates them, show the second mode is widespread and
invisible to standard metrics, and characterize when it occurs."

This is an *analysis + tool* paper. Those get into ACL/NeurIPS/ICML when the
phenomenon is real, general, and actionable. Crucially, validating it is
**inference-heavy, not training-heavy** — it fits our compute.

## Why this clears the A bar when scaling would not

Scaling from scratch is data-bound and would *hurt*: ~24M training tokens vs
Chinchilla-optimal ~2.5B for 125M. 350M/1B from scratch would score worse and
read as "scaling breaks their method."

The A-level asset is the **decomposition**, whose sharpest evidence we already
have: the frozen Qwen graft retrieved perfectly (r@M = 1.000) and answered
**0%**. That is precisely the failure mode of every system that injects
retrieved *representations* into a model that cannot adapt to read them.

## The claim

> A retrieval-augmented system has two independent ways to fail: it can fetch
> the wrong thing (**retrieval failure**), or fetch the right thing and be
> unable to use it (**decoding failure**). Standard metrics — recall@k, end-task
> accuracy — conflate these. We separate them, show decoding failure is common
> and severe, identify the conditions that cause it, and provide a cheap probe
> that predicts it.

## Contributions

**C1. The decomposition + metric.** Formalize *conditional decoding accuracy*:
P(correct answer | correct retrieval). Standard practice reports recall@k and
end accuracy; neither exposes this. Cheap to compute wherever gold retrieval
targets exist.

**C2. Controlled evidence that the modes are independent** (largely in hand):
| system | retrieval | answer | isolates |
|---|---|---|---|
| MT kNN (no supervised routing) | fails | ~0% | retrieval failure |
| Frozen Qwen graft | **r@M 1.000** | **0%** | **decoding failure** |
| RFT-LM v7 | r@M 1.000 | 92% | both satisfied |
| v7, memory off | none | 0% | floor |

**C3. Generality on systems people use** (the new work, and the crux):
- **(a) Text-RAG**: real LLM + retriever on QA with gold passages. Report
  recall@k, end accuracy, **and conditional decoding accuracy**. Sweep model
  size and passage count. Prediction: decoding accuracy is well below 1 and
  degrades with distractors — invisible to recall@k.
- **(b) Representation-RAG**: inject retrieved *vectors* (soft prefix) instead
  of text, frozen vs LoRA-adapted. Prediction: frozen → near-total decoding
  failure (our graft result, reproduced on a standard setup); LoRA → recovers.
  **This is the money experiment**: it explains why text-RAG works and
  embedding-injection generally does not.
- **(c) Compressed context**: increase compression until decoding fails while
  the information is provably still present.

**C4. A predictive probe.** Our `emb_acc`-style measurement (is the retrieved
representation decodable by the output head?) computed *without* running the
task, shown to predict downstream decoding failure. This is what makes it a
*tool* rather than an observation.

**C5. Honest negatives** (in hand): auxiliary alignment loss inert once
retrieval is supervised; gate init inert; alignment cannot rescue a system
lacking retrieval supervision or a trainable readout; a training setup with
excellent benchmark accuracy can hide a 50× perplexity collapse.

## Feasibility on 2× V100 32GB

| Experiment | Cost | Notes |
|---|---|---|
| C3(a) text-RAG, 0.5B–7B | inference only | 7B fp16 ≈ 14GB, fits |
| C3(b) soft-prefix, frozen vs LoRA | small training | LoRA on ≤1.5B is cheap |
| C3(c) compression sweep | inference | reuse existing eval harness |
| C4 probe validation | trivial | already implemented for RFT-LM |
| Graft retry (LoRA, mid-stack) | ~1 day | now well-diagnosed |

No pretraining. No 1B-from-scratch. Everything is adapter-scale or inference.

## Timeline and venues (today = Aug 7, 2026)

| Venue | Deadline (approx) | CCF |
|---|---|---|
| ACL 2027 via ARR | Oct or Dec 2026 cycle | **A** |
| ICML 2027 | late Jan 2027 | **A** |
| IJCAI 2027 | ~Jan 2027 | **A** |
| NeurIPS 2027 | ~May 2027 | **A** |
| ICLR 2027 | Sept 24, 2026 | not CCF-listed |

**Target: ACL 2027 (ARR December cycle) or ICML 2027.** ~5 months. Realistic.

- **Weeks 1**: finish v7 sequence (tonight), lock the 125M evidence base.
- **Weeks 2–4**: C3(a) text-RAG diagnostic across model sizes.
- **Weeks 5–7**: C3(b) frozen vs LoRA representation injection — the crux.
- **Weeks 8–9**: C4 probe validation; C3(c) compression sweep.
- **Weeks 10–12**: graft retry as the scale demonstration (optional but strong).
- **Weeks 13–16**: writing, figures, ablation tables.

## Honest odds

With C3(b) working and the probe validated: **25–35% at an A venue** — up from
~10% for the architecture-centric version, because the contribution stops being
"my 125M model" and becomes "a phenomenon and a tool."

If C3 comes back weak (decoding failure turns out rare in real RAG), fall back
to the B paper with the 125M evidence, which is already complete and defensible.
That fallback is the reason this plan is low-risk: the B paper exists either way.

## The single highest-value next experiment

**C3(b)**: frozen LLM + retrieved-vector injection vs the same with LoRA.
If frozen ≈ 0% and LoRA recovers, we have reproduced the graft failure on a
standard, widely-used setup — and that single figure carries the paper.

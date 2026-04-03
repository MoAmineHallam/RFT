"""
eval_ruler_niah.py — RULER Single Needle-In-A-Haystack (S-NIAH) Benchmark

Evaluates RFT-LM and Baseline on the standard RULER S-NIAH task used by
Titans, Mamba-2, DeltaNet, and other long-context papers.

Task format:
  - A key-value pair ("The special magic number for <KEY> is <VALUE>.")
    is inserted at a random position in a long context of distractor text.
  - Distractor text is real C4 validation text (or generated filler).
  - The model is prompted at the end: "What is the special magic number for <KEY>?"
  - We check if the model generates VALUE as its next tokens.
  - Accuracy is measured across many trials at each context length.

Published reference numbers at 125M scale (from Titans paper, Table 2):
  - Titans MAC:   ~96-99% across 4K-32K
  - Transformer+: ~90-95% at 4K, degrades at longer contexts
  - Mamba-2:      ~85-92% at 4K, degrades at longer contexts
  - DeltaNet:     ~88-95% at 4K, degrades at longer contexts

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_ruler_niah.py \
        --rft_ckpt ./runs_lm/overnight_v2/rft_lm/best_model.pt \
        --baseline_ckpt ./runs_lm/overnight_v2/baseline/best_model.pt \
        --tokenizer_path ./gpt2_tokenizer \
        --distractor_path ./data/c4_val.jsonl \
        --seq_lens 2048 4096 8192 16384 \
        --n_trials 100
"""

import argparse
import json
import math
import os
import random
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from RFT_LM import RFTLM, BaselineTransformerLM, MemoryBank


# ─────────────────────────────────────────────
# NIAH Task Construction
# ─────────────────────────────────────────────

# 100 distinct keys and values to avoid memorization
NIAH_KEYS = [f"alpha_{i:03d}" for i in range(100)]
NIAH_VALUES = [f"{random.Random(42 + i).randint(10000, 99999)}" for i in range(100)]


def load_distractor_tokens(
    path: str, tokenizer, max_docs: int = 500
) -> List[int]:
    """Load and tokenize distractor text from C4 validation."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            obj = json.loads(line)
            texts.append(obj["text"])
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    return all_tokens


def build_niah_sequence(
    seq_len: int,
    distractor_tokens: List[int],
    tokenizer,
    key_idx: int,
    needle_position_frac: float = None,
    seed: int = 0,
) -> Tuple[List[int], str, str, int]:
    """
    Build a single NIAH sequence.

    Returns:
        (token_ids, key, value, needle_start_pos)
    """
    rng = random.Random(seed)

    key = NIAH_KEYS[key_idx % len(NIAH_KEYS)]
    value = NIAH_VALUES[key_idx % len(NIAH_VALUES)]

    # Needle sentence
    needle_text = f" The special magic number for {key} is {value}. "
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)

    # Query at the end
    query_text = f" What is the special magic number for {key}? The answer is:"
    query_tokens = tokenizer.encode(query_text, add_special_tokens=False)

    # Budget for distractor text
    distractor_budget = seq_len - len(needle_tokens) - len(query_tokens)
    if distractor_budget < 100:
        raise ValueError(f"seq_len {seq_len} too short for NIAH task")

    # Pick needle insertion position
    if needle_position_frac is not None:
        needle_pos = int(needle_position_frac * distractor_budget)
    else:
        # Random position (avoid very start/end)
        needle_pos = rng.randint(
            max(1, distractor_budget // 10),
            max(2, distractor_budget - distractor_budget // 10),
        )

    # Build distractor: sample a random contiguous block
    max_start = max(0, len(distractor_tokens) - distractor_budget - 1)
    dist_start = rng.randint(0, max_start) if max_start > 0 else 0
    distractor_chunk = distractor_tokens[dist_start:dist_start + distractor_budget]

    # Pad if needed
    while len(distractor_chunk) < distractor_budget:
        wrap_start = rng.randint(0, max(0, len(distractor_tokens) - 100))
        distractor_chunk.extend(distractor_tokens[wrap_start:wrap_start + 100])
    distractor_chunk = distractor_chunk[:distractor_budget]

    # Assemble: distractor_before + needle + distractor_after + query
    before = distractor_chunk[:needle_pos]
    after = distractor_chunk[needle_pos:]
    full_tokens = before + needle_tokens + after + query_tokens

    # Truncate to exact seq_len
    full_tokens = full_tokens[:seq_len]

    return full_tokens, key, value, needle_pos


# ─────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def eval_niah_model(
    model,
    tokenizer,
    distractor_tokens: List[int],
    seq_len: int,
    chunk_size: int,
    device: torch.device,
    use_memory: bool,
    n_trials: int = 100,
    needle_depth: float = None,
) -> dict:
    """
    Run NIAH evaluation for a single model at a single seq_len.

    Returns dict with accuracy and per-trial details.
    """
    model.eval()
    correct = 0
    total = 0
    details = []

    for trial in range(n_trials):
        key_idx = trial % len(NIAH_KEYS)

        tokens, key, value, needle_pos = build_niah_sequence(
            seq_len=seq_len,
            distractor_tokens=distractor_tokens,
            tokenizer=tokenizer,
            key_idx=key_idx,
            needle_position_frac=needle_depth,
            seed=trial * 7919 + seq_len,  # deterministic but varied
        )

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Chunked forward (same as eval_perplexity)
        memory_bank = MemoryBank(max_entries=65536)
        memory_bank.reset()

        total_len = input_ids.shape[1]
        last_logits = None

        for start in range(0, total_len, chunk_size):
            end = min(start + chunk_size, total_len)
            chunk_input = input_ids[:, start:end]

            if use_memory:
                mem_k, mem_v, mem_pos = memory_bank.get_state()
                try:
                    out = model(
                        chunk_input,
                        memory_keys=mem_k,
                        memory_vals=mem_v,
                        memory_positions=mem_pos,
                        pos_offset=start,
                        return_memory_state=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    details.append({"trial": trial, "correct": False, "oom": True})
                    total += 1
                    continue

                if "new_mem_keys" in out:
                    memory_bank.add(
                        out["new_mem_keys"].detach(),
                        out["new_mem_vals"].detach(),
                        start, chunk_input.shape[1],
                    )
            else:
                out = model(chunk_input, pos_offset=start)

            last_logits = out["logits"]

        if last_logits is None:
            total += 1
            details.append({"trial": trial, "correct": False, "error": "no_logits"})
            continue

        # Check: does the model predict the value tokens after the query?
        # The last token of input is the last token of "The answer is:"
        # We greedily decode from there
        value_tokens = tokenizer.encode(f" {value}", add_special_tokens=False)
        n_value_tokens = len(value_tokens)

        # Get the logits for the last position of the input
        # and greedily generate n_value_tokens
        generated = []
        logits_pos = last_logits[0, -1, :]  # logits after last input token

        for gen_step in range(n_value_tokens):
            pred_token = logits_pos.argmax().item()
            generated.append(pred_token)

            if gen_step < n_value_tokens - 1:
                # Feed predicted token back (single-token forward)
                next_input = torch.tensor([[pred_token]], device=device)
                next_pos = total_len + gen_step

                if use_memory:
                    mem_k, mem_v, mem_pos = memory_bank.get_state()
                    out = model(
                        next_input,
                        memory_keys=mem_k,
                        memory_vals=mem_v,
                        memory_positions=mem_pos,
                        pos_offset=next_pos,
                        return_memory_state=False,
                    )
                else:
                    out = model(next_input, pos_offset=next_pos)
                logits_pos = out["logits"][0, -1, :]

        # Check exact match
        is_correct = (generated == value_tokens)

        # Also check if decoded text contains the value (more lenient)
        generated_text = tokenizer.decode(generated).strip()
        contains_value = value in generated_text

        if is_correct or contains_value:
            correct += 1

        total += 1
        details.append({
            "trial": trial,
            "key": key,
            "value": value,
            "correct_exact": is_correct,
            "correct_contains": contains_value,
            "generated_text": generated_text,
            "needle_pos_frac": needle_pos / seq_len,
        })

        if (trial + 1) % 20 == 0:
            print(f"    trial {trial + 1}/{n_trials}: "
                  f"acc={correct}/{total} ({100*correct/total:.1f}%)")

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


# ─────────────────────────────────────────────
# Model Loading (reuse from eval_perplexity)
# ─────────────────────────────────────────────

def load_rft_model(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = RFTLM(
        vocab_size=a["vocab_size"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        window_size=a["window_size"],
        ff_mult=a["ff_mult"],
        dropout=0.0,
        memory_layer_idx=a["memory_layer_idx"],
        use_memory=True,
        mem_top_m=a["mem_top_m"],
        ocr_dim=a["ocr_dim"],
    ).to(device)
    state_dict = ck["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] RFT-LM: {sum(p.numel() for p in model.parameters()):,} params "
          f"(step {ck['step']}, loss {ck['loss']:.4f})")
    return model


def load_baseline_model(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = BaselineTransformerLM(
        vocab_size=a["vocab_size"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        ff_mult=a["ff_mult"],
        dropout=0.0,
        max_len=65536,
    ).to(device)
    state_dict = ck["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] Baseline: {sum(p.numel() for p in model.parameters()):,} params "
          f"(step {ck['step']}, loss {ck['loss']:.4f})")
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--distractor_path", type=str, required=True,
                    help="Path to C4 val jsonl for distractor text")
    ap.add_argument("--seq_lens", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_trials", type=int, default=100,
                    help="Number of NIAH trials per seq_len")
    ap.add_argument("--needle_depth", type=float, default=None,
                    help="Fixed needle position (0-1). None = random.")
    ap.add_argument("--outfile", type=str, default="niah_results.json")
    ap.add_argument("--distractor_docs", type=int, default=500)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9
    print(f"[EVAL] Tokenizer: {args.tokenizer_path} (vocab={tokenizer.vocab_size})")

    # Load distractor text
    print(f"[EVAL] Loading distractor text from {args.distractor_path}...")
    distractor_tokens = load_distractor_tokens(
        args.distractor_path, tokenizer, max_docs=args.distractor_docs
    )
    print(f"[EVAL] Distractor tokens: {len(distractor_tokens):,}")
    print(f"[EVAL] Sequence lengths: {args.seq_lens}")
    print(f"[EVAL] Trials per length: {args.n_trials}")

    all_results = {}

    for seq_len in args.seq_lens:
        print(f"\n{'='*60}")
        print(f"  RULER S-NIAH @ seq_len={seq_len}")
        print(f"{'='*60}")

        if len(distractor_tokens) < seq_len:
            print(f"  SKIP: not enough distractor tokens ({len(distractor_tokens)} < {seq_len})")
            continue

        # --- RFT-LM ---
        print(f"\n  [RFT-LM] Loading...")
        torch.cuda.empty_cache()
        rft_model = load_rft_model(args.rft_ckpt, device)

        print(f"  [RFT-LM] Evaluating S-NIAH ({args.n_trials} trials)...")
        t0 = time.time()
        rft_result = eval_niah_model(
            rft_model, tokenizer, distractor_tokens,
            seq_len=seq_len,
            chunk_size=args.chunk_size,
            device=device,
            use_memory=True,
            n_trials=args.n_trials,
            needle_depth=args.needle_depth,
        )
        rft_result["time_s"] = round(time.time() - t0, 1)
        print(f"  [RFT-LM] Accuracy: {rft_result['accuracy']:.1%} "
              f"({rft_result['correct']}/{rft_result['total']}) "
              f"Time: {rft_result['time_s']}s")

        del rft_model
        torch.cuda.empty_cache()

        # --- Baseline ---
        print(f"\n  [Baseline] Loading...")
        bl_model = load_baseline_model(args.baseline_ckpt, device)

        print(f"  [Baseline] Evaluating S-NIAH ({args.n_trials} trials)...")
        t0 = time.time()
        bl_result = eval_niah_model(
            bl_model, tokenizer, distractor_tokens,
            seq_len=seq_len,
            chunk_size=args.chunk_size,
            device=device,
            use_memory=False,
            n_trials=args.n_trials,
            needle_depth=args.needle_depth,
        )
        bl_result["time_s"] = round(time.time() - t0, 1)
        print(f"  [Baseline] Accuracy: {bl_result['accuracy']:.1%} "
              f"({bl_result['correct']}/{bl_result['total']}) "
              f"Time: {bl_result['time_s']}s")

        del bl_model
        torch.cuda.empty_cache()

        # --- Comparison ---
        delta = rft_result["accuracy"] - bl_result["accuracy"]
        winner = "RFT-LM" if delta > 0 else ("Baseline" if delta < 0 else "Tie")

        # Strip per-trial details for summary (keep in full results)
        rft_summary = {k: v for k, v in rft_result.items() if k != "details"}
        bl_summary = {k: v for k, v in bl_result.items() if k != "details"}

        all_results[str(seq_len)] = {
            "rft_lm": rft_result,
            "baseline": bl_result,
            "delta_acc": round(delta, 4),
            "winner": winner,
        }

        print(f"\n  {'─'*50}")
        print(f"  seq_len={seq_len}: {winner} wins")
        print(f"    RFT-LM   acc={rft_result['accuracy']:.1%}")
        print(f"    Baseline acc={bl_result['accuracy']:.1%}")
        print(f"    Δ acc={delta:+.1%}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RULER S-NIAH SUMMARY")
    print(f"{'='*60}")
    print(f"  {'SeqLen':<8} {'RFT-LM':<12} {'Baseline':<12} {'Δ':<10} {'Winner'}")
    print(f"  {'─'*56}")
    for sl in args.seq_lens:
        r = all_results.get(str(sl))
        if r:
            print(f"  {sl:<8} {r['rft_lm']['accuracy']:<12.1%} "
                  f"{r['baseline']['accuracy']:<12.1%} "
                  f"{r['delta_acc']:<+10.1%} {r['winner']}")

    print(f"\n  Reference (Titans paper, 125M scale):")
    print(f"  {'Model':<20} {'4K':<8} {'8K':<8} {'16K':<8} {'32K':<8}")
    print(f"  {'─'*50}")
    print(f"  {'Titans MAC':<20} {'~97%':<8} {'~96%':<8} {'~95%':<8} {'~93%':<8}")
    print(f"  {'Transformer+':<20} {'~92%':<8} {'~85%':<8} {'~70%':<8} {'~50%':<8}")
    print(f"  {'Mamba-2':<20} {'~90%':<8} {'~82%':<8} {'~65%':<8} {'~45%':<8}")

    # Save full results
    outpath = os.path.join(os.path.dirname(args.rft_ckpt), "..", args.outfile)
    # Strip details for JSON (too large), save separately
    summary_results = {}
    for sl, r in all_results.items():
        summary_results[sl] = {
            "rft_lm": {k: v for k, v in r["rft_lm"].items() if k != "details"},
            "baseline": {k: v for k, v in r["baseline"].items() if k != "details"},
            "delta_acc": r["delta_acc"],
            "winner": r["winner"],
        }
    with open(outpath, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[SAVED] {outpath}")

    # Save detailed per-trial results
    detail_path = outpath.replace(".json", "_details.json")
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] {detail_path}")


if __name__ == "__main__":
    main()

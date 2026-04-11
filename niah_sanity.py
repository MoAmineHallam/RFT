"""
niah_sanity.py — Minimal sanity test for NIAH eval pipeline.

Goal: isolate whether 0%/0% accuracy is (a) a pipeline bug or (b) the
125M C4-only models genuinely cannot follow the NIAH prompt template.

Test design (trivially easy):
  - Very short context (e.g. 256 tokens of distractor)
  - Needle placed immediately before the query (depth=1.0)
  - Numeric value, single-key single-value
  - Greedy decode 20 new tokens

If BOTH models get 0 hits here, the problem is prompt/template (models
cannot follow this instruction format from C4 pretraining alone).
If baseline gets a reasonable hit rate, the pipeline works and the full
eval is meaningful.

We also print the raw generations for each trial so you can eyeball what
the model actually produces.
"""

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from RFT_LM import MemoryBank
from eval_ruler_niah import (
    build_mk_niah_sample,
    generate_greedy,
    load_baseline_model,
    load_distractor_tokens,
    load_rft_model,
    string_match_all_binary,
)


def run_sanity(model, tokenizer, distractor_tokens, seq_len, depth, n_trials,
               max_new_tokens, chunk_size, device, use_memory, tag, seed,
               prompt_style="continuation", value_type="numbers",
               mk_num_keys=1, mk_num_queries=1, tail_chunk_len=0):
    hits = 0
    print(f"\n=== {tag} ===")
    for t in range(n_trials):
        sample = build_mk_niah_sample(
            tokenizer=tokenizer,
            seq_len=seq_len,
            distractor_tokens=distractor_tokens,
            seed=seed + t * 7919,
            value_type=value_type,
            needle_depth=depth,
            num_needle_k=mk_num_keys,
            num_needle_v=1,
            num_needle_q=mk_num_queries,
            prompt_style=prompt_style,
        )
        gen_ids = generate_greedy(
            model=model,
            input_ids=sample["input_ids"],
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            use_memory=use_memory,
            tail_chunk_len=tail_chunk_len,
        )
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ok = string_match_all_binary(pred, sample["refs"])
        hits += int(ok)
        # Also check if the needle value appears ANYWHERE in the prediction
        ref_val = sample["refs"][0]
        contains_value = ref_val in pred
        # Show the full decoded context tail + prediction, to see what model saw
        tail_ids = sample["input_ids"][-80:]
        tail_txt = tokenizer.decode(tail_ids, skip_special_tokens=True)
        print(f"  [{t}] ok={ok} contains_val={contains_value} ref={ref_val}")
        print(f"      tail_ctx=...{tail_txt!r}")
        print(f"      pred={pred!r}")
    print(f"  {tag} hits: {hits}/{n_trials}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--distractor_path", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--depth", type=float, default=1.0,
                    help="1.0 = needle placed right before the query")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--distractor_docs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt_style", type=str, choices=["instruct", "continuation"],
                    default="continuation")
    ap.add_argument("--value_type", type=str, choices=["numbers", "short_int", "digit2", "uuids"],
                    default="digit2")
    ap.add_argument("--mk_num_keys", type=int, default=3)
    ap.add_argument("--mk_num_queries", type=int, default=1)
    ap.add_argument("--tail_chunk_len", type=int, default=0,
                    help="If >0, process last N prompt tokens as a separate mini-chunk "
                         "AFTER body is written to memory. Fixes depth=1.0.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9
    print(f"[SANITY] tokenizer eos_id={tokenizer.eos_token_id}")
    print(f"[SANITY] seq_len={args.seq_len} depth={args.depth} n_trials={args.n_trials}")

    distractor_tokens = load_distractor_tokens(args.distractor_path, tokenizer, args.distractor_docs)
    print(f"[SANITY] distractor tokens: {len(distractor_tokens):,}")

    # Test baseline first
    baseline = load_baseline_model(args.baseline_ckpt, device)
    baseline.eval()
    bl_hits = run_sanity(
        baseline, tokenizer, distractor_tokens, args.seq_len, args.depth,
        args.n_trials, args.max_new_tokens, args.chunk_size, device,
        use_memory=False, tag="BASELINE", seed=args.seed,
        prompt_style=args.prompt_style,
        value_type=args.value_type,
        mk_num_keys=args.mk_num_keys,
        mk_num_queries=args.mk_num_queries,
        tail_chunk_len=0,  # baseline has no memory, split is a no-op
    )
    del baseline
    torch.cuda.empty_cache()

    # Test RFT
    rft = load_rft_model(args.rft_ckpt, device)
    rft.eval()
    rft_hits = run_sanity(
        rft, tokenizer, distractor_tokens, args.seq_len, args.depth,
        args.n_trials, args.max_new_tokens, args.chunk_size, device,
        use_memory=True, tag="RFT-LM", seed=args.seed,
        prompt_style=args.prompt_style,
        value_type=args.value_type,
        mk_num_keys=args.mk_num_keys,
        mk_num_queries=args.mk_num_queries,
        tail_chunk_len=args.tail_chunk_len,
    )
    del rft
    torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"  SANITY RESULT: baseline={bl_hits}/{args.n_trials}  rft={rft_hits}/{args.n_trials}")
    print("=" * 72)
    if bl_hits == 0 and rft_hits == 0:
        print("  DIAGNOSIS: neither model produced the needle value.")
        print("  This is NOT necessarily a pipeline bug — 125M C4-only models")
        print("  may simply not follow this NIAH prompt template. Check the")
        print("  printed `pred` strings above to see what the model actually")
        print("  generated. If it's coherent continuation (but wrong content),")
        print("  the architecture is fine; the task is OOD for the pretraining.")
    elif bl_hits > 0:
        print("  DIAGNOSIS: pipeline works. Full eval is meaningful.")


if __name__ == "__main__":
    main()

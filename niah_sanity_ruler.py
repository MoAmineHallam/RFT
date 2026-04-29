"""
niah_sanity_ruler.py — RULER-format sanity test matching v12 training distribution.

Differences from niah_sanity.py:
  - Keys are adjective-noun pairs from english_words.json (matches v12 training)
  - Haystack is Paul Graham Essays (matches v12 training)
  - Needle/query strings match niah_batch.py exactly

Reports:
  - Full-match (contains_val substring check, RULER official metric)
  - First-BPE-token accuracy (diagnostic for memory retrieval signal)
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eval_ruler_niah import (
    generate_greedy,
    load_baseline_model,
    load_rft_model,
    string_match_all_binary,
    generate_value,
)


def load_pg_distractor_tokens(path: str, tokenizer, max_chars: int = None):
    """Paul Graham essays is a single {'text': '...'} JSON, not jsonl."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        blob = obj.get("text", "")
    elif isinstance(obj, str):
        blob = obj
    else:
        blob = str(obj)
    if max_chars is not None:
        blob = blob[:max_chars]
    return tokenizer.encode(blob, add_special_tokens=False)


def load_wordlist(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.values())
    return list(obj)


def build_ruler_sample(
    tokenizer,
    wordlist,
    distractor_tokens,
    seq_len: int,
    seed: int,
    value_type: str,
    needle_depth,
    num_needle_k: int,
    num_needle_q: int,
    chunk_size: int = 512,
    add_instruction: bool = False,
):
    """
    Two-chunk layout matching niah_batch.py training exactly:
      chunk 1 (ctx, size=chunk_size): distractor with needles interleaved
      chunk 2 (qry, size=chunk_size): padding-distractor + query + answer prefix

    Total returned input length = 2 * chunk_size (plus optional trailing trim).
    add_instruction defaults to False since most training samples had no prefix.
    """
    rng = random.Random(seed)
    num_needle_k = max(num_needle_k, num_needle_q)

    keys = [f"{rng.choice(wordlist)}-{rng.choice(wordlist)}" for _ in range(num_needle_k)]
    values = [generate_value(rng, value_type) for _ in range(num_needle_k)]

    _pl = {"numbers": "numbers", "short_int": "numbers", "digit2": "numbers",
           "words": "words", "uuids": "uuids"}.get(value_type, value_type)
    _sg = {"numbers": "number", "short_int": "number", "digit2": "number",
           "words": "word", "uuids": "uuid"}.get(value_type, value_type)

    needle_sentences = [
        f" The special magic {_sg} for {k} is: {v}."
        for k, v in zip(keys, values)
    ]
    needle_tok_lists = [tokenizer.encode(s, add_special_tokens=False) for s in needle_sentences]

    q_idx = rng.sample(range(num_needle_k), k=num_needle_q)
    query_keys = [keys[i] for i in q_idx]
    ref_values = [values[i] for i in q_idx]

    if len(query_keys) == 1:
        query_str = query_keys[0]
    elif len(query_keys) == 2:
        query_str = f"{query_keys[0]} and {query_keys[1]}"
    else:
        query_str = ", ".join(query_keys[:-1]) + f", and {query_keys[-1]}"

    # ---- Chunk 1: ctx (distractor + needles interleaved) ----
    if add_instruction:
        instr_str = (
            f"Some special magic {_pl} are hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about the {_pl} afterwards.\n"
        )
        instr_toks = tokenizer.encode(instr_str, add_special_tokens=False)
    else:
        instr_toks = []

    total_needle_len = sum(len(t) for t in needle_tok_lists)
    ctx_dist_budget = max(16, chunk_size - len(instr_toks) - total_needle_len)

    max_start = max(0, len(distractor_tokens) - ctx_dist_budget - 1)
    d_start = rng.randint(0, max_start) if max_start > 0 else 0
    ctx_dist = distractor_tokens[d_start:d_start + ctx_dist_budget]
    while len(ctx_dist) < ctx_dist_budget:
        extra_start = rng.randint(0, max(0, len(distractor_tokens) - 128))
        ctx_dist = ctx_dist + distractor_tokens[extra_start:extra_start + 128]
    ctx_dist = ctx_dist[:ctx_dist_budget]

    # Place the query-target needle at `needle_depth` of the context budget.
    # Distribute the remaining (distractor) needles uniformly around it.
    n_needles = len(needle_tok_lists)
    dist_len = len(ctx_dist)
    # Which needle index is the query target? q_idx[0] was chosen above.
    target_ni = q_idx[0] if num_needle_q >= 1 else 0
    depth_val = needle_depth if needle_depth is not None else 0.5
    depth_val = max(0.0, min(1.0, float(depth_val)))
    target_pos = int(depth_val * max(0, dist_len - 1))

    # Assign positions: target needle at target_pos, others evenly but avoiding overlap
    other_indices = [i for i in range(n_needles) if i != target_ni]
    n_other = len(other_indices)
    # Divide distractor into n_other+1 segments, place others at segment boundaries
    positions = {target_ni: target_pos}
    if n_other > 0:
        step = dist_len // (n_other + 1)
        for k, idx in enumerate(other_indices):
            p = (k + 1) * step
            # Push away from target_pos if too close
            if abs(p - target_pos) < 30:
                p = (p + step // 2) % dist_len
            positions[idx] = p

    # Sort needles by position and interleave with distractor
    sorted_ni = sorted(range(n_needles), key=lambda i: positions[i])
    ctx_toks = list(instr_toks)
    cursor = 0
    for ni in sorted_ni:
        p = max(cursor, min(positions[ni], dist_len))
        ctx_toks.extend(ctx_dist[cursor:p])
        ctx_toks.extend(needle_tok_lists[ni])
        cursor = p
    ctx_toks.extend(ctx_dist[cursor:])

    # Truncate or pad ctx to chunk_size
    ctx_toks = ctx_toks[:chunk_size]
    if len(ctx_toks) < chunk_size:
        pad_start = rng.randint(0, max(0, len(distractor_tokens) - 128))
        ctx_toks.extend(distractor_tokens[pad_start:pad_start + (chunk_size - len(ctx_toks))])
        ctx_toks = ctx_toks[:chunk_size]

    # ---- Chunk 2: qry (padding-distractor + query + answer prefix) ----
    query_full_str = (
        f"\nWhat are all the special magic {_pl} for {query_str} "
        f"mentioned in the provided text? "
        f"The special magic {_pl} for {query_str} mentioned in "
        f"the provided text are"
    )
    query_toks = tokenizer.encode(query_full_str, add_special_tokens=False)

    qry_dist_budget = max(16, chunk_size - len(query_toks))
    q_start = rng.randint(0, max(0, len(distractor_tokens) - qry_dist_budget - 1))
    qry_dist = distractor_tokens[q_start:q_start + qry_dist_budget]
    while len(qry_dist) < qry_dist_budget:
        extra_start = rng.randint(0, max(0, len(distractor_tokens) - 128))
        qry_dist = qry_dist + distractor_tokens[extra_start:extra_start + 128]
    qry_dist = qry_dist[:qry_dist_budget]

    qry_toks = list(qry_dist) + list(query_toks)
    qry_toks = qry_toks[:chunk_size]
    if len(qry_toks) < chunk_size:
        # Pad at the front with extra distractor
        pad_needed = chunk_size - len(qry_toks)
        pad_start = rng.randint(0, max(0, len(distractor_tokens) - pad_needed))
        qry_toks = list(distractor_tokens[pad_start:pad_start + pad_needed]) + qry_toks
        qry_toks = qry_toks[:chunk_size]

    input_ids = ctx_toks + qry_toks
    return {
        "input_ids": input_ids,
        "refs": ref_values,
        "query_keys": query_keys,
    }


def run_sanity(model, tokenizer, wordlist, distractor_tokens, args, use_memory, tag):
    hits_full = 0
    hits_firsttok = 0
    print(f"\n=== {tag} ===")
    for t in range(args.n_trials):
        sample = build_ruler_sample(
            tokenizer=tokenizer,
            wordlist=wordlist,
            distractor_tokens=distractor_tokens,
            seq_len=args.seq_len,
            seed=args.seed + t * 7919,
            value_type=args.value_type,
            needle_depth=args.depth,
            num_needle_k=args.mk_num_keys,
            num_needle_q=args.mk_num_queries,
        )
        gen_ids = generate_greedy(
            model=model,
            input_ids=sample["input_ids"],
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            device=next(model.parameters()).device,
            use_memory=use_memory,
            tail_chunk_len=args.tail_chunk_len,
        )
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ref_val = sample["refs"][0]

        ok = string_match_all_binary(pred, sample["refs"])
        contains_value = ref_val in pred
        hits_full += int(ok)

        # First-token accuracy
        ref_first_tok_ids = tokenizer.encode(" " + ref_val, add_special_tokens=False)
        ref_first_decoded = tokenizer.decode([ref_first_tok_ids[0]]).strip()
        first_ok = pred.lstrip().startswith(ref_first_decoded)
        hits_firsttok += int(first_ok)

        tail_ids = sample["input_ids"][-80:]
        tail_txt = tokenizer.decode(tail_ids, skip_special_tokens=True)
        mark_full = "✅" if ok else "❌"
        mark_first = "✅" if first_ok else "❌"
        print(f"  [{t}] full={mark_full} first_tok={mark_first} contains={contains_value} "
              f"ref={ref_val} (first_tok='{ref_first_decoded}') key={sample['query_keys'][0]}")
        print(f"      tail_ctx=...{tail_txt!r}")
        print(f"      pred={pred!r}")

    print(f"  {tag} full hits: {hits_full}/{args.n_trials}")
    print(f"  {tag} first-token hits: {hits_firsttok}/{args.n_trials}")
    return hits_full, hits_firsttok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, default=None,
                    help="Optional baseline for comparison.")
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--distractor_path", type=str, default="PaulGrahamEssays.json")
    ap.add_argument("--wordlist_path", type=str, default="english_words.json")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--value_type", type=str, default="numbers")
    ap.add_argument("--mk_num_keys", type=int, default=3)
    ap.add_argument("--mk_num_queries", type=int, default=1)
    ap.add_argument("--tail_chunk_len", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9
    print(f"[SANITY-RULER] tokenizer eos_id={tokenizer.eos_token_id}")
    print(f"[SANITY-RULER] seq_len={args.seq_len} depth={args.depth} "
          f"n_trials={args.n_trials} mk_num_keys={args.mk_num_keys}")

    # Load wordlist
    wordlist = load_wordlist(args.wordlist_path)
    print(f"[SANITY-RULER] wordlist size: {len(wordlist):,}")

    # Load PG distractor
    # Limit to ~1M chars (~250k tokens) to keep loading fast
    distractor_tokens = load_pg_distractor_tokens(args.distractor_path, tokenizer, max_chars=1_500_000)
    print(f"[SANITY-RULER] distractor tokens (Paul Graham): {len(distractor_tokens):,}")

    # Test baseline if provided
    if args.baseline_ckpt:
        baseline = load_baseline_model(args.baseline_ckpt, device)
        baseline.eval()
        bl_full, bl_first = run_sanity(baseline, tokenizer, wordlist, distractor_tokens,
                                        args, use_memory=False, tag="BASELINE")
        del baseline
        torch.cuda.empty_cache()
    else:
        bl_full, bl_first = -1, -1

    # Test RFT
    rft = load_rft_model(args.rft_ckpt, device)
    rft.eval()
    rft_full, rft_first = run_sanity(rft, tokenizer, wordlist, distractor_tokens,
                                      args, use_memory=True, tag="RFT-LM")
    del rft
    torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"  RULER-FORMAT SANITY RESULT")
    if args.baseline_ckpt:
        print(f"  Baseline : full={bl_full}/{args.n_trials}  first_tok={bl_first}/{args.n_trials}")
    print(f"  RFT-LM   : full={rft_full}/{args.n_trials}  first_tok={rft_first}/{args.n_trials}")
    print("=" * 72)
    if rft_full >= args.n_trials // 2:
        print("  VERDICT: Strong retrieval. v12 training is on track. ✅")
    elif rft_first >= args.n_trials // 2:
        print("  VERDICT: Memory fires (first-tok good), but generation fails after.")
        print("           Check generate_greedy fix / multi-BPE value handling.")
    elif rft_first > 3:
        print("  VERDICT: Weak signal. Model is learning but needs more training.")
    else:
        print("  VERDICT: No retrieval signal. Training may have an issue.")


if __name__ == "__main__":
    main()
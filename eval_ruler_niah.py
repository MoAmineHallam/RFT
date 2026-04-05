"""
eval_ruler_niah.py — Paper-grade RULER-style NIAH evaluator for local checkpoints.

Adds:
  - Depth-stratified sweep (single-depth or grid)
  - Multi-key / multi-needle NIAH (MK-NIAH)
  - Wilson confidence intervals
  - RFT ablation hooks (disable memory, neutralize OCR influence)
"""

import argparse
import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from RFT_LM import BaselineTransformerLM, MemoryBank, RFTLM


def string_match_all_binary(pred: str, refs: List[str]) -> bool:
    if not refs:
        return False
    p = pred.lower()
    return all(r.lower() in p for r in refs)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def load_distractor_tokens(path: str, tokenizer, max_docs: int) -> List[int]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            obj = json.loads(line)
            txt = obj.get("text", "")
            if txt:
                texts.append(txt)
    blob = " ".join(texts)
    return tokenizer.encode(blob, add_special_tokens=False)


def generate_value(rng: random.Random, value_type: str) -> str:
    if value_type == "numbers":
        return str(rng.randint(1_000_000, 9_999_999))
    if value_type == "short_int":
        # 3-digit int: usually tokenizes to 1 BPE piece with leading space in GPT-2
        return str(rng.randint(100, 999))
    if value_type == "digit2":
        # 2-digit int (10-99): reliably 1 BPE token with leading space in GPT-2.
        # Means one correct token = full match, avoiding multi-token generation failure.
        return str(rng.randint(10, 99))
    if value_type == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise ValueError(f"Unsupported value_type={value_type}")


def build_mk_niah_sample(
    tokenizer,
    seq_len: int,
    distractor_tokens: List[int],
    seed: int,
    value_type: str,
    needle_depth: Optional[float],
    num_needle_k: int,
    num_needle_v: int,
    num_needle_q: int,
    prompt_style: str = "instruct",
) -> Dict:
    rng = random.Random(seed)
    num_needle_k = max(num_needle_k, num_needle_q)

    keys = [f"alpha{rng.randint(0, 99999):05d}{i}" for i in range(num_needle_k)]
    values_by_key: List[List[str]] = []
    needle_sentences = []

    # continuation style: bare assertion; instruct style: labeled sentence
    if prompt_style == "continuation":
        for k in keys:
            vals = [generate_value(rng, value_type) for _ in range(num_needle_v)]
            values_by_key.append(vals)
            for v in vals:
                needle_sentences.append(f"The magic number {k} is {v}.")
    else:
        for k in keys:
            vals = [generate_value(rng, value_type) for _ in range(num_needle_v)]
            values_by_key.append(vals)
            for v in vals:
                needle_sentences.append(f"One of the special magic {value_type} for {k} is: {v}.")

    # Query subset
    q_idx = rng.sample(range(num_needle_k), k=num_needle_q)
    query_keys = [keys[i] for i in q_idx]
    ref_values = [v for i in q_idx for v in values_by_key[i]]

    if len(query_keys) == 1:
        query_str = query_keys[0]
    elif len(query_keys) == 2:
        query_str = f"{query_keys[0]} and {query_keys[1]}"
    else:
        query_str = ", ".join(query_keys[:-1]) + f", and {query_keys[-1]}"

    if prompt_style == "continuation":
        # No instruction. Just a bare probe that the model must complete.
        # Pattern in context: "The magic number K is V." -> probe "The magic number K is"
        prompt_prefix = ""
        prompt_suffix = f"\nThe magic number {query_str} is"
    else:
        prompt_prefix = (
            f"Some special magic {value_type} are hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about the {value_type} afterwards.\n"
        )
        prompt_suffix = (
            f"\nWhat are all the special magic {value_type} for {query_str} mentioned in the provided text?"
            f" The special magic {value_type} for {query_str} mentioned in the provided text are"
        )
    prefix_toks = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    suffix_toks = tokenizer.encode(prompt_suffix, add_special_tokens=False)
    needle_toks = [tokenizer.encode(" " + s + " ", add_special_tokens=False) for s in needle_sentences]
    needles_total = sum(len(t) for t in needle_toks)

    context_budget = seq_len - len(prefix_toks) - len(suffix_toks) - needles_total
    if context_budget < 128:
        raise ValueError("Sequence too short for current MK-NIAH settings")

    max_start = max(0, len(distractor_tokens) - context_budget - 1)
    dist_start = rng.randint(0, max_start) if max_start > 0 else 0
    distractor_chunk = distractor_tokens[dist_start:dist_start + context_budget]
    while len(distractor_chunk) < context_budget:
        extra_start = rng.randint(0, max(0, len(distractor_tokens) - 128)) if distractor_tokens else 0
        distractor_chunk.extend(distractor_tokens[extra_start:extra_start + 128])
    distractor_chunk = distractor_chunk[:context_budget]

    if needle_depth is None:
        depth_positions = sorted(
            rng.randint(max(1, context_budget // 10), max(2, context_budget - context_budget // 10))
            for _ in needle_toks
        )
    else:
        base = int(max(0.0, min(1.0, needle_depth)) * context_budget)
        depth_positions = [min(context_budget - 1, max(0, base + i)) for i in range(len(needle_toks))]

    # Interleave needles into distractor by position
    context = []
    cursor = 0
    positions_frac = []
    for pos, nt in sorted(zip(depth_positions, needle_toks), key=lambda x: x[0]):
        pos = max(cursor, min(pos, context_budget))
        context.extend(distractor_chunk[cursor:pos])
        context.extend(nt)
        positions_frac.append(pos / max(context_budget, 1))
        cursor = pos
    context.extend(distractor_chunk[cursor:])

    input_ids = (prefix_toks + context + suffix_toks)[:seq_len]
    return {
        "input_ids": input_ids,
        "refs": ref_values,
        "needle_pos_fracs": positions_frac,
        "query_keys": query_keys,
    }


def maybe_apply_rft_ablation(model: RFTLM, mode: str):
    # mode: none | disable_memory | disable_ocr
    if mode == "none":
        return
    if mode == "disable_memory":
        model.use_memory = False
        return
    if mode == "disable_ocr":
        # Neutralize OCR contribution in retrieval logits.
        if hasattr(model, "memory_layer") and hasattr(model.memory_layer, "ocr_alpha"):
            with torch.no_grad():
                model.memory_layer.ocr_alpha.fill_(0.0)
        return
    raise ValueError(f"Unknown rft_ablation={mode}")


def _forward_chunk(model, x, pos_offset, use_memory, memory_bank, update_memory):
    if use_memory:
        mem_k, mem_v, mem_pos = memory_bank.get_state()
        out = model(
            x,
            memory_keys=mem_k,
            memory_vals=mem_v,
            memory_positions=mem_pos,
            pos_offset=pos_offset,
            return_memory_state=update_memory,
        )
        if update_memory and "new_mem_keys" in out:
            memory_bank.add(out["new_mem_keys"].detach(), out["new_mem_vals"].detach(), pos_offset, x.shape[1])
        return out
    return model(x, pos_offset=pos_offset)


@torch.no_grad()
def generate_greedy(
    model,
    input_ids: List[int],
    chunk_size: int,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    device: torch.device,
    use_memory: bool,
) -> List[int]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    total_len = ids.shape[1]
    mb = MemoryBank(max_entries=65536) if use_memory else None
    if mb is not None:
        mb.reset()

    last_logits = None
    for s in range(0, total_len, chunk_size):
        e = min(total_len, s + chunk_size)
        out = _forward_chunk(model, ids[:, s:e], s, use_memory, mb, True)
        last_logits = out["logits"]
    if last_logits is None:
        return []

    logits = last_logits[:, -1, :]
    out_ids = []
    for i in range(max_new_tokens):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        tid = int(nxt.item())
        out_ids.append(tid)
        if eos_token_id is not None and tid == eos_token_id:
            break
        out = _forward_chunk(model, nxt, total_len + i, use_memory, mb, True)
        logits = out["logits"][:, -1, :]
    return out_ids


def eval_at_depth(
    model,
    tokenizer,
    seq_len: int,
    depth: Optional[float],
    distractor_tokens: List[int],
    chunk_size: int,
    n_trials: int,
    max_new_tokens: int,
    seed: int,
    value_type: str,
    mk_num_keys: int,
    mk_num_values: int,
    mk_num_queries: int,
    device: torch.device,
    use_memory: bool,
    prompt_style: str = "instruct",
) -> Dict:
    hits = 0
    preds, refs, details = [], [], []
    for t in range(n_trials):
        sample = build_mk_niah_sample(
            tokenizer=tokenizer,
            seq_len=seq_len,
            distractor_tokens=distractor_tokens,
            seed=seed + seq_len * 100_003 + t * 7_919,
            value_type=value_type,
            needle_depth=depth,
            num_needle_k=mk_num_keys,
            num_needle_v=mk_num_values,
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
        )
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ref = sample["refs"]
        ok = string_match_all_binary(pred, ref)
        hits += int(ok)
        preds.append(pred)
        refs.append(ref)
        details.append(
            {
                "trial": t,
                "correct": ok,
                "prediction": pred,
                "refs": ref,
                "needle_pos_fracs": sample["needle_pos_fracs"],
                "query_keys": sample["query_keys"],
            }
        )
        if (t + 1) % 20 == 0:
            print(f"    trial {t+1}/{n_trials}: acc={100.0 * hits / (t+1):.2f}%")

    lo, hi = wilson_ci(hits, n_trials)
    return {
        "correct": hits,
        "total": n_trials,
        "accuracy": hits / max(n_trials, 1),
        "ci95_low": lo,
        "ci95_high": hi,
        "details": details,
    }


def load_rft_model(path: str, device: torch.device) -> RFTLM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
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
    sd = ck["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def load_baseline_model(path: str, device: torch.device) -> BaselineTransformerLM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
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
    sd = ck["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def make_depth_grid(arg_depth_grid: Optional[List[float]], arg_depth: Optional[float]) -> List[Optional[float]]:
    if arg_depth_grid:
        return [max(0.0, min(1.0, d)) for d in arg_depth_grid]
    if arg_depth is not None:
        return [max(0.0, min(1.0, arg_depth))]
    # Paper-friendly default sweep
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--distractor_path", type=str, required=True)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_trials", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--needle_depth", type=float, default=None)
    ap.add_argument("--needle_depth_grid", type=float, nargs="+", default=None)
    ap.add_argument("--value_type", type=str, choices=["numbers", "uuids"], default="numbers")
    ap.add_argument("--distractor_docs", type=int, default=2000)
    ap.add_argument("--mk_num_keys", type=int, default=1)
    ap.add_argument("--mk_num_values", type=int, default=1)
    ap.add_argument("--mk_num_queries", type=int, default=1)
    ap.add_argument("--rft_ablation", type=str, choices=["none", "disable_memory", "disable_ocr"], default="none")
    ap.add_argument("--prompt_style", type=str, choices=["instruct", "continuation"], default="continuation",
                    help="continuation: bare probe for base LMs. instruct: RULER-style template.")
    ap.add_argument("--outfile", type=str, default="niah_results.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9

    depths = make_depth_grid(args.needle_depth_grid, args.needle_depth)

    print(f"[EVAL] Device: {device}")
    print(f"[EVAL] Depths: {depths}")
    print(f"[EVAL] MK-NIAH: keys={args.mk_num_keys}, values/key={args.mk_num_values}, queried_keys={args.mk_num_queries}")
    print(f"[EVAL] RFT ablation mode: {args.rft_ablation}")

    distractor_tokens = load_distractor_tokens(args.distractor_path, tokenizer, args.distractor_docs)
    print(f"[EVAL] Distractor tokens: {len(distractor_tokens):,}")

    all_results = {}
    for sl in args.seq_lens:
        print(f"\n{'='*72}\n  seq_len={sl}\n{'='*72}")
        all_results[str(sl)] = {}

        print("  [RFT-LM] loading...")
        rft = load_rft_model(args.rft_ckpt, device)
        maybe_apply_rft_ablation(rft, args.rft_ablation)

        print("  [Baseline] loading...")
        baseline = load_baseline_model(args.baseline_ckpt, device)

        for depth in depths:
            depth_key = "random" if depth is None else f"{depth:.2f}"
            print(f"\n  --- depth={depth_key} ---")

            rft_res = eval_at_depth(
                model=rft,
                tokenizer=tokenizer,
                seq_len=sl,
                depth=depth,
                distractor_tokens=distractor_tokens,
                chunk_size=args.chunk_size,
                n_trials=args.n_trials,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                value_type=args.value_type,
                mk_num_keys=args.mk_num_keys,
                mk_num_values=args.mk_num_values,
                mk_num_queries=args.mk_num_queries,
                prompt_style=args.prompt_style,
                device=device,
                use_memory=(args.rft_ablation != "disable_memory"),
            )
            print(
                f"    RFT-LM: {100*rft_res['accuracy']:.2f}% "
                f"[{100*rft_res['ci95_low']:.2f}, {100*rft_res['ci95_high']:.2f}]"
            )

            bl_res = eval_at_depth(
                model=baseline,
                tokenizer=tokenizer,
                seq_len=sl,
                depth=depth,
                distractor_tokens=distractor_tokens,
                chunk_size=args.chunk_size,
                n_trials=args.n_trials,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                value_type=args.value_type,
                mk_num_keys=args.mk_num_keys,
                mk_num_values=args.mk_num_values,
                mk_num_queries=args.mk_num_queries,
                prompt_style=args.prompt_style,
                device=device,
                use_memory=False,
            )
            print(
                f"    Baseline: {100*bl_res['accuracy']:.2f}% "
                f"[{100*bl_res['ci95_low']:.2f}, {100*bl_res['ci95_high']:.2f}]"
            )

            all_results[str(sl)][depth_key] = {
                "rft_lm": rft_res,
                "baseline": bl_res,
                "delta_acc": rft_res["accuracy"] - bl_res["accuracy"],
            }

        del rft, baseline
        torch.cuda.empty_cache()

    print(f"\n{'='*72}\n  HEATMAP TABLE (accuracy %)\n{'='*72}")
    for sl in args.seq_lens:
        row = []
        for depth in depths:
            key = "random" if depth is None else f"{depth:.2f}"
            rec = all_results[str(sl)][key]
            row.append(
                f"d={key}: RFT {100*rec['rft_lm']['accuracy']:.1f} | "
                f"Base {100*rec['baseline']['accuracy']:.1f}"
            )
        print(f"  L={sl}: " + " || ".join(row))

    print("\n[NOTE] For fair paper framing, compare against published Titans/Mamba numbers externally using matching task config.")

    out_dir = str(Path(args.rft_ckpt).resolve().parent.parent)
    out_path = os.path.join(out_dir, args.outfile)

    summary = {
        "config": vars(args),
        "results": {
            sl: {
                d: {
                    "rft_lm": {k: v for k, v in rec["rft_lm"].items() if k != "details"},
                    "baseline": {k: v for k, v in rec["baseline"].items() if k != "details"},
                    "delta_acc": rec["delta_acc"],
                }
                for d, rec in depth_map.items()
            }
            for sl, depth_map in all_results.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {out_path}")

    detail_path = out_path.replace(".json", "_details.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"[SAVED] {detail_path}")


if __name__ == "__main__":
    main()


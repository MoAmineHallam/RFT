"""
eval_vanilla_hf.py — Run NIAH evaluation on a vanilla HuggingFace causal LM.

Used as the comparison baseline for RFTGraftLM. Same NIAH protocol as
eval_ruler_niah.py (build_mk_niah_sample, string_match_all_binary), so
vanilla-vs-graft numbers are directly comparable.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ruler_niah import (
    build_mk_niah_sample,
    load_distractor_tokens,
    string_match_all_binary,
    wilson_ci,
)


@torch.no_grad()
def generate_hf_greedy(model, tokenizer, input_ids, max_new_tokens, device):
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos = tokenizer.eos_token_id
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=eos if eos is not None else 0,
        use_cache=True,
    )
    return out[0, ids.shape[1]:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True,
                    help="HF model name or local path, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--distractor_path", type=str, required=True)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--needle_depth", type=float, default=0.5)
    ap.add_argument("--needle_depth_grid", type=float, nargs="+", default=None)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--value_type", type=str,
                    choices=["numbers", "short_int", "digit2", "uuids"],
                    default="numbers")
    ap.add_argument("--distractor_docs", type=int, default=300)
    ap.add_argument("--mk_num_keys", type=int, default=3)
    ap.add_argument("--mk_num_values", type=int, default=1)
    ap.add_argument("--mk_num_queries", type=int, default=1)
    ap.add_argument("--prompt_style", type=str,
                    choices=["instruct", "continuation"], default="instruct")
    ap.add_argument("--outfile", type=str, default="vanilla_hf_results.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[VANILLA] loading {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tokenizer.model_max_length = 10**9
    model = AutoModelForCausalLM.from_pretrained(args.base).to(device).eval()

    distractor_tokens = load_distractor_tokens(
        args.distractor_path, tokenizer, args.distractor_docs
    )
    print(f"[VANILLA] {len(distractor_tokens):,} distractor tokens")

    depths = (
        args.needle_depth_grid
        if args.needle_depth_grid
        else [args.needle_depth]
    )

    all_results = {}
    for sl in args.seq_lens:
        all_results[str(sl)] = {}
        for depth in depths:
            print(f"\n  --- seq_len={sl} depth={depth:.2f} ---")
            hits_full = 0
            hits_first = 0
            details = []
            for t in range(args.n_trials):
                sample = build_mk_niah_sample(
                    tokenizer=tokenizer,
                    seq_len=sl,
                    distractor_tokens=distractor_tokens,
                    seed=args.seed + sl * 100_003 + t * 7_919,
                    value_type=args.value_type,
                    needle_depth=depth,
                    num_needle_k=args.mk_num_keys,
                    num_needle_v=args.mk_num_values,
                    num_needle_q=args.mk_num_queries,
                    prompt_style=args.prompt_style,
                )
                gen_ids = generate_hf_greedy(
                    model, tokenizer, sample["input_ids"],
                    args.max_new_tokens, device,
                )
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                ok_full = string_match_all_binary(pred, sample["refs"])
                ref0 = sample["refs"][0] if sample["refs"] else ""
                first_tok_ref = tokenizer.encode(
                    f" {ref0}", add_special_tokens=False
                )[:1]
                first_tok_pred = gen_ids[:1]
                ok_first = (first_tok_pred == first_tok_ref) if first_tok_ref else False
                hits_full += int(ok_full)
                hits_first += int(ok_first)
                details.append({
                    "trial": t, "ok_full": ok_full, "ok_first": ok_first,
                    "pred": pred, "refs": sample["refs"],
                })
                if (t + 1) % 20 == 0:
                    print(f"    trial {t+1}/{args.n_trials}: "
                          f"full={hits_full}/{t+1} first={hits_first}/{t+1}")

            lo_f, hi_f = wilson_ci(hits_full, args.n_trials)
            lo_t, hi_t = wilson_ci(hits_first, args.n_trials)
            print(f"  vanilla {args.base}: "
                  f"full={hits_full}/{args.n_trials} [{100*lo_f:.1f},{100*hi_f:.1f}] "
                  f"first={hits_first}/{args.n_trials} [{100*lo_t:.1f},{100*hi_t:.1f}]")
            all_results[str(sl)][f"{depth:.2f}"] = {
                "full_correct": hits_full,
                "first_correct": hits_first,
                "total": args.n_trials,
                "full_accuracy": hits_full / max(args.n_trials, 1),
                "first_accuracy": hits_first / max(args.n_trials, 1),
                "details": details,
            }

    out_path = args.outfile
    summary = {
        "config": vars(args),
        "results": {
            sl: {d: {k: v for k, v in r.items() if k != "details"}
                 for d, r in dm.items()}
            for sl, dm in all_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    detail_path = out_path.replace(".json", "_details.json")
    with open(detail_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"[SAVED] {detail_path}")


if __name__ == "__main__":
    main()

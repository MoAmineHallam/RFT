"""
Evaluate RFT-LM / baseline checkpoints on RULER S-NIAH (Single Needle-In-A-Haystack).

Expected data format: jsonl with at least one context field, one question field, and one answer field.
This script accepts common aliases used by long-context datasets.
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from RFT_LM import RFTLM, BaselineTransformerLM, MemoryBank


CTX_KEYS = ["context", "input", "prompt", "document", "haystack", "text"]
Q_KEYS = ["question", "query", "instruction"]
A_KEYS = ["answer", "target", "needle", "gold"]
LEN_KEYS = ["context_length", "ctx_len", "length", "n_tokens"]


def _pick(example: Dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in example and example[k] is not None:
            return str(example[k])
    return default


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\-\._ ]", "", s)
    return s


def build_prompt(ctx: str, q: str) -> str:
    return (
        "Read the context and answer the question with only the exact answer.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )


def load_examples(path: str, max_examples: int = None) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            ex = json.loads(line)
            ctx = _pick(ex, CTX_KEYS)
            q = _pick(ex, Q_KEYS)
            a = _pick(ex, A_KEYS)
            if not ctx or not q or not a:
                continue
            ctx_len = None
            for lk in LEN_KEYS:
                if lk in ex:
                    try:
                        ctx_len = int(ex[lk])
                    except Exception:
                        pass
            items.append({"context": ctx, "question": q, "answer": a, "context_length": ctx_len})
    return items


def load_rft_model(ckpt_path: str, device: torch.device) -> Tuple[RFTLM, Dict]:
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
        ocr_alpha_init=a.get("ocr_alpha_init", 1e-2),
        ocr_margin=a.get("ocr_margin", 0.10),
        mem_gate_alpha_init=a.get("mem_gate_alpha_init", 1.0),
    ).to(device)
    sd = ck["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, a


def load_baseline_model(ckpt_path: str, device: torch.device) -> Tuple[BaselineTransformerLM, Dict]:
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
    sd = ck["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, a


@torch.no_grad()
def generate_rft(model: RFTLM, input_ids: torch.Tensor, max_new_tokens: int, chunk_size: int, eos_id: int) -> torch.Tensor:
    device = input_ids.device
    mem = MemoryBank(max_entries=131072)
    mem.reset()

    total_len = input_ids.shape[1]
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        chunk = input_ids[:, start:end]
        mk, mv, mp = mem.get_state()
        out = model(chunk, memory_keys=mk, memory_vals=mv, memory_positions=mp, pos_offset=start, return_memory_state=True)
        if "new_mem_keys" in out:
            mem.add(out["new_mem_keys"].detach(), out["new_mem_vals"].detach(), start, chunk.shape[1])

    cur = input_ids[:, -1:]
    pos = total_len
    generated = []
    for _ in range(max_new_tokens):
        mk, mv, mp = mem.get_state()
        out = model(cur, memory_keys=mk, memory_vals=mv, memory_positions=mp, pos_offset=pos, return_memory_state=True)
        nxt = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(nxt)
        if "new_mem_keys" in out:
            mem.add(out["new_mem_keys"].detach(), out["new_mem_vals"].detach(), pos, 1)
        cur = nxt
        pos += 1
        if eos_id is not None and int(nxt.item()) == int(eos_id):
            break
    return torch.cat(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)


@torch.no_grad()
def generate_baseline(model: BaselineTransformerLM, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
    cur = input_ids.clone()
    generated = []
    for _ in range(max_new_tokens):
        out = model(cur, pos_offset=0)
        nxt = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(nxt)
        cur = torch.cat([cur, nxt], dim=1)
        if eos_id is not None and int(nxt.item()) == int(eos_id):
            break
    return torch.cat(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long, device=cur.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--out_path", type=str, default="./runs_lm/overnight/ruler_sniah_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    data = load_examples(args.data_path, args.max_examples)
    print(f"[DATA] Loaded {len(data)} valid S-NIAH examples")

    rft, _ = load_rft_model(args.rft_ckpt, device)
    baseline, _ = load_baseline_model(args.baseline_ckpt, device)

    results = []
    t0 = time.time()
    for i, ex in enumerate(data):
        prompt = build_prompt(ex["context"], ex["question"])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        rft_ids = generate_rft(rft, ids, args.max_new_tokens, args.chunk_size, tok.eos_token_id)
        base_ids = generate_baseline(baseline, ids, args.max_new_tokens, tok.eos_token_id)

        rft_txt = tok.decode(rft_ids[0], skip_special_tokens=True).strip()
        base_txt = tok.decode(base_ids[0], skip_special_tokens=True).strip()
        gold = ex["answer"].strip()

        g = normalize_text(gold)
        r = normalize_text(rft_txt)
        b = normalize_text(base_txt)

        rft_ok = (r == g) or (g in r)
        base_ok = (b == g) or (g in b)

        results.append({
            "idx": i,
            "context_length": ex["context_length"],
            "gold": gold,
            "rft_pred": rft_txt,
            "baseline_pred": base_txt,
            "rft_correct": bool(rft_ok),
            "baseline_correct": bool(base_ok),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            ra = sum(int(x["rft_correct"]) for x in results) / len(results)
            ba = sum(int(x["baseline_correct"]) for x in results) / len(results)
            print(f"  [{i+1}/{len(data)}] RFT acc={ra:.4f} | Baseline acc={ba:.4f}")

    rft_acc = sum(int(x["rft_correct"]) for x in results) / max(len(results), 1)
    base_acc = sum(int(x["baseline_correct"]) for x in results) / max(len(results), 1)

    by_len = {}
    for r in results:
        k = str(r["context_length"]) if r["context_length"] is not None else "unknown"
        by_len.setdefault(k, []).append(r)
    by_len_summary = {}
    for k, rows in by_len.items():
        by_len_summary[k] = {
            "n": len(rows),
            "rft_acc": sum(int(x["rft_correct"]) for x in rows) / len(rows),
            "baseline_acc": sum(int(x["baseline_correct"]) for x in rows) / len(rows),
        }

    out = {
        "data_path": args.data_path,
        "n_examples": len(results),
        "rft_acc": rft_acc,
        "baseline_acc": base_acc,
        "delta_acc": rft_acc - base_acc,
        "elapsed_sec": time.time() - t0,
        "by_context_length": by_len_summary,
        "examples": results,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== RULER S-NIAH RESULT ===")
    print(f"RFT acc:      {rft_acc:.4f}")
    print(f"Baseline acc: {base_acc:.4f}")
    print(f"Delta acc:    {rft_acc - base_acc:+.4f}")
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()

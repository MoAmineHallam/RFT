"""
eval_perplexity.py — Compare RFT-LM vs Baseline on C4 validation set.

Evaluates perplexity at multiple sequence lengths. Both models are
evaluated with the same chunked processing used during training.
The key difference: RFT-LM builds a memory bank across chunks,
while the baseline processes each chunk independently.

Usage:
    CUDA_VISIBLE_DEVICES=3 python eval_perplexity.py \
        --val_path ./data/c4_val.jsonl \
        --rft_ckpt ./runs_lm/overnight/rft_lm/best_model.pt \
        --baseline_ckpt ./runs_lm/overnight/baseline/best_model.pt \
        --tokenizer_path ./gpt2_tokenizer \
        --seq_lens 2048 4096 8192 \
        --max_docs 2000 \
        --batch_size 1
"""

import argparse
import json
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from RFT_LM import RFTLM, BaselineTransformerLM, MemoryBank


# ─────────────────────────────────────────────
# Data Loading (matches training exactly)
# ─────────────────────────────────────────────
def load_val_tokens(path: str, vocab_size: int = 50257,
                    max_docs: int = None, tokenizer_path: str = None) -> torch.Tensor:
    """Load and tokenize validation data, matching training tokenization."""
    print(f"[DATA] Loading {path}...")
    t0 = time.time()
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            obj = json.loads(line)
            texts.append(obj["text"])
    print(f"[DATA] Loaded {len(texts):,} documents in {time.time() - t0:.1f}s")

    print(f"[DATA] Tokenizing...")
    t0 = time.time()
    all_tokens = []
    if tokenizer_path and os.path.exists(tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        tokenizer.model_max_length = 10**9
        print(f"[DATA] Using GPT-2 BPE tokenizer from {tokenizer_path} (vocab={tokenizer.vocab_size})")
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
    else:
        print(f"[DATA] Using byte-level tokenizer (vocab=256)")
        for text in texts:
            tokens = [min(b + 1, vocab_size - 1) for b in text.encode("utf-8")]
            all_tokens.extend(tokens)

    tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"[DATA] {len(tokens):,} tokens in {time.time() - t0:.1f}s")
    return tokens


# ─────────────────────────────────────────────
# Unified Chunked Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_chunked(model, tokens: torch.Tensor, seq_len: int,
                 chunk_size: int, device: torch.device,
                 use_memory: bool = False,
                 batch_size: int = 1, max_seqs: int = None) -> dict:
    """
    Evaluate any model with chunked processing, matching training.

    For RFT-LM (use_memory=True): builds memory bank across chunks.
    For Baseline (use_memory=False): each chunk is independent.
    """
    model.eval()

    n_sequences = (len(tokens) - 1) // seq_len
    if max_seqs:
        n_sequences = min(n_sequences, max_seqs)
    if n_sequences == 0:
        return {"loss": float("inf"), "perplexity": float("inf"),
                "n_tokens": 0, "n_sequences": 0}

    total_loss = 0.0
    total_tokens = 0

    for seq_idx in range(0, n_sequences, batch_size):
        actual_bs = min(batch_size, n_sequences - seq_idx)
        batch_seqs = []
        for b in range(actual_bs):
            start = (seq_idx + b) * seq_len
            end = start + seq_len + 1  # +1 for target
            seq = tokens[start:end]
            if len(seq) < seq_len + 1:
                seq = F.pad(seq, (0, seq_len + 1 - len(seq)))
            batch_seqs.append(seq)

        input_ids = torch.stack(batch_seqs).to(device)  # [B, seq_len+1]
        total_len = input_ids.shape[1] - 1  # seq_len

        # Fresh memory bank per sequence
        memory_bank = MemoryBank(max_entries=32768)
        memory_bank.reset()

        for start in range(0, total_len, chunk_size):
            end = min(start + chunk_size, total_len)
            chunk_input = input_ids[:, start:end]
            chunk_target = input_ids[:, start + 1:end + 1]
            L = chunk_input.shape[1]

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
                    print(f"    OOM at seq_len={seq_len}, chunk_start={start}, "
                          f"mem_size={memory_bank.size}")
                    torch.cuda.empty_cache()
                    if total_tokens == 0:
                        return {"loss": float("inf"), "perplexity": float("inf"),
                                "n_tokens": 0, "n_sequences": 0, "oom": True}
                    avg_loss = total_loss / total_tokens
                    return {"loss": avg_loss, "perplexity": math.exp(avg_loss),
                            "n_tokens": total_tokens, "n_sequences": seq_idx,
                            "oom": True, "oom_at_chunk": start}
            else:
                # Baseline: no memory, just pos_offset for RoPE consistency
                out = model(chunk_input, pos_offset=start)

            logits = out["logits"]  # both models return {"logits": ...}
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chunk_target.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += L * actual_bs

            # Update memory bank for RFT-LM
            if use_memory and "new_mem_keys" in out:
                memory_bank.add(
                    out["new_mem_keys"].detach(),
                    out["new_mem_vals"].detach(),
                    start, L,
                )

        if (seq_idx // batch_size) % 20 == 0:
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(f"    seq {seq_idx + actual_bs}/{n_sequences} | "
                  f"ppl: {ppl_so_far:.2f} | tokens: {total_tokens:,}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": ppl,
            "n_tokens": total_tokens, "n_sequences": n_sequences}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
def load_rft_model(ckpt_path: str, device: torch.device) -> Tuple[RFTLM, dict]:
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
    model.load_state_dict(state_dict)

    print(f"[LOAD] RFT-LM: {sum(p.numel() for p in model.parameters()):,} params "
          f"(step {ck['step']}, loss {ck['loss']:.4f})")
    return model, a


def load_baseline_model(ckpt_path: str, device: torch.device) -> Tuple[BaselineTransformerLM, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]

    model = BaselineTransformerLM(
        vocab_size=a["vocab_size"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        ff_mult=a["ff_mult"],
        dropout=0.0,
        max_len=65536,  # large enough for any eval seq_len
    ).to(device)

    state_dict = ck["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    print(f"[LOAD] Baseline: {sum(p.numel() for p in model.parameters()):,} params "
          f"(step {ck['step']}, loss {ck['loss']:.4f})")
    return model, a


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_path", type=str, required=True)
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=None,
                    help="Path to GPT-2 tokenizer (must match training)")
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_docs", type=int, default=2000)
    ap.add_argument("--max_seqs", type=int, default=100)
    ap.add_argument("--outfile", type=str, default="eval_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}")
    print(f"[EVAL] Sequence lengths: {args.seq_lens}")
    print(f"[EVAL] Chunk size: {args.chunk_size}")

    # Determine tokenizer from checkpoint if not specified
    rft_ck = torch.load(args.rft_ckpt, map_location="cpu", weights_only=False)
    vocab_size = rft_ck["args"]["vocab_size"]
    tokenizer_path = args.tokenizer_path or rft_ck["args"].get("tokenizer_path")
    del rft_ck

    if tokenizer_path:
        print(f"[EVAL] Tokenizer: {tokenizer_path}")
    else:
        print(f"[EVAL] Tokenizer: byte-level (vocab={vocab_size})")

    # Load validation tokens once
    tokens = load_val_tokens(
        args.val_path,
        vocab_size=vocab_size,
        max_docs=args.max_docs,
        tokenizer_path=tokenizer_path,
    )
    print(f"[EVAL] Total val tokens: {len(tokens):,}\n")

    all_results = {}

    for seq_len in args.seq_lens:
        print(f"{'='*60}")
        print(f"  EVALUATING AT SEQ_LEN = {seq_len}")
        print(f"{'='*60}")

        n_possible = (len(tokens) - 1) // seq_len
        n_eval = min(n_possible, args.max_seqs)
        if n_eval == 0:
            print(f"  Not enough tokens (need {seq_len+1}, have {len(tokens)})")
            continue
        print(f"  Sequences available: {n_possible}, evaluating: {n_eval}")

        # --- RFT-LM ---
        print(f"\n  [RFT-LM] Loading...")
        torch.cuda.empty_cache()
        rft_model, _ = load_rft_model(args.rft_ckpt, device)

        print(f"  [RFT-LM] Evaluating (chunked, with memory)...")
        t0 = time.time()
        rft_result = eval_chunked(
            rft_model, tokens, seq_len,
            chunk_size=args.chunk_size,
            device=device,
            use_memory=True,
            batch_size=args.batch_size,
            max_seqs=args.max_seqs,
        )
        rft_result["time_s"] = round(time.time() - t0, 1)
        print(f"  [RFT-LM] Loss: {rft_result['loss']:.4f} | "
              f"PPL: {rft_result['perplexity']:.2f} | "
              f"Time: {rft_result['time_s']}s")

        del rft_model
        torch.cuda.empty_cache()

        # --- Baseline ---
        print(f"\n  [Baseline] Loading...")
        bl_model, _ = load_baseline_model(args.baseline_ckpt, device)

        print(f"  [Baseline] Evaluating (chunked, no memory)...")
        t0 = time.time()
        bl_result = eval_chunked(
            bl_model, tokens, seq_len,
            chunk_size=args.chunk_size,
            device=device,
            use_memory=False,
            batch_size=args.batch_size,
            max_seqs=args.max_seqs,
        )
        bl_result["time_s"] = round(time.time() - t0, 1)
        print(f"  [Baseline] Loss: {bl_result['loss']:.4f} | "
              f"PPL: {bl_result['perplexity']:.2f} | "
              f"Time: {bl_result['time_s']}s")

        del bl_model
        torch.cuda.empty_cache()

        # --- Comparison ---
        delta_loss = bl_result["loss"] - rft_result["loss"]
        delta_ppl = bl_result["perplexity"] - rft_result["perplexity"]
        winner = "RFT-LM" if rft_result["loss"] < bl_result["loss"] else "Baseline"

        all_results[str(seq_len)] = {
            "rft_lm": rft_result,
            "baseline": bl_result,
            "delta_loss": round(delta_loss, 6),
            "delta_ppl": round(delta_ppl, 4),
            "winner": winner,
        }

        print(f"\n  {'─'*50}")
        print(f"  seq_len={seq_len}: {winner} wins")
        print(f"    RFT-LM   PPL={rft_result['perplexity']:.2f}  loss={rft_result['loss']:.4f}")
        print(f"    Baseline PPL={bl_result['perplexity']:.2f}  loss={bl_result['loss']:.4f}")
        print(f"    Δ loss={delta_loss:+.4f}  Δ PPL={delta_ppl:+.2f}")

        # Stop if OOM occurred
        if rft_result.get("oom") or bl_result.get("oom"):
            print(f"\n  OOM detected — skipping longer sequence lengths")
            break

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'SeqLen':<8} {'RFT PPL':<12} {'Base PPL':<12} {'Δ PPL':<10} {'Winner'}")
    print(f"  {'─'*56}")
    for sl in args.seq_lens:
        r = all_results.get(str(sl))
        if r:
            oom = " (OOM)" if r["rft_lm"].get("oom") or r["baseline"].get("oom") else ""
            print(f"  {sl:<8} {r['rft_lm']['perplexity']:<12.2f} "
                  f"{r['baseline']['perplexity']:<12.2f} "
                  f"{r['delta_ppl']:<+10.2f} {r['winner']}{oom}")

    # Save
    outpath = os.path.join(os.path.dirname(args.rft_ckpt), "..", args.outfile)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {outpath}")


if __name__ == "__main__":
    main()

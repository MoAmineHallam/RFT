"""
diag_ppl_memory.py — Why is RFT-LM's held-out perplexity catastrophic?

RFT-LM scores PPL ~6450 on C4-val vs a matched baseline's ~127. Two very
different explanations, with opposite implications for the paper:

  (H1) CATASTROPHIC FORGETTING — 40 epochs of NIAH-heavy mixed training
       destroyed the language model. The weights themselves are broken.
       => memory-OFF perplexity stays terrible.

  (H2) MEMORY-INJECTION NOISE — the weights are fine, but at inference the
       memory layer adds tanh(mem_gate_alpha) * mem_ctx (~0.76 of a vector)
       into the residual stream at every position. On ordinary text there is
       no relevant needle to retrieve, so this injects a large irrelevant
       perturbation and wrecks next-token prediction.
       => memory-OFF perplexity recovers toward baseline.

H2 would also predict that the gate-0.1 model (7.6x weaker injection) has far
better perplexity than the gate-1.0 model — while we already know both score
identically on NIAH (93.7 vs 92.7). That combination is a genuine, reportable
trade-off: gate strength is free for retrieval but expensive for language
modeling.

Usage:
  python diag_ppl_memory.py --val_path .../c4_val.jsonl \
      --tokenizer_path .../gpt2_tokenizer \
      --ckpt NAME=/path/to/best_model.pt [--ckpt NAME2=/path2 ...] \
      --seq_len 2048 --max_seqs 100
"""
import argparse
import json

import torch

from eval_perplexity import eval_chunked, load_rft_model, load_val_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--ckpt", type=str, action="append", required=True,
                    help="NAME=/path/to/best_model.pt  (repeatable)")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_docs", type=int, default=2000)
    ap.add_argument("--max_seqs", type=int, default=100)
    ap.add_argument("--outfile", type=str, default="diag_ppl_memory.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = load_val_tokens(args.val_path, tokenizer_path=args.tokenizer_path,
                             max_docs=args.max_docs)

    rows = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        model, cfg = load_rft_model(path, device)

        gate = float(torch.tanh(model.memory_layer.mem_gate_alpha).item())
        print(f"  tanh(mem_gate_alpha) = {gate:.4f}  "
              f"(fraction of mem_ctx added to the residual stream)")

        model.use_memory = True
        on = eval_chunked(model, tokens, args.seq_len, args.chunk_size, device,
                          use_memory=True, max_seqs=args.max_seqs)
        print(f"  memory ON : loss {on['loss']:.4f}  PPL {on['perplexity']:.2f}")

        model.use_memory = False
        off = eval_chunked(model, tokens, args.seq_len, args.chunk_size, device,
                           use_memory=False, max_seqs=args.max_seqs)
        print(f"  memory OFF: loss {off['loss']:.4f}  PPL {off['perplexity']:.2f}")

        ratio = on["perplexity"] / max(off["perplexity"], 1e-9)
        print(f"  ON/OFF PPL ratio = {ratio:.1f}x")
        rows.append({"name": name, "path": path, "gate": gate,
                     "ppl_memory_on": on["perplexity"],
                     "ppl_memory_off": off["perplexity"],
                     "ratio": ratio})
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  SUMMARY (baseline reference PPL ~127)\n{'='*60}")
    print(f"  {'model':<22}{'gate':>7}{'PPL on':>12}{'PPL off':>12}{'ratio':>9}")
    for r in rows:
        print(f"  {r['name']:<22}{r['gate']:>7.3f}"
              f"{r['ppl_memory_on']:>12.1f}{r['ppl_memory_off']:>12.1f}"
              f"{r['ratio']:>9.1f}x")
    print("\n  memory-OFF near ~127  => H2 (memory-injection noise)")
    print("  memory-OFF still huge => H1 (catastrophic forgetting)")

    with open(args.outfile, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[SAVED] {args.outfile}")


if __name__ == "__main__":
    main()

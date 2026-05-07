"""
train_graft.py — Fine-tune RFT memory operator on top of pretrained HF base.
Trainable: memory_layer + mem_key_proj + mem_val_proj.
Frozen: the base LM.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoTokenizer

from rft_graft import RFTGraftLM
from niah_batch import NIAHBatchConfig, NIAHBatchGen, niah_retrieval_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data_path", type=str, required=True,
                    help="C4 jsonl file (used as distractor pool).")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--memory_layer_idx", type=int, default=12)
    ap.add_argument("--mem_top_m", type=int, default=64)
    ap.add_argument("--ocr_dim", type=int, default=256)
    ap.add_argument("--mem_gate_alpha_init", type=float, default=1.0)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--niah_value_type", type=str, default="numbers",
                    choices=["digit2", "short_int", "numbers"])
    ap.add_argument("--niah_max_val_tokens", type=int, default=8)
    ap.add_argument("--niah_num_needles", type=int, default=3)
    ap.add_argument("--niah_distractor_docs", type=int, default=300)
    ap.add_argument("--niah_lm_w", type=float, default=3.0)
    ap.add_argument("--niah_decode_w", type=float, default=10.0)
    ap.add_argument("--niah_router_w", type=float, default=1.0)
    ap.add_argument("--niah_topm_w", type=float, default=0.25)
    ap.add_argument("--niah_pointer_w", type=float, default=0.5)
    ap.add_argument("--niah_ocr_w", type=float, default=0.2)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[GRAFT] loading base {args.base}")
    model = RFTGraftLM(
        base_model_name=args.base,
        memory_layer_idx=args.memory_layer_idx,
        mem_top_m=args.mem_top_m,
        ocr_dim=args.ocr_dim,
        mem_gate_alpha_init=args.mem_gate_alpha_init,
        freeze_base=True,
    ).to(device)
    model.train()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[GRAFT] trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M")
    print(f"[GRAFT] d_model={model.d_model} n_layers={model.n_layers} "
          f"memory_layer_idx={model.memory_layer_idx}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tokenizer.model_max_length = 10**9

    print(f"[DATA] reading distractor docs from {args.data_path}")
    texts = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.niah_distractor_docs:
                break
            texts.append(json.loads(line).get("text", ""))
    distractor_tokens = tokenizer.encode(" ".join(texts), add_special_tokens=False)
    print(f"[DATA] {len(distractor_tokens):,} distractor tokens")

    niah_cfg = NIAHBatchConfig(
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_needles=args.niah_num_needles,
        value_type=args.niah_value_type,
        max_val_tokens=args.niah_max_val_tokens,
    )
    niah_gen = NIAHBatchGen(
        niah_cfg, tokenizer, distractor_tokens, device=device, seed=args.seed + 2000
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    loss_weights = {
        "lm_ce": args.niah_lm_w,
        "decode": args.niah_decode_w,
        "router_ce": args.niah_router_w,
        "topm_hinge": args.niah_topm_w,
        "pointer_ce": args.niah_pointer_w,
        "ocr_contrastive": args.niah_ocr_w,
    }

    metrics_path = os.path.join(args.outdir, "train_metrics.jsonl")
    best = float("inf")
    t0 = time.time()
    for step in range(args.steps):
        for g in optim.param_groups:
            g["lr"] = lr_at(step)
        optim.zero_grad(set_to_none=True)
        batch = next(niah_gen)
        loss_val, m = niah_retrieval_step(model, batch, loss_weights, do_backward=True)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()

        if step % args.log_every == 0:
            tps = (step + 1) * args.batch_size * args.chunk_size * 2 / max(1, time.time() - t0)
            print(f"  step {step:6d} | loss {loss_val:7.3f} "
                  f"| lm_acc {m['niah_lm_acc']:.3f} "
                  f"| full {m['niah_lm_full_match']:.3f} "
                  f"| emb {m['niah_embed_acc']:.3f} "
                  f"| r@M {m['niah_recall_at_m']:.3f} "
                  f"| lr {lr_at(step):.2e} | {tps:.0f} tok/s")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, **m}) + "\n")

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            # Save only trainable bits (base is reloaded from HF on eval)
            trainable_state = {
                k: v for k, v in model.state_dict().items() if not k.startswith("base.")
            }
            ckpt = os.path.join(args.outdir, f"checkpoint_step{step+1}.pt")
            torch.save({"model": trainable_state, "args": vars(args), "step": step + 1}, ckpt)
            print(f"  [SAVED] {ckpt}")
            if loss_val < best:
                best = loss_val
                torch.save(
                    {"model": trainable_state, "args": vars(args), "step": step + 1},
                    os.path.join(args.outdir, "best_model.pt"),
                )


if __name__ == "__main__":
    main()

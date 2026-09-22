"""
RFT-LM Overnight Training — C4 Pretraining on 3×L40S

Trains two models on real C4 text:
  1. RFT-LM (Transformer + RFT memory layer + OCR) 
  2. Baseline Transformer (same size, no memory)

Both use the same data, same hyperparams, same training budget.
The comparison shows whether the memory layer helps.

Usage:
    # Run on GPUs 3,4,5 overnight (~8-10 hours)
    CUDA_VISIBLE_DEVICES=3,4,5 python train_overnight.py \
        --data_path ./data/c4_train.jsonl \
        --outdir ./runs_lm/overnight \
        --model rft_lm \
        --n_gpus 3

    # Then run baseline on GPUs 3,4,5
    CUDA_VISIBLE_DEVICES=3,4,5 python train_overnight.py \
        --data_path ./data/c4_train.jsonl \
        --outdir ./runs_lm/overnight \
        --model baseline \
        --n_gpus 3
"""

import argparse
import json
import math
import os
import random
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Import from our architecture file
from RFT_LM import (
    RFTLM, MTLM, BaselineTransformerLM, MemoryBank,
    train_step_chunked,
)
from synth_batch import SynthBatchConfig, SyntheticKVBatchGen, synthetic_retrieval_step
from niah_batch import (
    NIAHBatchConfig, NIAHBatchGen, niah_retrieval_step, mt_niah_retrieval_step,
)


# ─────────────────────────────────────────────
# C4 JSONL Dataset
# ─────────────────────────────────────────────
class C4TokenDataset(Dataset):
    """
    Loads C4 jsonl and tokenizes with GPT-2 BPE (if tokenizer_path provided)
    or falls back to byte-level encoding.
    """

    def __init__(self, path: str, seq_len: int, max_docs: int = None,
                 vocab_size: int = 50257, seed: int = 42,
                 tokenizer_path: str = None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        print(f"[DATA] Loading {path}...")
        t0 = time.time()
        
        # Load all text
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                obj = json.loads(line)
                texts.append(obj["text"])
        
        print(f"[DATA] Loaded {len(texts):,} documents in {time.time() - t0:.1f}s")

        # Tokenize
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
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.n_sequences = (len(self.tokens) - 1) // seq_len
        
        print(f"[DATA] {len(self.tokens):,} tokens, {self.n_sequences:,} sequences "
              f"of length {seq_len} in {time.time() - t0:.1f}s")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            # Pad if at the end
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk  # [seq_len + 1], will split into input/target


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train_rft_lm(
    rank: int,
    world_size: int,
    args,
):
    """Train RFT-LM with chunked processing."""
    
    # DDP setup
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main = (rank == 0)

    if is_main:
        model_label = {
            "rft_lm": "RFT-LM",
            "rft_lm_disable_memory": "RFT-LM (disable_memory)",
            "rft_lm_disable_ocr": "RFT-LM (disable_ocr)",
            "baseline": "Baseline Transformer",
        }.get(args.model, args.model)
        print(f"\n{'='*70}")
        print(f"  Training: {model_label}")
        print(f"  GPUs: {world_size}, Device: {device}")
        print(f"{'='*70}")

    # Load data
    dataset = C4TokenDataset(
        args.data_path,
        seq_len=args.total_seq_len,
        max_docs=args.max_docs,
        vocab_size=args.vocab_size,
        tokenizer_path=getattr(args, 'tokenizer_path', None),
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=args.seed) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    is_rft_variant = args.model in {"rft_lm", "rft_lm_disable_memory", "rft_lm_disable_ocr"}
    is_mt_variant = args.model in {"mt_lm", "mt_lm_disable_memory"}
    has_memory_layer = is_rft_variant or is_mt_variant
    use_memory = args.model not in {"rft_lm_disable_memory", "mt_lm_disable_memory"}
    if is_rft_variant:
        model = RFTLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            window_size=args.window_size,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            memory_layer_idx=args.memory_layer_idx,
            use_memory=use_memory,
            mem_top_m=args.mem_top_m,
            ocr_dim=args.ocr_dim,
        ).to(device)
        if args.model == "rft_lm_disable_ocr" and hasattr(model, "memory_layer") and hasattr(model.memory_layer, "ocr_alpha"):
            with torch.no_grad():
                model.memory_layer.ocr_alpha.fill_(0.0)
            model.memory_layer.ocr_alpha.requires_grad_(False)
    elif is_mt_variant:
        model = MTLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            window_size=args.window_size,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            memory_layer_idx=args.memory_layer_idx,
            use_memory=use_memory,
            mem_top_k=args.mem_top_m,  # reuse flag
            mem_gate_alpha_init=(args.mem_gate_alpha_init if args.mem_gate_alpha_init is not None else 1.0),
        ).to(device)
    else:
        model = BaselineTransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            max_len=args.total_seq_len + 1,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"[MODEL] {args.model}: {n_params:,} params")
        print(f"[MODEL] d_model={args.d_model}, n_layers={args.n_layers}, "
              f"n_heads={args.n_heads}, window={args.window_size}")
        if is_rft_variant:
            print(f"[MODEL] memory_layer_idx={args.memory_layer_idx}, "
                  f"mem_top_m={args.mem_top_m}, ocr_dim={args.ocr_dim}")
            print(f"[MODEL] use_memory={use_memory}")
        elif is_mt_variant:
            print(f"[MODEL] memory_layer_idx={args.memory_layer_idx}, "
                  f"mem_top_k={args.mem_top_m} (kNN, no OCR, no recency)")
            print(f"[MODEL] use_memory={use_memory}")

    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=False)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    
    # Cosine LR schedule with warmup
    total_steps = args.epochs * len(dataloader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    # Resume from checkpoint if specified
    resume_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        raw_m = model.module if world_size > 1 else model
        raw_m.load_state_dict(sd, strict=False)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        resume_step = ckpt.get("step", 0)
        if is_main:
            print(f"[RESUME] Loaded checkpoint from {args.resume_from} (step={resume_step})")

    # Override mem_gate_alpha if requested (works for RFT + MT variants)
    if args.mem_gate_alpha_init is not None and has_memory_layer:
        raw_m = model.module if world_size > 1 else model
        if hasattr(raw_m, "memory_layer") and hasattr(raw_m.memory_layer, "mem_gate_alpha"):
            old_val = raw_m.memory_layer.mem_gate_alpha.item()
            with torch.no_grad():
                raw_m.memory_layer.mem_gate_alpha.fill_(args.mem_gate_alpha_init)
            if is_main:
                print(f"[OVERRIDE] mem_gate_alpha: {old_val:.4f} → {args.mem_gate_alpha_init:.4f} "
                      f"(tanh={torch.tanh(raw_m.memory_layer.mem_gate_alpha).item():.4f})")

    # Training
    if is_main:
        print(f"\n[TRAIN] epochs={args.epochs}, steps/epoch={len(dataloader)}, "
              f"total_steps={total_steps}")
        print(f"[TRAIN] batch_size={args.batch_size}×{world_size}={args.batch_size * world_size}, "
              f"seq_len={args.total_seq_len}, chunk_size={args.chunk_size}")
        print(f"[TRAIN] lr={args.lr}, warmup={warmup_steps}\n")

    memory_bank = MemoryBank(max_entries=args.total_seq_len) if has_memory_layer else None

    # Synthetic retrieval batch generator (only for rft_lm with memory enabled)
    use_synth = is_rft_variant and use_memory and args.model == "rft_lm" and args.synth_ratio > 0
    synth_gen = None
    synth_loss_weights = {
        "router_ce": args.synth_router_w,
        "topm_hinge": args.synth_topm_w,
        "pointer_ce": args.synth_pointer_w,
        "ocr_contrastive": args.synth_ocr_w,
    }
    if use_synth:
        synth_cfg = SynthBatchConfig(
            chunk_size=args.chunk_size,
            key_vocab=args.synth_key_vocab,
            val_vocab=args.synth_val_vocab,
            num_facts=args.synth_num_facts,
            num_decoys=args.synth_num_decoys,
            batch_size=args.synth_batch_size,
        )
        synth_gen = SyntheticKVBatchGen(synth_cfg, device=device, seed=args.seed + rank)
        assert synth_gen.max_token_id() < args.vocab_size, (
            f"synth vocab exceeds model vocab_size ({synth_gen.max_token_id()} >= {args.vocab_size})"
        )
        if is_main:
            print(f"[SYNTH] mixed-objective ON: ratio {args.synth_ratio:.2f}->{args.synth_ratio_end:.2f}, "
                  f"bs={args.synth_batch_size} chunk={args.chunk_size} facts={args.synth_num_facts} "
                  f"decoys={args.synth_num_decoys}")
            print(f"[SYNTH] loss weights: router={args.synth_router_w} topm={args.synth_topm_w} "
                  f"ptr={args.synth_pointer_w} ocr={args.synth_ocr_w} scale={args.synth_loss_scale}")

    # NIAH natural-language retrieval batch generator
    # Enabled for RFT-LM (full pipeline) and MTLM (lm + optional embed align).
    use_niah = (
        use_memory
        and args.niah_ratio > 0
        and (args.model == "rft_lm" or args.model == "mt_lm")
    )
    niah_gen = None
    niah_loss_weights = {
        "lm_ce": args.niah_lm_w,
        "decode": args.niah_decode_w,
        "router_ce": args.niah_router_w,
        "topm_hinge": args.niah_topm_w,
        "pointer_ce": args.niah_pointer_w,
        "ocr_contrastive": args.niah_ocr_w,
    }
    if use_niah:
        # Load tokenizer and distractor tokens for NIAH batches
        from transformers import AutoTokenizer as _AT
        niah_tokenizer = _AT.from_pretrained(
            args.tokenizer_path or "gpt2", use_fast=True
        )
        niah_tokenizer.model_max_length = 10**9

        # Load distractor tokens from C4 data
        import json as _json
        _niah_texts = []
        with open(args.data_path, "r", encoding="utf-8") as _f:
            for _i, _line in enumerate(_f):
                if _i >= args.niah_distractor_docs:
                    break
                _niah_texts.append(_json.loads(_line).get("text", ""))
        niah_dist_tokens = niah_tokenizer.encode(
            " ".join(_niah_texts), add_special_tokens=False
        )

        niah_cfg = NIAHBatchConfig(
            chunk_size=args.chunk_size,
            batch_size=args.niah_batch_size,
            num_needles=args.niah_num_needles,
            value_type=args.niah_value_type,
            max_val_tokens=args.niah_max_val_tokens,
        )
        niah_gen = NIAHBatchGen(
            niah_cfg, niah_tokenizer, niah_dist_tokens,
            device=device, seed=args.seed + 2000 + rank,
        )
        if is_main:
            print(f"[NIAH] natural-language NIAH ON: ratio {args.niah_ratio:.2f}->"
                  f"{args.niah_ratio_end:.2f}, bs={args.niah_batch_size} "
                  f"needles={args.niah_num_needles} dist_toks={len(niah_dist_tokens):,}")
            print(f"[NIAH] value_type={args.niah_value_type} max_val_tokens={args.niah_max_val_tokens}")
            print(f"[NIAH] loss weights: lm={args.niah_lm_w} decode={args.niah_decode_w} "
                  f"router={args.niah_router_w} topm={args.niah_topm_w} "
                  f"ptr={args.niah_pointer_w} ocr={args.niah_ocr_w}")

    rng_synth = random.Random(args.seed + 1000 + rank)

    global_step = 0
    best_loss = float("inf")
    log_interval = args.log_interval
    save_interval = args.save_every
    all_losses = []

    outdir = os.path.join(args.outdir, args.model)
    os.makedirs(outdir, exist_ok=True)

    metrics_path = os.path.join(outdir, "train_metrics.jsonl")
    latest_path = os.path.join(outdir, "latest_metrics.json")
    if is_main and os.path.exists(metrics_path):
        os.remove(metrics_path)

    raw_model = model.module if world_size > 1 else model

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)  # [B, seq_len+1]
            input_ids = batch[:, :-1]  # [B, seq_len]
            target_ids = batch[:, 1:]  # [B, seq_len]

            opt.zero_grad()

            # Decide step type: NIAH / synthetic / C4
            progress = global_step / max(total_steps, 1)
            if use_synth:
                cur_synth_ratio = args.synth_ratio + progress * (args.synth_ratio_end - args.synth_ratio)
            else:
                cur_synth_ratio = 0.0
            if use_niah:
                cur_niah_ratio = args.niah_ratio + progress * (args.niah_ratio_end - args.niah_ratio)
            else:
                cur_niah_ratio = 0.0

            r = rng_synth.random()
            do_niah_step = use_niah and (r < cur_niah_ratio)
            do_synth_step = (not do_niah_step) and use_synth and (r < cur_niah_ratio + cur_synth_ratio)

            if do_niah_step:
                niah_batch = next(niah_gen)
                scaled_nw = {k: v * args.niah_loss_scale for k, v in niah_loss_weights.items()}
                if args.model == "mt_lm":
                    _, niah_metrics = mt_niah_retrieval_step(
                        raw_model, niah_batch, scaled_nw, do_backward=True,
                    )
                else:
                    _, niah_metrics = niah_retrieval_step(
                        raw_model, niah_batch, scaled_nw, do_backward=True,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                step_loss = niah_metrics["niah_loss"]
                metrics = {"loss": step_loss, "is_niah": 1.0, **niah_metrics}
                epoch_loss += step_loss
                epoch_tokens += args.niah_batch_size * args.chunk_size * 2
                global_step += 1
                all_losses.append(step_loss)

                if is_main and (global_step % log_interval == 0 or global_step == 1):
                    print(
                        f"  step {global_step:5d} | NIAH loss {step_loss:.4f} | "
                        f"lm_acc {niah_metrics['niah_lm_acc']:.3f} | "
                        f"emb_acc {niah_metrics['niah_embed_acc']:.3f} | "
                        f"emb_loss {niah_metrics['niah_embed_loss']:.3f} | "
                        f"r@M {niah_metrics['niah_recall_at_m']:.3f} | "
                        f"fused {niah_metrics['niah_fused_acc']:.3f}",
                        flush=True,
                    )
                    record = {
                        "step": int(global_step),
                        "is_niah": 1,
                        "loss": float(step_loss),
                        **{k: float(v) for k, v in niah_metrics.items()},
                        "niah_ratio": float(cur_niah_ratio),
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                continue

            if do_synth_step:
                synth_batch = next(synth_gen)
                # Scale into the backward via synth_loss_scale by temporarily rescaling weights
                scaled_weights = {k: v * args.synth_loss_scale for k, v in synth_loss_weights.items()}
                _, synth_metrics = synthetic_retrieval_step(
                    raw_model, synth_batch, scaled_weights, do_backward=True,
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                step_loss = synth_metrics["synth_loss"]
                metrics = {"loss": step_loss, "is_synth": 1.0, **synth_metrics}
                epoch_loss += step_loss
                epoch_tokens += args.synth_batch_size * args.chunk_size * 2
                global_step += 1
                all_losses.append(step_loss)

                if is_main and (global_step % log_interval == 0 or global_step == 1):
                    print(
                        f"  step {global_step:5d} | SYNTH loss {step_loss:.4f} | "
                        f"router_ce {synth_metrics['synth_router_ce']:.3f} | "
                        f"r@1 {synth_metrics['synth_router_top1_acc']:.3f} | "
                        f"r@M {synth_metrics['synth_recall_at_m']:.3f} | "
                        f"ptr {synth_metrics['synth_pointer_acc']:.3f} | "
                        f"fused {synth_metrics['synth_fused_acc']:.3f} | "
                        f"ratio {cur_synth_ratio:.2f}",
                        flush=True,
                    )
                    record = {
                        "step": int(global_step),
                        "is_synth": 1,
                        "loss": float(step_loss),
                        **{k: float(v) for k, v in synth_metrics.items()},
                        "synth_ratio": float(cur_synth_ratio),
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                continue

            if has_memory_layer:
                # Chunked training with memory (RFT or MT)
                memory_bank.reset()
                effective_ocr_loss_weight = args.ocr_loss_weight if args.model == "rft_lm" else 0.0
                loss, metrics = train_step_chunked(
                    raw_model, batch, args.chunk_size, memory_bank,
                    do_backward=True,
                    ocr_loss_weight=effective_ocr_loss_weight,
                    ocr_margin=args.ocr_margin,
                )
            else:
                # Standard training — process full sequence
                # For long sequences, chunk to avoid OOM
                if args.total_seq_len > args.window_size * 2:
                    # Chunk for baseline too
                    chunk_losses = []
                    n_chunks = max(1, args.total_seq_len // args.chunk_size)
                    for cs in range(0, args.total_seq_len, args.chunk_size):
                        ce = min(cs + args.chunk_size, args.total_seq_len)
                        chunk_in = input_ids[:, cs:ce]
                        chunk_tgt = target_ids[:, cs:ce]
                        out = raw_model(chunk_in, pos_offset=cs)
                        closs = F.cross_entropy(
                            out["logits"].reshape(-1, args.vocab_size),
                            chunk_tgt.reshape(-1),
                        )
                        (closs / n_chunks).backward()
                        chunk_losses.append(closs.item())
                    loss_val = sum(chunk_losses) / len(chunk_losses)
                    metrics = {"loss": loss_val}
                else:
                    out = raw_model(input_ids)
                    loss = F.cross_entropy(
                        out["logits"].reshape(-1, args.vocab_size),
                        target_ids.reshape(-1),
                    )
                    loss.backward()
                    metrics = {"loss": loss.item()}

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            step_loss = metrics["loss"]
            epoch_loss += step_loss
            epoch_tokens += args.total_seq_len * args.batch_size
            global_step += 1
            all_losses.append(step_loss)

            if is_main and (global_step % log_interval == 0 or global_step == 1):
                avg_recent = sum(all_losses[-log_interval:]) / len(all_losses[-log_interval:])
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_epoch
                tok_per_sec = epoch_tokens / max(elapsed, 1)
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                mem_size = int(memory_bank.size) if memory_bank is not None else None
                eta_sec = (total_steps - global_step) * (elapsed / max(batch_idx + 1, 1))

                ocr_loss_val = metrics.get("ocr_loss", 0.0) if isinstance(metrics, dict) else 0.0
                record = {
                    "step": int(global_step),
                    "epoch": int(epoch + 1),
                    "batch_idx": int(batch_idx + 1),
                    "loss": float(step_loss),
                    "ocr_loss": float(ocr_loss_val),
                    "avg_recent": float(avg_recent),
                    "lr": float(lr_now),
                    "tok_per_sec": float(tok_per_sec),
                    "peak_mb": float(peak_mb),
                    "memory_size": mem_size,
                    "elapsed_sec": float(elapsed),
                    "eta_sec": float(eta_sec),
                }

                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                with open(latest_path, "w") as f:
                    json.dump(record, f, indent=2)

                mem_info = f" mem={mem_size}" if mem_size is not None else ""
                print(
                    f"  step {global_step:5d} | loss {step_loss:.4f} | "
                    f"avg {avg_recent:.4f} | lr {lr_now:.2e} | "
                    f"{tok_per_sec:.0f} tok/s | peak {peak_mb:.0f}MB{mem_info} | "
                    f"eta {eta_sec/60:.1f}m",
                    flush=True,
                )

            if is_main and save_interval > 0 and global_step % save_interval == 0:
                ckpt_path = os.path.join(outdir, f"checkpoint_step{global_step}.pt")
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "loss": step_loss,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  [SAVED] {ckpt_path}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(batch_idx + 1, 1)
        epoch_time = time.time() - t_epoch

        if is_main:
            print(f"\n  Epoch {epoch + 1}/{args.epochs}: avg_loss={avg_epoch_loss:.4f} "
                  f"time={epoch_time:.0f}s ({epoch_time / 60:.1f}min)")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(outdir, "best_model.pt")
                torch.save({
                    "model": raw_model.state_dict(),
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": avg_epoch_loss,
                    "args": vars(args),
                }, best_path)
                print(f"  [BEST] {best_path} (loss={avg_epoch_loss:.4f})")

    # Final save
    if is_main:
        final_path = os.path.join(outdir, "final_model.pt")
        torch.save({
            "model": raw_model.state_dict(),
            "step": global_step,
            "loss": all_losses[-1] if all_losses else float("inf"),
            "args": vars(args),
        }, final_path)

        results = {
            "model": args.model,
            "n_params": n_params,
            "final_loss": all_losses[-1] if all_losses else None,
            "best_loss": best_loss,
            "total_steps": global_step,
            "all_losses": all_losses,
            "args": vars(args),
        }
        with open(os.path.join(outdir, "train_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETE: {args.model}")
        print(f"  Steps: {global_step}, Best loss: {best_loss:.4f}")
        print(f"  Saved: {outdir}/")
        print(f"{'='*70}\n")

    if world_size > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="/data/AmineHL_data/overnight")
    ap.add_argument("--max_docs", type=int, default=None,
                    help="Limit number of documents (None = use all)")

    # Model selection
    ap.add_argument("--model", type=str, default="rft_lm",
                    choices=["rft_lm", "rft_lm_disable_memory", "rft_lm_disable_ocr",
                             "mt_lm", "mt_lm_disable_memory", "baseline"])

    # Model architecture
    ap.add_argument("--vocab_size", type=int, default=50257,
                    help="GPT-2 = 50257. Byte-level = 256.")
    ap.add_argument("--tokenizer_path", type=str, default=None,
                    help="Path to GPT-2 tokenizer directory. If None, uses byte-level.")
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--window_size", type=int, default=512)
    
    # RFT Memory config
    ap.add_argument("--memory_layer_idx", type=int, default=6)
    ap.add_argument("--mem_top_m", type=int, default=64)
    ap.add_argument("--ocr_dim", type=int, default=256)
    ap.add_argument("--ocr_loss_weight", type=float, default=0.0,
                    help="Weight for self-supervised circular OCR loss on C4 batches "
                         "(keep 0 when using synthetic supervision).")
    ap.add_argument("--ocr_margin", type=float, default=0.10,
                    help="Margin for self-supervised OCR loss")

    # Mixed-objective synthetic retrieval supervision
    ap.add_argument("--synth_ratio", type=float, default=0.30,
                    help="Probability of replacing a C4 step with a synthetic "
                         "KV-retrieval step at the start of training.")
    ap.add_argument("--synth_ratio_end", type=float, default=0.10,
                    help="synth_ratio anneals linearly to this value by end of run.")
    ap.add_argument("--synth_loss_scale", type=float, default=1.0,
                    help="Global scale on the synthetic retrieval loss.")
    ap.add_argument("--synth_batch_size", type=int, default=4)
    ap.add_argument("--synth_num_facts", type=int, default=8)
    ap.add_argument("--synth_num_decoys", type=int, default=64)
    ap.add_argument("--synth_key_vocab", type=int, default=2048)
    ap.add_argument("--synth_val_vocab", type=int, default=2048)
    ap.add_argument("--synth_router_w", type=float, default=1.0)
    ap.add_argument("--synth_topm_w", type=float, default=0.25)
    ap.add_argument("--synth_pointer_w", type=float, default=0.5)
    ap.add_argument("--synth_ocr_w", type=float, default=0.2)

    # NIAH-format natural-language retrieval supervision
    ap.add_argument("--niah_ratio", type=float, default=0.15,
                    help="Probability of a NIAH-format step (natural language needles).")
    ap.add_argument("--niah_ratio_end", type=float, default=0.10,
                    help="niah_ratio anneals linearly to this value.")
    ap.add_argument("--niah_loss_scale", type=float, default=1.0)
    ap.add_argument("--niah_batch_size", type=int, default=4)
    ap.add_argument("--niah_num_needles", type=int, default=3)
    ap.add_argument("--niah_value_type", type=str,
                    choices=["digit2", "short_int", "numbers"], default="digit2",
                    help="Value distribution. digit2=10-99 (1 BPE), "
                         "short_int=100-999 (~1-2 BPE), numbers=7-digit (~3 BPE).")
    ap.add_argument("--niah_max_val_tokens", type=int, default=8,
                    help="Cap on supervised BPE tokens per value (full-sequence loss).")
    ap.add_argument("--niah_lm_w", type=float, default=2.0,
                    help="Weight for LM loss at probe position in NIAH steps.")
    ap.add_argument("--niah_decode_w", type=float, default=5.0,
                    help="Weight for direct memory-decode shortcut loss.")
    ap.add_argument("--niah_router_w", type=float, default=1.0)
    ap.add_argument("--niah_topm_w", type=float, default=0.25)
    ap.add_argument("--niah_pointer_w", type=float, default=0.5)
    ap.add_argument("--niah_ocr_w", type=float, default=0.2)
    ap.add_argument("--niah_distractor_docs", type=int, default=200,
                    help="Number of C4 documents to use as NIAH distractors.")

    # Training config
    ap.add_argument("--total_seq_len", type=int, default=2048)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000,
                    help="Save checkpoint every N steps. 0 = no intermediate saves.")
    ap.add_argument("--log_interval", type=int, default=20,
                    help="Print and save metrics every N steps.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader worker processes.")
    ap.add_argument("--seed", type=int, default=42)
    
    # Checkpoint resumption
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Path to checkpoint .pt to resume from (loads model + optimizer).")
    ap.add_argument("--mem_gate_alpha_init", type=float, default=None,
                    help="Override mem_gate_alpha after checkpoint load (e.g. 1.0 for tanh(1.0)≈0.76).")

    # DDP
    ap.add_argument("--n_gpus", type=int, default=1)

    args = ap.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.n_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.multiprocessing.spawn(
            train_rft_lm,
            args=(args.n_gpus, args),
            nprocs=args.n_gpus,
            join=True,
        )
    else:
        train_rft_lm(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()

##### RFT_BASE.PY #####
import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


# -----------------------------
# DDP helpers (safe for single GPU)
# -----------------------------
def ddp_init():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world, local_rank

    rank = 0
    world = 1
    local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return False, rank, world, local_rank


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def dist_ready(ddp_enabled: bool) -> bool:
    return ddp_enabled and dist.is_available() and dist.is_initialized()


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Synthetic KV Retrieval with positions (last-occurrence rule)
# -----------------------------
class SyntheticKVRetrievalWithPos:
    def __init__(
        self,
        seq_len: int,
        key_vocab: int,
        val_vocab: int,
        num_facts: int,
        num_decoys: int,
        batch_size: int,
        steps: int,
        device: torch.device,
        base_seed: int,
        rank: int,
        target_repeats: int = 1,
    ):
        self.seq_len = seq_len
        self.key_vocab = key_vocab
        self.val_vocab = val_vocab
        self.num_facts = num_facts
        self.num_decoys = num_decoys
        self.batch_size = batch_size
        self.steps = steps
        self.device = device
        self.target_repeats = int(target_repeats)
        assert self.target_repeats >= 1

        self.PAD = 0
        self.QUERY = key_vocab + val_vocab + 1
        self.VAL_OFFSET = key_vocab + 1
        self.vocab_size = self.QUERY + 1

        self.rng = random.Random(base_seed + 10000 * rank + seq_len)

        self.num_pair_slots = (seq_len - 2) // 2
        assert self.num_pair_slots > 0, "seq_len too small"

        total_pairs = (self.num_facts - 1) + self.target_repeats + self.num_decoys
        assert total_pairs <= self.num_pair_slots, (
            f"Too many pairs to place: total_pairs={total_pairs} > num_pair_slots={self.num_pair_slots}. "
            f"Increase seq_len or reduce num_facts/num_decoys/target_repeats."
        )

    def __iter__(self):
        for _ in range(self.steps):
            x = torch.full((self.batch_size, self.seq_len), self.PAD, dtype=torch.long, device=self.device)
            y = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
            pos_key = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
            pos_val = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)

            for b in range(self.batch_size):
                target_key = self.rng.randint(1, self.key_vocab)

                used_keys = {target_key}
                other_facts = []
                for _k in range(self.num_facts - 1):
                    k = self.rng.randint(1, self.key_vocab)
                    while k in used_keys:
                        k = self.rng.randint(1, self.key_vocab)
                    used_keys.add(k)
                    v = self.rng.randint(0, self.val_vocab - 1)
                    other_facts.append((k, v))

                target_pairs = []
                for _t in range(self.target_repeats):
                    tv = self.rng.randint(0, self.val_vocab - 1)
                    target_pairs.append((target_key, tv))

                decoys = []
                for _d in range(self.num_decoys):
                    dk = self.rng.randint(1, self.key_vocab)
                    dv = self.rng.randint(0, self.val_vocab - 1)
                    decoys.append((dk, dv))

                all_pairs = other_facts + target_pairs + decoys
                self.rng.shuffle(all_pairs)

                available_positions = list(range(0, self.seq_len - 2, 2))
                self.rng.shuffle(available_positions)

                for (k, v), p in zip(all_pairs, available_positions):
                    x[b, p] = k
                    x[b, p + 1] = self.VAL_OFFSET + v

                x[b, -2] = self.QUERY
                x[b, -1] = target_key

                kv_keys = x[b, : self.seq_len - 2 : 2]
                where = (kv_keys == target_key).nonzero(as_tuple=False).view(-1)
                assert where.numel() >= 1

                last_pair_idx = int(where.max().item())
                last_pos_key = 2 * last_pair_idx
                last_pos_val = last_pos_key + 1

                pos_key[b] = last_pos_key
                pos_val[b] = last_pos_val
                y[b] = int(x[b, last_pos_val].item() - self.VAL_OFFSET)

            yield x, y, pos_key, pos_val


# -----------------------------
# Utility + metrics
# -----------------------------
def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.no_grad()
def batch_acc(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=-1)
    return (pred == y).float().mean()


@torch.no_grad()
def recall_at_k(scores: torch.Tensor, target_idx: torch.Tensor, k: int) -> torch.Tensor:
    topk = torch.topk(scores, k=k, dim=1).indices
    hit = (topk == target_idx.unsqueeze(1)).any(dim=1).float()
    return hit.mean()


def measure_forward_ms(model, x, y, pos_key, pos_val, route_topm, amp: bool, iters: int = 50, warmup: int = 10) -> float:
    if not torch.cuda.is_available():
        return float("nan")

    model.eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available()), dtype=torch.float16):
                _ = model(x, y=y, pos_key=pos_key, pos_val=pos_val, route_topm=route_topm)

    torch.cuda.synchronize()

    total = 0.0
    for _ in range(iters):
        starter.record()
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available()), dtype=torch.float16):
                _ = model(x, y=y, pos_key=pos_key, pos_val=pos_val, route_topm=route_topm)
        ender.record()
        torch.cuda.synchronize()
        total += starter.elapsed_time(ender)

    return total / iters


# -----------------------------
# Embedding
# -----------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.emb(x)


# -----------------------------
# RFT Lite (only)
# -----------------------------
class RFT_Lite(nn.Module):
    """
    RFT Lite:
      - KV-only depthwise scout
      - query-conditioned recency slopes (router + pointer)
      - pointer/value mixture
      - BUGFIX: mask PAD/unfilled KV slots so router/topk never selects them
    """
    def __init__(self, vocab_size: int, d_model: int, num_classes: int, max_len: int, pad_mask_value: float = -1e9):
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.max_len = int(max_len)
        self.pad_mask_value = float(pad_mask_value)

        self.embed = TokenEmbedding(vocab_size, self.d_model)
        self.pos = nn.Embedding(self.max_len, self.d_model)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.k_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.head = nn.Linear(self.d_model, self.num_classes)

        self.recency_beta_raw = nn.Parameter(torch.tensor(-3.0))
        self.pointer_recency_beta_raw = nn.Parameter(torch.tensor(-3.0))
        self.router_beta_q = nn.Linear(self.d_model, 1, bias=True)
        self.ptr_beta_q = nn.Linear(self.d_model, 1, bias=True)

        self.scout_dw = nn.Conv1d(
            self.d_model, self.d_model,
            kernel_size=3, padding=1,
            groups=self.d_model, bias=False
        )
        self.scout_ln = nn.LayerNorm(self.d_model)

        Pmax = (self.max_len - 2) // 2
        pair_starts_full = torch.arange(0, 2 * Pmax, 2)
        pos_norm_full = torch.arange(Pmax, dtype=torch.float32) / max(float(Pmax - 1), 1.0)
        self.register_buffer("pair_starts_full", pair_starts_full, persistent=False)
        self.register_buffer("pos_norm_full", pos_norm_full, persistent=False)

    def _pos(self, idx: torch.Tensor) -> torch.Tensor:
        return self.pos(idx.clamp(0, self.max_len - 1))

    def _kv_scout(self, kv_keys: torch.Tensor) -> torch.Tensor:
        t = kv_keys.transpose(1, 2)  # [B,D,P]
        t = self.scout_dw(t)
        out = t.transpose(1, 2)      # [B,P,D]
        return self.scout_ln(kv_keys + out)

    def _router_beta(self, q_raw: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.recency_beta_raw)
        bq = self.router_beta_q(q_raw).squeeze(-1)  # [B]
        return F.softplus(base + bq)

    def _ptr_beta(self, q_raw: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.pointer_recency_beta_raw)
        bq = self.ptr_beta_q(q_raw).squeeze(-1)      # [B]
        return F.softplus(base + bq)

    def forward(self, x, y=None, pos_key=None, pos_val=None, route_topm=32, **kwargs):
        B, L = x.shape
        D = self.d_model
        device = x.device
        PAD = 0

        P = (L - 2) // 2
        pair_starts = self.pair_starts_full[:P].to(device)
        pos_norm = self.pos_norm_full[:P].to(device)

        key_pos = pair_starts
        val_pos = pair_starts + 1

        x_keys = x[:, key_pos]   # [B,P]
        x_vals = x[:, val_pos]   # [B,P]
        x_q = x[:, -1]
        x_qm = x[:, -2]

        valid = (x_keys != PAD) & (x_vals != PAD)  # IMPORTANT

        kv_keys = self.embed(x_keys)               # [B,P,D]
        ev0 = self.embed(x_vals)                   # [B,P,D]
        q_raw = self.embed(x_q)                    # [B,D]
        qm_tok = self.embed(x_qm).unsqueeze(1)     # [B,1,D]

        kv_keys = self._kv_scout(kv_keys)

        key_tok = kv_keys + self._pos(pair_starts).unsqueeze(0)
        q = self.q_proj(q_raw)
        k = self.k_proj(key_tok)
        scores = torch.einsum("bd,bpd->bp", q, k)

        beta = self._router_beta(q_raw)  # [B]
        scores = scores + beta.unsqueeze(1) * pos_norm.unsqueeze(0)

        # PAD mask fix
        scores = scores.masked_fill(~valid, self.pad_mask_value)

        aux = {"router_scores": scores, "recency_beta": beta.detach()}

        if pos_key is not None:
            target_idx = torch.div(pos_key, 2, rounding_mode="trunc").long()
            aux["router_target"] = target_idx
            pred = torch.argmax(scores, dim=1)
            aux["router_acc1"] = (pred == target_idx).float().mean()
            aux["router_recall_at_m"] = recall_at_k(scores, target_idx, k=min(int(route_topm), P))

        M = min(int(route_topm), P)
        top_idx = torch.topk(scores, k=M, dim=1).indices  # [B,M]

        starts = pair_starts[top_idx]
        starts, sort_idx = torch.sort(starts, dim=1)
        top_idx = torch.gather(top_idx, 1, sort_idx)

        denom = max(float(pair_starts[P - 1].item()), 1.0)
        starts_norm = starts.float() / denom
        aux["starts_norm"] = starts_norm.detach()

        ek_tok = torch.gather(kv_keys, 1, top_idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()
        ev_tok = torch.gather(ev0,     1, top_idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()

        ek = ek_tok + self._pos(starts)
        ev = ev_tok + self._pos(starts + 1)

        fact = torch.sigmoid(self.k_gate(ek)) * self.v_proj(ev)
        fact = fact + ek

        q_h = self.q_proj(q_raw).unsqueeze(1)
        fact_h = fact
        aux["transformer_tokens"] = torch.tensor(0, device=device)

        pointer_logits = torch.einsum("bid,bjd->bij", q_h, fact_h).squeeze(1) / (D ** 0.5)

        beta_ptr = self._ptr_beta(q_raw)
        pointer_logits = pointer_logits + beta_ptr.unsqueeze(1) * starts_norm
        aux["pointer_recency_beta"] = beta_ptr.detach()
        aux["pointer_logits"] = pointer_logits

        fact_logits = self.head(fact_h)
        aux["fact_logits"] = fact_logits

        w = torch.softmax(pointer_logits, dim=-1).unsqueeze(-1)
        logits = (w * fact_logits).sum(dim=1)

        ignore_index = -100
        ptr_target = torch.full((B,), ignore_index, device=device, dtype=torch.long)

        match = None
        if pos_val is not None:
            match = ((starts + 1) == pos_val.unsqueeze(-1))
        elif pos_key is not None:
            match = (starts == pos_key.unsqueeze(-1))

        if match is not None:
            hit_mask = match.any(dim=1)
            aux["hit_mask"] = hit_mask
            if hit_mask.any():
                hit_rows = hit_mask.nonzero(as_tuple=True)[0]
                hit_cols = match[hit_rows].nonzero(as_tuple=True)[1]
                ptr_target[hit_rows] = hit_cols

        aux["pointer_target"] = ptr_target
        return logits, aux


# -----------------------------
# Results
# -----------------------------
@dataclass
class RunResult:
    dataset: str
    model: str
    seed: int
    seq_len: int
    params: int
    steps: int
    batch_size: int
    lr: float
    train_sec: float
    val_loss: float
    val_acc: float
    transformer_tokens: int
    kv_est_mb: float
    peak_mem_mb: float
    extra: Dict[str, float]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval_ddp(
    seq_len: int,
    seed: int,
    rank: int,
    local_rank: int,
    batch_size: int,
    steps_train: int,
    steps_val: int,
    lr: float,
    d_model: int,
    key_vocab: int,
    val_vocab: int,
    num_facts: int,
    num_decoys: int,
    target_repeats: int,
    amp: bool,
    grad_accum: int,
    max_len: int,
    route_topm: int,
    router_loss_weight: float,
    ddp_enabled: bool,
    ptr_alpha: float,
    ptr_warmup: int,
    ptr_cap: float,
    lambda_topm: float,
    margin_topm: float,
):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(seed + 1234 * rank)

    # IMPORTANT DDP FIX: compute vocab_size purely from args (identical across ranks)
    vocab_size = key_vocab + val_vocab + 2
    num_classes = val_vocab

    train_gen = SyntheticKVRetrievalWithPos(
        seq_len=seq_len,
        key_vocab=key_vocab,
        val_vocab=val_vocab,
        num_facts=num_facts,
        num_decoys=num_decoys,
        batch_size=batch_size,
        steps=steps_train,
        target_repeats=target_repeats,
        device=device,
        base_seed=seed,
        rank=rank,
    )
    val_gen = SyntheticKVRetrievalWithPos(
        seq_len=seq_len,
        key_vocab=key_vocab,
        val_vocab=val_vocab,
        num_facts=num_facts,
        num_decoys=num_decoys,
        batch_size=batch_size,
        steps=steps_val,
        target_repeats=target_repeats,
        device=device,
        base_seed=seed + 999,
        rank=rank,
    )

    model = RFT_Lite(
        vocab_size=vocab_size,
        d_model=d_model,
        num_classes=num_classes,
        max_len=max_len,
    ).to(device)

    # (optional) print shapes per rank once for sanity
    if step_train_debug_once(rank):
        print(f"[RANK {rank}] vocab_size={vocab_size} d_model={d_model} num_classes={num_classes} max_len={max_len}")

    params = count_params(model)

    ddp_on = dist_ready(ddp_enabled)
    if ddp_on:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    # param groups: betas faster LR
    beta_params: List[torch.nn.Parameter] = []
    main_params: List[torch.nn.Parameter] = []
    for n, p in ddp_model.named_parameters():
        if ("recency_beta_raw" in n) or ("pointer_recency_beta_raw" in n) or ("router_beta_q" in n) or ("ptr_beta_q" in n):
            beta_params.append(p)
        else:
            main_params.append(p)

    opt = torch.optim.AdamW(
        [{"params": main_params, "lr": lr}, {"params": beta_params, "lr": 10 * lr}],
        weight_decay=0.01,
    )

    ce = nn.CrossEntropyLoss()
    ce_ptr = nn.CrossEntropyLoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and torch.cuda.is_available()))

    # =========================
    # Training
    # =========================
    ddp_model.train()
    reset_cuda_peak()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    opt.zero_grad(set_to_none=True)
    step_i = 0

    for x, y, pos_key, pos_val in train_gen:
        step_i += 1
        with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available()), dtype=torch.float16):
            cur_topm = max(1, int(route_topm))
            logits, aux = ddp_model(x, y=y, pos_key=pos_key, pos_val=pos_val, route_topm=cur_topm)

            if is_main(rank) and step_i == 1:
                print(f"[DEBUG] logits.shape={tuple(logits.shape)} y.shape={tuple(y.shape)}")

            loss = torch.tensor(0.0, device=device)

            # Router CE
            if ("router_scores" in aux) and ("router_target" in aux):
                loss = loss + float(router_loss_weight) * ce(aux["router_scores"].float(), aux["router_target"])

            # Top-M hinge
            if (lambda_topm > 0) and ("router_scores" in aux) and ("router_target" in aux):
                r_scores = aux["router_scores"]
                r_tgt = aux["router_target"]
                topk_vals = torch.topk(r_scores, k=cur_topm, dim=1).values
                cutoff = topk_vals[:, -1]
                s_t = r_scores.gather(1, r_tgt.unsqueeze(1)).squeeze(1)
                loss = loss + float(lambda_topm) * torch.relu(cutoff - s_t + float(margin_topm)).mean()

            # Pointer loss (hits only) w/ warmup
            if ("pointer_logits" in aux) and ("pointer_target" in aux) and ("starts_norm" in aux):
                ptr_tgt = aux["pointer_target"]
                mask = (ptr_tgt != -100)
                if mask.any():
                    ptr_logits = aux["pointer_logits"][mask].float()
                    pl = ce_ptr(ptr_logits, ptr_tgt[mask])
                    tgt_rec = aux["starts_norm"][mask].gather(1, ptr_tgt[mask].unsqueeze(1)).squeeze(1)

                    ptr_scale = 1.0 if step_i > int(ptr_warmup) else 0.0
                    w = (1.0 + float(ptr_alpha) * tgt_rec)
                    if float(ptr_cap) > 0.0:
                        w = w.clamp(1.0, float(ptr_cap))
                    loss = loss + ptr_scale * (w.detach() * pl).mean()

            # Aux value loss on hit rows
            if ("fact_logits" in aux) and ("pointer_target" in aux):
                ptr_tgt = aux["pointer_target"]
                mask = (ptr_tgt != -100)
                if mask.any():
                    hit_rows = mask.nonzero(as_tuple=True)[0]
                    tgt_cols = ptr_tgt[mask]
                    gold_fact_logits = aux["fact_logits"][hit_rows, tgt_cols].float()
                    loss = loss + 1.0 * ce(gold_fact_logits, y[hit_rows])

            # Value CE
            loss = loss + ce(logits, y)
            loss = loss / max(1, grad_accum)

        scaler.scale(loss).backward()

        if (step_i % grad_accum) == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

    if (step_i % grad_accum) != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = float(time.perf_counter() - t0)
    peak_mem = float(get_cuda_peak_mb())

    # =========================
    # Validation
    # =========================
    ddp_model.eval()
    val_loss_sum = 0.0
    val_acc_real_sum = 0.0
    n_batches = 0.0

    router_acc1_sum = 0.0
    router_rec_sum = 0.0
    pointer_acc_sum = 0.0
    pointer_batches = 0.0
    hit_rate_sum = 0.0
    ptr_beta_sum = 0.0

    val_topm = max(1, int(route_topm))
    infer_ms_oracle = float("nan")
    infer_ms_real = float("nan")

    with torch.no_grad():
        for x, y, pos_key, pos_val in val_gen:
            if is_main(rank) and n_batches == 0.0 and torch.cuda.is_available():
                infer_ms_oracle = measure_forward_ms(ddp_model, x, y, pos_key, pos_val, route_topm=val_topm, amp=amp)
                infer_ms_real = measure_forward_ms(ddp_model, x, None, pos_key, pos_val, route_topm=val_topm, amp=amp)

            with torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available()), dtype=torch.float16):
                logits_real, aux_real = ddp_model(x, y=None, pos_key=pos_key, pos_val=pos_val, route_topm=val_topm)
                logits, aux = ddp_model(x, y=y, pos_key=pos_key, pos_val=pos_val, route_topm=val_topm)

                val_acc_real_sum += float(batch_acc(logits_real.float(), y).item())

                loss = torch.tensor(0.0, device=device)
                if "router_scores" in aux and "router_target" in aux:
                    loss = loss + float(router_loss_weight) * ce(aux["router_scores"].float(), aux["router_target"])
                loss = loss + ce(logits, y)

            val_loss_sum += float(loss.item())
            n_batches += 1.0

            if "router_acc1" in aux:
                router_acc1_sum += float(aux["router_acc1"].item())
            if "router_recall_at_m" in aux:
                router_rec_sum += float(aux["router_recall_at_m"].item())
            if "hit_mask" in aux:
                hit_rate_sum += float(aux["hit_mask"].float().mean().item())

            pb = aux.get("pointer_recency_beta", None)
            if pb is not None:
                ptr_beta_sum += float(pb.mean().item())

            ptr_tgt = aux.get("pointer_target", None)
            if ("pointer_logits" in aux) and (ptr_tgt is not None):
                mask = (ptr_tgt != -100)
                if mask.any():
                    p_pred = torch.argmax(aux["pointer_logits"][mask], dim=-1)
                    p_acc = (p_pred == ptr_tgt[mask]).float().mean()
                    pointer_acc_sum += float(p_acc.item())
                    pointer_batches += 1.0

    val_loss = val_loss_sum / max(n_batches, 1.0)
    val_acc_real = val_acc_real_sum / max(n_batches, 1.0)

    extra = {
        "router_acc1": float(router_acc1_sum / max(n_batches, 1.0)),
        "router_recall_at_m": float(router_rec_sum / max(n_batches, 1.0)),
        "hit_rate": float(hit_rate_sum / max(n_batches, 1.0)),
        "pointer_acc": float(pointer_acc_sum / max(pointer_batches, 1.0)),
        "ptr_beta": float(ptr_beta_sum / max(n_batches, 1.0)),
        "infer_ms_oracle_y": float(infer_ms_oracle),
        "infer_ms_real_no_y": float(infer_ms_real),
        "route_topm": float(route_topm),
        "target_repeats": float(target_repeats),
    }

    return RunResult(
        dataset="synthetic_kv_last_occurrence",
        model="rft_lite",
        seed=seed,
        seq_len=seq_len,
        params=params,
        steps=steps_train,
        batch_size=batch_size,
        lr=lr,
        train_sec=train_sec,
        val_loss=val_loss,
        val_acc=val_acc_real,
        transformer_tokens=0,
        kv_est_mb=0.0,
        peak_mem_mb=peak_mem,
        extra=extra,
    )


def save_results(outdir: str, results: List[RunResult], rank: int, args_dict: Optional[dict] = None):
    if not is_main(rank):
        return
    os.makedirs(outdir, exist_ok=True)

    if args_dict is not None:
        with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, sort_keys=True)

    csv_path = os.path.join(outdir, "results.csv")
    jsonl_path = os.path.join(outdir, "results.jsonl")

    fieldnames = list(asdict(results[0]).keys())
    extra_keys = sorted({k for r in results for k in r.extra.keys()})

    with open(csv_path, "w", encoding="utf-8") as f:
        header = [k for k in fieldnames if k != "extra"] + [f"extra_{k}" for k in extra_keys]
        f.write(",".join(header) + "\n")
        for r in results:
            row = asdict(r)
            base = [str(row[k]) for k in fieldnames if k != "extra"]
            extra = [str(r.extra.get(k, "")) for k in extra_keys]
            f.write(",".join(base + extra) + "\n")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    print("=" * 90)
    print(f"[DONE] Wrote {csv_path}")
    print(f"[DONE] Wrote {jsonl_path}")
    if args_dict is not None:
        print(f"[DONE] Wrote {os.path.join(outdir, 'args.json')}")


# small helper to avoid spamming the print
_printed_rank_cfg = set()
def step_train_debug_once(rank: int) -> bool:
    if rank in _printed_rank_cfg:
        return False
    _printed_rank_cfg.add(rank)
    return True


def main():
    ddp_enabled, rank, world, local_rank = ddp_init()
    torch.autograd.set_detect_anomaly(False)

    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--root_outdir", type=str, default="./runs")
        ap.add_argument("--exp_name", type=str, default="rft_lite_lastocc")

        ap.add_argument("--seq_lens", type=str, default="2048")
        ap.add_argument("--seeds", type=str, default="0")

        ap.add_argument("--batch_size", type=int, default=16)
        ap.add_argument("--steps_train", type=int, default=2000)
        ap.add_argument("--steps_val", type=int, default=400)
        ap.add_argument("--lr", type=float, default=3e-4)

        ap.add_argument("--d_model", type=int, default=384)

        ap.add_argument("--key_vocab", type=int, default=32)
        ap.add_argument("--val_vocab", type=int, default=64)
        ap.add_argument("--num_facts", type=int, default=16)
        ap.add_argument("--num_decoys", type=int, default=128)
        ap.add_argument("--target_repeats", type=int, default=1)

        ap.add_argument("--amp", action="store_true")
        ap.add_argument("--grad_accum", type=int, default=1)
        ap.add_argument("--max_len", type=int, default=8192)

        ap.add_argument("--route_topm", type=int, default=32)
        ap.add_argument("--router_loss_weight", type=float, default=1.0)

        ap.add_argument("--ptr_alpha", type=float, default=2.0)
        ap.add_argument("--ptr_warmup", type=int, default=500)
        ap.add_argument("--ptr_cap", type=float, default=0.0)

        ap.add_argument("--lambda_topm", type=float, default=0.0)
        ap.add_argument("--margin_topm", type=float, default=0.1)

        args = ap.parse_args()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_root = os.path.join(args.root_outdir, f"{ts}__{args.exp_name}")
        if is_main(rank):
            os.makedirs(exp_root, exist_ok=True)
            print(f"[INFO] exp_root={exp_root}")

        seq_lens = [int(x) for x in args.seq_lens.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]

        if is_main(rank):
            eff_bs = args.batch_size * args.grad_accum * world
            print(f"[INFO] ddp_enabled={ddp_enabled} world={world} rank={rank} local_rank={local_rank}")
            print(f"[INFO] device={'cuda' if torch.cuda.is_available() else 'cpu'} amp={args.amp}")
            print(f"[INFO] eff_batch={eff_bs} (bs={args.batch_size} accum={args.grad_accum} world={world})")
            print(f"[INFO] seq_lens={seq_lens} seeds={seeds}")
            print(f"[INFO] d_model={args.d_model}")
            print(f"[INFO] facts={args.num_facts} decoys={args.num_decoys} repeats={args.target_repeats}")
            print(f"[INFO] topm={args.route_topm} router_w={args.router_loss_weight} lambda_topm={args.lambda_topm}")

        for L in seq_lens:
            for seed in seeds:
                if is_main(rank):
                    print("=" * 90)
                    print(f"RUN model=rft_lite seq_len={L} seed={seed}")

                r = train_and_eval_ddp(
                    seq_len=L,
                    seed=seed,
                    rank=rank,
                    local_rank=local_rank,
                    batch_size=args.batch_size,
                    steps_train=args.steps_train,
                    steps_val=args.steps_val,
                    lr=args.lr,
                    d_model=args.d_model,
                    key_vocab=args.key_vocab,
                    val_vocab=args.val_vocab,
                    num_facts=args.num_facts,
                    num_decoys=args.num_decoys,
                    target_repeats=args.target_repeats,
                    amp=args.amp,
                    grad_accum=args.grad_accum,
                    max_len=args.max_len,
                    route_topm=args.route_topm,
                    router_loss_weight=args.router_loss_weight,
                    ddp_enabled=ddp_enabled,
                    ptr_alpha=args.ptr_alpha,
                    ptr_warmup=args.ptr_warmup,
                    ptr_cap=args.ptr_cap,
                    lambda_topm=args.lambda_topm,
                    margin_topm=args.margin_topm,
                )

                if is_main(rank):
                    print(
                        f"params={r.params:,} val_acc={r.val_acc:.4f} val_loss={r.val_loss:.4f} "
                        f"peakMB={r.peak_mem_mb:.1f} "
                        f"router_acc1={r.extra.get('router_acc1',0):.4f} "
                        f"recall@M={r.extra.get('router_recall_at_m',0):.4f} "
                        f"hit_rate={r.extra.get('hit_rate',0):.4f} "
                        f"pointer_acc={r.extra.get('pointer_acc',0):.4f} "
                        f"infer_ms(y)={r.extra.get('infer_ms_oracle_y', float('nan')):.3f} "
                        f"infer_ms(no_y)={r.extra.get('infer_ms_real_no_y', float('nan')):.3f}"
                    )

                run_name = (
                    f"model=rft_lite"
                    f"__L={L}"
                    f"__seed={seed}"
                    f"__decoys={args.num_decoys}"
                    f"__repeats={args.target_repeats}"
                    f"__topm={args.route_topm}"
                    f"__lr={args.lr}"
                    f"__bs={args.batch_size}"
                    f"__steps={args.steps_train}"
                )
                run_dir = os.path.join(exp_root, run_name)
                save_results(run_dir, [r], rank, args_dict=vars(args))

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()

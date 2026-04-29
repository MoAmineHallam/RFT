import json
"""
niah_batch.py â€” RULER-faithful NIAH training batches for RFT-LM.

Uses the EXACT RULER S-NIAH template (Hsieh et al., 2024) so that
training and evaluation use the same format. This makes results
directly comparable to Titans and other papers citing RULER.

RULER template:
  Instruction: "Some special magic {type} are hidden within the
    following text. Make sure to memorize it. I will quiz you
    about the {type} afterwards."
  Needle: "The special magic {type} for {key} is: {value}."
  Query: "What are all the special magic {type} for {key}
    mentioned in the provided text?"
  Answer prefix: "The special magic {type} for {key} mentioned
    in the provided text are"

Layout per example (2-chunk for memory training):
  Chunk 0 (context):
    [RULER instruction prefix]
    [C4 distractor text with RULER-format needles embedded]
    -> Processed through model, written to memory bank.

  Chunk 1 (query):
    [C4 distractor text]
    [RULER query + answer prefix]
    [value tokens (teacher-forced)]
    -> Forward pass computes multi-token LM loss over ALL value
      tokens, embedding alignment loss, and supervised retrieval loss.

Multi-token value support:
  Value tokens (e.g. 3-4 tokens for 7-digit numbers) are appended
  to the query chunk AFTER the answer prefix, enabling teacher-forced
  training over the full value sequence.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from RFT_LM import MemoryBank


MAX_VAL_TOKENS = 8  # enough for 7-digit numbers (3-4 BPE) or UUIDs


def _ruler_type_label(value_type: str) -> str:
    """Map value_type config to RULER type_needle_v string."""
    if value_type in ("digit2", "short_int", "numbers"):
        return "numbers"
    if value_type == "uuids":
        return "uuids"
    if value_type == "words":
        return "words"
    return "numbers"
def _ruler_type_singular(type_v: str) -> str:
    """Singular form for RULER needle sentences."""
    return {"numbers": "number", "words": "word", "uuids": "uuid"}.get(type_v, type_v)

@dataclass
class NIAHBatchConfig:
    chunk_size: int = 512
    batch_size: int = 4
    num_needles: int = 3       # max needles per context chunk
    value_type: str = "digit2"  # default value_type (used if value_type_mix is None)
    # ── Curriculum mixing (added for RULER-general training) ─────────────
    # If set, randomly sample needle count and value type per example.
    # This prevents overfitting to a single (needle_count, value_type) regime.
    needle_count_mix: tuple = (1, 2, 3)  # sample uniformly from these
    value_type_mix: tuple = ("digit2", "short_int", "numbers")  # sample per example
    # Add the official RULER instruction prefix to match eval-time format
    add_instruct_prefix: bool = True
    # Inject prompt-format variation: with prob, train without prefix
    no_prefix_prob: float = 0.5  # 50/50 with-prefix vs no-prefix


class NIAHBatchGen:
    """Generates RULER-faithful NIAH training batches."""

    def __init__(
        self,
        cfg: NIAHBatchConfig,
        tokenizer,
        distractor_tokens: List[int],
        device: torch.device,
        seed: int,
    ):
        self.cfg = cfg
        self.tok = tokenizer
        self.dist = distractor_tokens
        self.device = device
        self.rng = random.Random(seed)
        import json
        self._wlist = list(json.load(open("english_words.json")).values())
        self.type_v = _ruler_type_label(cfg.value_type)
        self.type_v_sg = _ruler_type_singular(self.type_v)

        self.instr_toks = []  # continuation style: no instruction prefix

        assert len(distractor_tokens) > cfg.chunk_size * 4, (
            f"Need more distractor tokens ({len(distractor_tokens)}) "
            f"for chunk_size={cfg.chunk_size}"
        )

    # â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _gen_value(self, value_type=None) -> str:
        vt = value_type if value_type is not None else self.cfg.value_type
        if vt == "digit2":
            return str(self.rng.randint(10, 99))
        if vt == "short_int":
            return str(self.rng.randint(100, 999))
        if vt == "numbers":
            return str(self.rng.randint(1_000_000, 9_999_999))
        if vt == "uuids":
            import uuid
            return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        raise ValueError(f"Unsupported value_type={vt}")

    def _dist_chunk(self, n: int) -> List[int]:
        """Random contiguous slice of distractor tokens, length >= n."""
        mx = max(0, len(self.dist) - n - 1)
        s = self.rng.randint(0, mx) if mx > 0 else 0
        out = list(self.dist[s : s + n])
        while len(out) < n:
            s2 = self.rng.randint(0, max(0, len(self.dist) - 128))
            out.extend(self.dist[s2 : s2 + 128])
        return out[:n]

    # â”€â”€ iterator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def __iter__(self):
        return self

    def __next__(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Returns 6 tensors:
          ctx:         [B, cs]  context chunk (instruction + needles in distractor)
          qry:         [B, cs]  query chunk (RULER query + answer prefix + value tokens)
          tgt_mem:     [B]      target memory slot index in context chunk
          qry_tok:     [B]      probe position (last token of answer prefix "are")
          tgt_id:      [B]      first value token ID (for embedding alignment loss)
          val_targets: [B, MAX_VAL_TOKENS]  all value token IDs, padded with -100
        """
        B = self.cfg.batch_size
        cs = self.cfg.chunk_size
        tv = self.type_v

        ctx = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        qry = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        tgt_mem = torch.zeros(B, dtype=torch.long, device=self.device)
        qry_tok = torch.zeros(B, dtype=torch.long, device=self.device)
        tgt_id = torch.zeros(B, dtype=torch.long, device=self.device)
        val_targets = torch.full(
            (B, MAX_VAL_TOKENS), -100, dtype=torch.long, device=self.device
        )

        for b in range(B):
            # ── per-example sampling: needle count + value type + prefix on/off ──
            # This is the key change vs original: instead of fixed num_needles=3
            # and fixed value_type=digit2, sample fresh per example.
            n_needles = self.rng.choice(self.cfg.needle_count_mix)
            ex_value_type = self.rng.choice(self.cfg.value_type_mix)
            ex_type_v = _ruler_type_label(ex_value_type)  # "numbers"
            ex_type_v_sg = _ruler_type_singular(ex_type_v)  # "number"
            # Decide prefix on/off for this example
            use_prefix = self.cfg.add_instruct_prefix and (self.rng.random() > self.cfg.no_prefix_prob)
            if use_prefix:
                instr_str = (
                    f"Some special magic {ex_type_v} are hidden within the following text. "
                    f"Make sure to memorize it. I will quiz you about the {ex_type_v} afterwards.\n"
                )
                ex_instr_toks = self.tok.encode(instr_str, add_special_tokens=False)
            else:
                ex_instr_toks = []

            # ── generate needles (RULER format) ──────────────────────
            needles = []
            for i in range(n_needles):
                key = f"{self.rng.choice(self._wlist)}-{self.rng.choice(self._wlist)}"
                val = self._gen_value(value_type=ex_value_type)
                # RULER needle format
                needle_str = f" The special magic {ex_type_v_sg} for {key} is: {val}."
                needle_toks = self.tok.encode(needle_str, add_special_tokens=False)
                # Full value tokens for teacher forcing
                val_tok_ids = self.tok.encode(f" {val}", add_special_tokens=False)
                # Prefix up to value (for tgt_mem position calculation)
                prefix_str = f" The special magic {ex_type_v_sg} for {key} is:"
                prefix_toks = self.tok.encode(prefix_str, add_special_tokens=False)
                needles.append(
                    {
                        "key": key,
                        "val": val,
                        "toks": needle_toks,
                        "val_toks": val_tok_ids,
                        "val_first_tok": val_tok_ids[0],
                        "prefix_len": len(prefix_toks),
                    }
                )

            tgt_i = self.rng.randrange(len(needles))
            tgt_n = needles[tgt_i]

            # ── build context chunk ──────────────────────────────────
            # Layout: [instruction_prefix] [distractor + needles]
            total_needle_len = sum(len(n["toks"]) for n in needles)
            dist_budget = max(16, cs - len(ex_instr_toks) - total_needle_len)
            dist_toks = self._dist_chunk(dist_budget)

            seg_len = len(dist_toks) // (len(needles) + 1)
            ctx_toks: List[int] = list(ex_instr_toks)  # start with instruction (or empty)
            val_pos = -1

            for i, n in enumerate(needles):
                seg_start = i * seg_len
                ctx_toks.extend(dist_toks[seg_start : seg_start + seg_len])
                if i == tgt_i:
                    val_pos = len(ctx_toks) + n["prefix_len"]
                ctx_toks.extend(n["toks"])

            # remaining distractor
            ctx_toks.extend(dist_toks[len(needles) * seg_len :])

            # truncate / pad to chunk_size
            ctx_toks = ctx_toks[:cs]
            if len(ctx_toks) < cs:
                ctx_toks.extend(self._dist_chunk(cs - len(ctx_toks)))
                ctx_toks = ctx_toks[:cs]

            val_pos = max(0, min(val_pos, cs - 1))

            # ── build query chunk (RULER query + answer prefix + value) ──
            query_str = (
                f"\nWhat are all the special magic {ex_type_v} for {tgt_n['key']} "
                f"mentioned in the provided text? "
                f"The special magic {ex_type_v} for {tgt_n['key']} mentioned in "
                f"the provided text are"
            )
            query_toks = self.tok.encode(query_str, add_special_tokens=False)
            val_toks = tgt_n["val_toks"]
            n_val = len(val_toks)

            # Reserve space for query + answer prefix + value tokens
            q_dist_budget = max(16, cs - len(query_toks) - n_val)
            q_dist = self._dist_chunk(q_dist_budget)
            q_toks = q_dist[:q_dist_budget] + query_toks + val_toks
            q_toks = q_toks[:cs]
            if len(q_toks) < cs:
                pad = self._dist_chunk(cs - len(q_toks))
                q_toks = pad + q_toks
                q_toks = q_toks[:cs]

            # Probe = last token of answer prefix ("are"), just before value
            q_idx = len(q_toks) - n_val - 1

            ctx[b] = torch.tensor(ctx_toks[:cs], dtype=torch.long)
            qry[b] = torch.tensor(q_toks[:cs], dtype=torch.long)
            tgt_mem[b] = val_pos
            qry_tok[b] = q_idx
            tgt_id[b] = tgt_n["val_first_tok"]

            # Fill multi-token value targets
            for j, vt in enumerate(val_toks[:MAX_VAL_TOKENS]):
                val_targets[b, j] = vt

        return ctx, qry, tgt_mem, qry_tok, tgt_id, val_targets


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Training step
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def niah_retrieval_step(
    raw_model,
    batch,
    loss_weights: dict,
    do_backward: bool = True,
) -> Tuple[float, dict]:
    """
    One NIAH-format training step through the RFT-LM.

    Combines:
      (a) Multi-token LM loss â€” teacher-forced over ALL value tokens.
      (b) Embedding alignment loss (first value token, at probe position).
      (c) Supervised retrieval loss â€” router/OCR training.

    Returns (total_loss_value, metrics_dict).
    """
    ctx_chunk, qry_chunk, target_mem_idx, query_token_idx, target_token_ids, val_targets = batch
    B, L0 = ctx_chunk.shape
    _, L1 = qry_chunk.shape
    device = ctx_chunk.device
    batch_idx = torch.arange(B, device=device)

    memory_bank = MemoryBank(max_entries=L0)

    # â”€â”€ Chunk 0: populate memory WITH gradients â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out0 = raw_model(
        ctx_chunk,
        memory_keys=None,
        memory_vals=None,
        memory_positions=None,
        pos_offset=0,
        return_memory_state=True,
        compute_ocr_loss=False,
        detach_memory=False,
    )
    memory_bank.add(out0["new_mem_keys"], out0["new_mem_vals"], 0, L0)
    mem_k, mem_v, mem_pos = memory_bank.get_state()

    # â”€â”€ Chunk 1: manual forward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model = raw_model
    x = model.embed(qry_chunk) * model.embed_scale
    x = model.drop(x)
    rope_cos, rope_sin = model.rope(L1, offset=L0)

    x_at_mem = None
    mem_ctx = None
    for i, layer in enumerate(model.layers):
        x = layer(x, rope_cos, rope_sin, L0)
        if i == model.memory_layer_idx:
            x_at_mem = x
            if model.use_memory and mem_k is not None and mem_k.shape[1] > 0:
                mem_ctx = model.memory_layer(x, mem_k, mem_v, mem_pos, L0)
                x = x + mem_ctx

    x = model.ln_f(x)
    logits = model.lm_head(x)  # [B, L1, vocab]

    # â”€â”€ Multi-token LM loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lm_loss = torch.tensor(0.0, device=device)
    n_correct = 0
    n_total = 0

    for j in range(val_targets.shape[1]):
        valid = val_targets[:, j] != -100
        if not valid.any():
            break
        pos_j = (query_token_idx[valid] + j).clamp(max=L1 - 1)
        logits_j = logits[batch_idx[valid], pos_j]
        targets_j = val_targets[valid, j]
        lm_loss = lm_loss + F.cross_entropy(logits_j, targets_j, reduction='sum')
        n_total += valid.sum().item()
        with torch.no_grad():
            n_correct += (logits_j.argmax(-1) == targets_j).sum().item()

    if n_total > 0:
        lm_loss = lm_loss / n_total
    lm_acc = n_correct / max(n_total, 1)

    # â”€â”€ Embedding alignment loss (first value token only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    embed_loss = torch.tensor(0.0, device=device)
    embed_acc = torch.tensor(0.0, device=device)
    if mem_ctx is not None:
        target_val = mem_ctx[batch_idx, query_token_idx]
        target_embed = model.embed.weight[target_token_ids]

        cos_sim = F.cosine_similarity(target_val, target_embed, dim=-1)
        embed_loss = (1.0 - cos_sim).mean()

        val_logits = torch.matmul(target_val, model.embed.weight.T)
        embed_ce = F.cross_entropy(val_logits, target_token_ids)
        embed_loss = embed_loss + embed_ce

        with torch.no_grad():
            embed_pred = val_logits.argmax(dim=-1)
            embed_acc = (embed_pred == target_token_ids).float().mean()

    # â”€â”€ Supervised retrieval loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ret = model.memory_layer.supervised_retrieval_loss(
        x_at_mem,
        mem_k,
        mem_v,
        mem_pos,
        L0,
        target_mem_idx=target_mem_idx,
        query_token_idx=query_token_idx,
    )

    # â”€â”€ Combined loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total = (
        loss_weights.get("lm_ce", 1.0) * lm_loss
        + loss_weights.get("decode", 5.0) * embed_loss
        + loss_weights.get("router_ce", 1.0) * ret["router_ce"]
        + loss_weights.get("topm_hinge", 0.25) * ret["topm_hinge"]
        + loss_weights.get("pointer_ce", 0.5) * ret["pointer_ce"]
        + loss_weights.get("ocr_contrastive", 0.2) * ret["ocr_contrastive"]
    )

    if do_backward:
        total.backward()

    metrics = {
        "niah_loss": float(total.item()),
        "niah_lm_ce": float(lm_loss.item()),
        "niah_lm_acc": float(lm_acc),
        "niah_embed_loss": float(embed_loss.item()),
        "niah_embed_acc": float(embed_acc.item()),
        "niah_router_ce": float(ret["router_ce"].item()),
        "niah_topm_hinge": float(ret["topm_hinge"].item()),
        "niah_recall_at_m": float(ret["recall_at_m"].item()),
        "niah_pointer_acc": float(ret["pointer_acc"].item()),
        "niah_fused_acc": float(ret["fused_acc"].item()),
    }
    return float(total.item()), metrics


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Memorizing-Transformers NIAH step (Gate 1A head-to-head)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def mt_niah_retrieval_step(
    raw_model,
    batch,
    loss_weights: dict,
    do_backward: bool = True,
) -> Tuple[float, dict]:
    """
    NIAH training step for MTLM (Memorizing-Transformers baseline).
    RULER-faithful template + multi-token value support.
    """
    ctx_chunk, qry_chunk, target_mem_idx, query_token_idx, target_token_ids, val_targets = batch
    B, L0 = ctx_chunk.shape
    _, L1 = qry_chunk.shape
    device = ctx_chunk.device
    batch_idx = torch.arange(B, device=device)

    memory_bank = MemoryBank(max_entries=L0)

    out0 = raw_model(
        ctx_chunk,
        memory_keys=None,
        memory_vals=None,
        memory_positions=None,
        pos_offset=0,
        return_memory_state=True,
        detach_memory=False,
    )
    memory_bank.add(out0["new_mem_keys"], out0["new_mem_vals"], 0, L0)
    mem_k, mem_v, mem_pos = memory_bank.get_state()

    model = raw_model
    x = model.embed(qry_chunk) * model.embed_scale
    x = model.drop(x)
    rope_cos, rope_sin = model.rope(L1, offset=L0)

    mem_ctx = None
    for i, layer in enumerate(model.layers):
        x = layer(x, rope_cos, rope_sin, L0)
        if i == model.memory_layer_idx:
            if model.use_memory and mem_k is not None and mem_k.shape[1] > 0:
                mem_ctx = model.memory_layer(x, mem_k, mem_v, mem_pos, L0)
                x = x + mem_ctx

    x = model.ln_f(x)
    logits = model.lm_head(x)

    # Multi-token LM loss
    lm_loss = torch.tensor(0.0, device=device)
    n_correct = 0
    n_total = 0

    for j in range(val_targets.shape[1]):
        valid = val_targets[:, j] != -100
        if not valid.any():
            break
        pos_j = (query_token_idx[valid] + j).clamp(max=L1 - 1)
        logits_j = logits[batch_idx[valid], pos_j]
        targets_j = val_targets[valid, j]
        lm_loss = lm_loss + F.cross_entropy(logits_j, targets_j, reduction='sum')
        n_total += valid.sum().item()
        with torch.no_grad():
            n_correct += (logits_j.argmax(-1) == targets_j).sum().item()

    if n_total > 0:
        lm_loss = lm_loss / n_total
    lm_acc = n_correct / max(n_total, 1)

    embed_loss = torch.tensor(0.0, device=device)
    embed_acc = torch.tensor(0.0, device=device)
    decode_w = loss_weights.get("decode", 0.0)
    if mem_ctx is not None and decode_w > 0:
        target_val = mem_ctx[batch_idx, query_token_idx]
        target_embed = model.embed.weight[target_token_ids]

        cos_sim = F.cosine_similarity(target_val, target_embed, dim=-1)
        embed_loss = (1.0 - cos_sim).mean()

        val_logits = torch.matmul(target_val, model.embed.weight.T)
        embed_ce = F.cross_entropy(val_logits, target_token_ids)
        embed_loss = embed_loss + embed_ce

        with torch.no_grad():
            embed_pred = val_logits.argmax(dim=-1)
            embed_acc = (embed_pred == target_token_ids).float().mean()

    total = loss_weights.get("lm_ce", 1.0) * lm_loss + decode_w * embed_loss

    if do_backward:
        total.backward()

    metrics = {
        "niah_loss": float(total.item()),
        "niah_lm_ce": float(lm_loss.item()),
        "niah_lm_acc": float(lm_acc),
        "niah_embed_loss": float(embed_loss.item()),
        "niah_embed_acc": float(embed_acc.item()),
        "niah_router_ce": 0.0,
        "niah_topm_hinge": 0.0,
        "niah_recall_at_m": 0.0,
        "niah_pointer_acc": 0.0,
        "niah_fused_acc": 0.0,
    }
    return float(total.item()), metrics
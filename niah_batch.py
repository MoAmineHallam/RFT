"""
niah_batch.py — Natural-language NIAH training batches for RFT-LM.

Bridges the gap between synthetic KV-retrieval (abstract token IDs) and
RULER S-NIAH evaluation (natural language). Uses the same
"The magic number alphaXXXXX is YY." format as eval, embedded in C4
distractor text, split across two chunks so the memory layer must
retrieve the needle from a previous chunk.

Layout per example:
  - Chunk 0 (context): C4 distractor text with needle sentences embedded
    at random depths.  Processed through RFT-LM and written into the
    memory bank (WITH gradients so mem_key_proj/mem_val_proj learn).
  - Chunk 1 (query): C4 distractor text ending with the probe
    "\\nThe magic number <key> is".  Full forward through the model to
    get both supervised_retrieval_loss (trains router/OCR) and LM loss
    at the probe position (trains the output head to emit the value).
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from RFT_LM import MemoryBank


@dataclass
class NIAHBatchConfig:
    chunk_size: int = 512
    batch_size: int = 4
    num_needles: int = 3       # needles per context chunk
    value_type: str = "digit2"  # match eval config
    max_val_tokens: int = 8     # supervise up to N BPE tokens per value (full-sequence loss)


class NIAHBatchGen:
    """Generates NIAH-format training batches with natural-language needles."""

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
        assert len(distractor_tokens) > cfg.chunk_size * 4, (
            f"Need more distractor tokens ({len(distractor_tokens)}) "
            f"for chunk_size={cfg.chunk_size}"
        )

    # ── helpers ──────────────────────────────────────────────────────

    def _gen_value(self) -> str:
        vt = self.cfg.value_type
        if vt == "digit2":
            return str(self.rng.randint(10, 99))
        if vt == "short_int":
            return str(self.rng.randint(100, 999))
        if vt == "numbers":
            return str(self.rng.randint(1_000_000, 9_999_999))
        raise ValueError(f"Unsupported value_type={vt}")

    def _dist_chunk(self, n: int) -> List[int]:
        """Random contiguous slice of distractor tokens, length ≥ n."""
        mx = max(0, len(self.dist) - n - 1)
        s = self.rng.randint(0, mx) if mx > 0 else 0
        out = list(self.dist[s : s + n])
        while len(out) < n:
            s2 = self.rng.randint(0, max(0, len(self.dist) - 128))
            out.extend(self.dist[s2 : s2 + 128])
        return out[:n]

    # ── iterator ─────────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (ctx, qry, tgt_mem, val_positions, val_targets, val_lengths) where
          val_positions: [B, max_val_tokens] absolute positions in qry chunk whose
                         logits should predict the corresponding val_targets entry
                         (padded with 0 where invalid)
          val_targets:   [B, max_val_tokens] target token ids for each value BPE
                         (padded with -100 so F.cross_entropy ignores them)
          val_lengths:   [B] number of supervised value tokens per example (>=1)
        """
        B = self.cfg.batch_size
        cs = self.cfg.chunk_size
        K = self.cfg.max_val_tokens

        ctx = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        qry = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        tgt_mem = torch.zeros(B, dtype=torch.long, device=self.device)
        val_positions = torch.zeros(B, K, dtype=torch.long, device=self.device)
        val_targets = torch.full((B, K), -100, dtype=torch.long, device=self.device)
        val_lengths = torch.zeros(B, dtype=torch.long, device=self.device)

        for b in range(B):
            # ── generate needles ──
            needles = []
            for i in range(self.cfg.num_needles):
                key = f"alpha{self.rng.randint(0, 99999):05d}{i}"
                val = self._gen_value()
                needle_str = f" The magic number {key} is {val}."
                needle_toks = self.tok.encode(needle_str, add_special_tokens=False)
                # Full value tokens (may be 1+ BPE pieces with leading-space tokenization)
                val_tok_ids = self.tok.encode(f" {val}", add_special_tokens=False)
                prefix_toks = self.tok.encode(
                    f" The magic number {key} is", add_special_tokens=False
                )
                needles.append(
                    {
                        "key": key,
                        "val": val,
                        "toks": needle_toks,
                        "val_toks": val_tok_ids,            # FULL list, not just [0]
                        "prefix_len": len(prefix_toks),
                    }
                )

            tgt_i = self.rng.randrange(len(needles))
            tgt_n = needles[tgt_i]

            # ── build context chunk (unchanged) ──
            total_needle_len = sum(len(n["toks"]) for n in needles)
            dist_budget = max(16, cs - total_needle_len)
            dist_toks = self._dist_chunk(dist_budget)

            seg_len = len(dist_toks) // (len(needles) + 1)
            ctx_toks: List[int] = []
            val_pos = -1

            for i, n in enumerate(needles):
                seg_start = i * seg_len
                ctx_toks.extend(dist_toks[seg_start : seg_start + seg_len])
                if i == tgt_i:
                    val_pos = len(ctx_toks) + n["prefix_len"]
                ctx_toks.extend(n["toks"])

            ctx_toks.extend(dist_toks[len(needles) * seg_len :])
            ctx_toks = ctx_toks[:cs]
            if len(ctx_toks) < cs:
                ctx_toks.extend(self._dist_chunk(cs - len(ctx_toks)))
                ctx_toks = ctx_toks[:cs]
            val_pos = max(0, min(val_pos, cs - 1))

            # ── build query chunk: distractor + probe + value tokens ──
            probe_str = f"\nThe magic number {tgt_n['key']} is"
            probe_toks = self.tok.encode(probe_str, add_special_tokens=False)
            tgt_val_toks = tgt_n["val_toks"][:K]  # cap at max_val_tokens
            n_val = len(tgt_val_toks)

            # Reserve room for probe + full value at the END of the chunk
            tail_len = len(probe_toks) + n_val
            q_dist_budget = max(16, cs - tail_len)
            q_dist = self._dist_chunk(q_dist_budget)
            q_toks = q_dist + probe_toks + tgt_val_toks
            if len(q_toks) < cs:
                # Pad on the right with distractor (so the value isn't at the very end
                # adjacent to the chunk boundary — the LM head still sees a normal context)
                q_toks = q_toks + self._dist_chunk(cs - len(q_toks))
            q_toks = q_toks[:cs]

            # Position of the "is" token = last token of probe inside the chunk
            is_pos = len(q_dist) + len(probe_toks) - 1
            # logits[is_pos] should predict tgt_val_toks[0]
            # logits[is_pos + i] should predict tgt_val_toks[i] for i in 0..n_val-1
            n_supervised = 0
            for i in range(n_val):
                p = is_pos + i
                if p >= cs:
                    break
                val_positions[b, i] = p
                val_targets[b, i] = tgt_val_toks[i]
                n_supervised += 1
            val_lengths[b] = max(1, n_supervised)

            ctx[b] = torch.tensor(ctx_toks[:cs], dtype=torch.long)
            qry[b] = torch.tensor(q_toks[:cs], dtype=torch.long)
            tgt_mem[b] = val_pos

        return ctx, qry, tgt_mem, val_positions, val_targets, val_lengths


# ─────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────
def niah_retrieval_step(
    raw_model,
    batch,
    loss_weights: dict,
    do_backward: bool = True,
) -> Tuple[float, dict]:
    """
    One NIAH-format training step through the RFT-LM.

    Combines:
      (a) supervised_retrieval_loss — trains router_q/router_k + OCR to
          retrieve the correct memory slot (same losses as synth_batch.py).
      (b) LM cross-entropy at the probe position — trains the output head
          (and all downstream layers) to actually *emit* the value token,
          closing the gap between "memory can retrieve it" and "model
          actually generates it".

    Returns (total_loss_value, metrics_dict).
    """
    ctx_chunk, qry_chunk, target_mem_idx, val_positions, val_targets, val_lengths = batch
    B, L0 = ctx_chunk.shape
    _, L1 = qry_chunk.shape

    memory_bank = MemoryBank(max_entries=L0)

    # ── Chunk 0: populate memory WITH gradients ──────────────────────
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

    # ── Chunk 1: manual forward, capturing hidden states at memory layer
    model = raw_model
    if hasattr(model, "forward_manual"):
        x_at_mem, mem_ctx, logits = model.forward_manual(
            qry_chunk, mem_k, mem_v, mem_pos, L0
        )
    else:
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

    # ── Full-sequence LM loss across every value token position ─────
    # val_positions: [B, K], val_targets: [B, K] with -100 for padding.
    # logits[b, val_positions[b, i]] should predict val_targets[b, i].
    K = val_positions.shape[1]
    device = logits.device
    batch_idx_2d = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)  # [B, K]
    safe_pos = val_positions.clamp(min=0, max=L1 - 1)
    val_logits_full = logits[batch_idx_2d, safe_pos]                          # [B, K, V]
    lm_loss = F.cross_entropy(
        val_logits_full.reshape(-1, val_logits_full.size(-1)),
        val_targets.reshape(-1),
        ignore_index=-100,
    )

    with torch.no_grad():
        valid = (val_targets != -100)                                          # [B, K]
        preds = val_logits_full.argmax(dim=-1)                                 # [B, K]
        per_pos_correct = ((preds == val_targets) & valid)
        lm_acc = per_pos_correct.sum().float() / valid.sum().clamp(min=1).float()
        # Per-example full-match: ALL valid positions correct
        per_ex_correct = (per_pos_correct.sum(dim=1) == valid.sum(dim=1).clamp(min=1))
        lm_full_match_acc = per_ex_correct.float().mean()
        # Token-0 accuracy: the ONLY position where memory content is required.
        # Positions >=1 are teacher-forced digit continuations whose ground-truth
        # prefixes are visible in the input, so lm_acc overstates retrieval.
        first_valid = valid[:, 0]
        first_tok_acc = ((per_pos_correct[:, 0] & first_valid).sum().float()
                         / first_valid.sum().clamp(min=1).float())

    # ── Embedding alignment loss (only at first value position) ─────
    # Memory retrieval is most critical at position 0 (the "is" token).
    # Subsequent positions get learned via the full-sequence LM loss above.
    batch_idx_1d = torch.arange(B, device=device)
    first_pos = val_positions[:, 0].clamp(min=0, max=L1 - 1)                   # [B]
    first_tgt = val_targets[:, 0].clamp(min=0)                                 # [B], -100 -> 0 (always valid since val_lengths>=1)

    embed_loss = torch.tensor(0.0, device=device)
    embed_acc = torch.tensor(0.0, device=device)
    if mem_ctx is not None:
        target_val = mem_ctx[batch_idx_1d, first_pos]                          # [B, D]
        target_embed = model.embed.weight[first_tgt]                           # [B, D]

        cos_sim = F.cosine_similarity(target_val, target_embed, dim=-1)
        embed_loss = (1.0 - cos_sim).mean()

        val_logits_emb = torch.matmul(target_val, model.embed.weight.T)        # [B, V]
        embed_ce = F.cross_entropy(val_logits_emb, first_tgt)
        embed_loss = embed_loss + embed_ce

        with torch.no_grad():
            embed_pred = val_logits_emb.argmax(dim=-1)
            embed_acc = (embed_pred == first_tgt).float().mean()

    # ── Supervised retrieval loss (router + OCR), keyed at first value position ─
    ret = model.memory_layer.supervised_retrieval_loss(
        x_at_mem,
        mem_k,
        mem_v,
        mem_pos,
        L0,
        target_mem_idx=target_mem_idx,
        query_token_idx=first_pos,
    )

    # ── Combined loss ────────────────────────────────────────────────
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
        "niah_lm_acc": float(lm_acc.item()),                 # per-token (first-tok-equivalent if K=1)
        "niah_lm_full_match": float(lm_full_match_acc.item()),  # ALL value tokens correct (TF)
        "niah_first_tok_acc": float(first_tok_acc.item()),   # token-0 only: the true retrieval metric
        "niah_embed_loss": float(embed_loss.item()),
        "niah_embed_acc": float(embed_acc.item()),
        "niah_router_ce": float(ret["router_ce"].item()),
        "niah_topm_hinge": float(ret["topm_hinge"].item()),
        "niah_recall_at_m": float(ret["recall_at_m"].item()),
        "niah_pointer_acc": float(ret["pointer_acc"].item()),
        "niah_fused_acc": float(ret["fused_acc"].item()),
    }
    return float(total.item()), metrics


# ─────────────────────────────────────────────────────────────────────
# Memorizing-Transformers NIAH step (Gate 1A head-to-head)
# ─────────────────────────────────────────────────────────────────────
def mt_niah_retrieval_step(
    raw_model,
    batch,
    loss_weights: dict,
    do_backward: bool = True,
) -> Tuple[float, dict]:
    """
    NIAH training step for MTLM (Memorizing-Transformers baseline).

    Differences vs. niah_retrieval_step (RFT):
      - No OCR / router / pointer / topm loss (MT has no supervised router)
      - Only LM loss at the probe + optional embedding alignment loss on
        the post-retrieval memory context at the query position.

    The single knob that toggles Gate 1A's two variants is
    loss_weights['decode']:
        decode > 0  → "MT + alignment loss"
        decode = 0  → "MT (vanilla Memorizing Transformers baseline)"
    """
    ctx_chunk, qry_chunk, target_mem_idx, val_positions, val_targets, val_lengths = batch
    B, L0 = ctx_chunk.shape
    _, L1 = qry_chunk.shape

    memory_bank = MemoryBank(max_entries=L0)

    # Chunk 0: populate memory WITH gradients
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

    # Chunk 1: manual forward, capturing mem_ctx at the memory layer
    model = raw_model
    if hasattr(model, "forward_manual"):
        _x_at_mem, mem_ctx, logits = model.forward_manual(
            qry_chunk, mem_k, mem_v, mem_pos, L0
        )
    else:
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
        logits = model.lm_head(x)  # [B, L1, vocab]

    # Full-sequence LM loss across every value token position
    K = val_positions.shape[1]
    device = logits.device
    batch_idx_2d = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
    safe_pos = val_positions.clamp(min=0, max=L1 - 1)
    val_logits_full = logits[batch_idx_2d, safe_pos]                            # [B, K, V]
    lm_loss = F.cross_entropy(
        val_logits_full.reshape(-1, val_logits_full.size(-1)),
        val_targets.reshape(-1),
        ignore_index=-100,
    )
    with torch.no_grad():
        valid = (val_targets != -100)
        preds = val_logits_full.argmax(dim=-1)
        per_pos_correct = (preds == val_targets) & valid
        lm_acc = per_pos_correct.sum().float() / valid.sum().clamp(min=1).float()
        per_ex_correct = (per_pos_correct.sum(dim=1) == valid.sum(dim=1).clamp(min=1))
        lm_full_match_acc = per_ex_correct.float().mean()
        first_valid = valid[:, 0]
        first_tok_acc = ((per_pos_correct[:, 0] & first_valid).sum().float()
                         / first_valid.sum().clamp(min=1).float())

    # Embedding alignment loss (Gate 1A knob), only at first value position
    batch_idx_1d = torch.arange(B, device=device)
    first_pos = val_positions[:, 0].clamp(min=0, max=L1 - 1)
    first_tgt = val_targets[:, 0].clamp(min=0)

    embed_loss = torch.tensor(0.0, device=device)
    embed_acc = torch.tensor(0.0, device=device)
    decode_w = loss_weights.get("decode", 0.0)
    if mem_ctx is not None and decode_w > 0:
        target_val = mem_ctx[batch_idx_1d, first_pos]
        target_embed = model.embed.weight[first_tgt]

        cos_sim = F.cosine_similarity(target_val, target_embed, dim=-1)
        embed_loss = (1.0 - cos_sim).mean()

        val_logits_emb = torch.matmul(target_val, model.embed.weight.T)
        embed_ce = F.cross_entropy(val_logits_emb, first_tgt)
        embed_loss = embed_loss + embed_ce

        with torch.no_grad():
            embed_pred = val_logits_emb.argmax(dim=-1)
            embed_acc = (embed_pred == first_tgt).float().mean()

    total = loss_weights.get("lm_ce", 1.0) * lm_loss + decode_w * embed_loss

    if do_backward:
        total.backward()

    metrics = {
        "niah_loss": float(total.item()),
        "niah_lm_ce": float(lm_loss.item()),
        "niah_lm_acc": float(lm_acc.item()),
        "niah_lm_full_match": float(lm_full_match_acc.item()),
        "niah_first_tok_acc": float(first_tok_acc.item()),
        "niah_embed_loss": float(embed_loss.item()),
        "niah_embed_acc": float(embed_acc.item()),
        "niah_router_ce": 0.0,
        "niah_topm_hinge": 0.0,
        "niah_recall_at_m": 0.0,
        "niah_pointer_acc": 0.0,
        "niah_fused_acc": 0.0,
    }
    return float(total.item()), metrics

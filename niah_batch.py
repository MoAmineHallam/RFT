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
        if self.cfg.value_type == "digit2":
            return str(self.rng.randint(10, 99))
        return str(self.rng.randint(100, 999))

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.cfg.batch_size
        cs = self.cfg.chunk_size

        ctx = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        qry = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        tgt_mem = torch.zeros(B, dtype=torch.long, device=self.device)
        qry_tok = torch.zeros(B, dtype=torch.long, device=self.device)
        tgt_id = torch.zeros(B, dtype=torch.long, device=self.device)

        for b in range(B):
            # ── generate needles ──
            needles = []
            for i in range(self.cfg.num_needles):
                key = f"alpha{self.rng.randint(0, 99999):05d}{i}"
                val = self._gen_value()
                needle_str = f" The magic number {key} is {val}."
                needle_toks = self.tok.encode(needle_str, add_special_tokens=False)
                # " {val}" should be exactly 1 BPE token for digit2
                val_tok_ids = self.tok.encode(f" {val}", add_special_tokens=False)
                # prefix up to value: " The magic number {key} is"
                prefix_toks = self.tok.encode(
                    f" The magic number {key} is", add_special_tokens=False
                )
                needles.append(
                    {
                        "key": key,
                        "val": val,
                        "toks": needle_toks,
                        "val_tok": val_tok_ids[0],
                        "prefix_len": len(prefix_toks),  # offset of val inside needle
                    }
                )

            tgt_i = self.rng.randrange(len(needles))
            tgt_n = needles[tgt_i]

            # ── build context chunk ──
            total_needle_len = sum(len(n["toks"]) for n in needles)
            dist_budget = max(16, cs - total_needle_len)
            dist_toks = self._dist_chunk(dist_budget)

            seg_len = len(dist_toks) // (len(needles) + 1)
            ctx_toks: List[int] = []
            val_pos = -1

            for i, n in enumerate(needles):
                # distractor segment before needle
                seg_start = i * seg_len
                ctx_toks.extend(dist_toks[seg_start : seg_start + seg_len])
                # track value-token position for target needle
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

            # clamp target position
            val_pos = max(0, min(val_pos, cs - 1))

            # ── build query chunk ──
            probe_str = f"\nThe magic number {tgt_n['key']} is"
            probe_toks = self.tok.encode(probe_str, add_special_tokens=False)
            q_dist_budget = max(16, cs - len(probe_toks))
            q_dist = self._dist_chunk(q_dist_budget)
            q_toks = q_dist[:q_dist_budget] + probe_toks
            q_toks = q_toks[:cs]
            if len(q_toks) < cs:
                pad = self._dist_chunk(cs - len(q_toks))
                q_toks = pad + q_toks
                q_toks = q_toks[:cs]

            # query token = last position ("is" token → should predict value)
            q_idx = len(q_toks) - 1

            ctx[b] = torch.tensor(ctx_toks[:cs], dtype=torch.long)
            qry[b] = torch.tensor(q_toks[:cs], dtype=torch.long)
            tgt_mem[b] = val_pos
            qry_tok[b] = q_idx
            tgt_id[b] = tgt_n["val_tok"]

        return ctx, qry, tgt_mem, qry_tok, tgt_id


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
    ctx_chunk, qry_chunk, target_mem_idx, query_token_idx, target_token_ids = batch
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
        detach_memory=False,  # e2e gradient through memory keys/vals
    )
    memory_bank.add(out0["new_mem_keys"], out0["new_mem_vals"], 0, L0)
    mem_k, mem_v, mem_pos = memory_bank.get_state()

    # ── Chunk 1: manual forward, capturing hidden states at memory layer
    model = raw_model
    x = model.embed(qry_chunk) * model.embed_scale
    x = model.drop(x)
    rope_cos, rope_sin = model.rope(L1, offset=L0)

    x_at_mem = None
    for i, layer in enumerate(model.layers):
        x = layer(x, rope_cos, rope_sin, L0)
        if i == model.memory_layer_idx:
            # Save pre-retrieval hidden states for supervised_retrieval_loss
            x_at_mem = x
            # Memory retrieval (mirrors RFTLM.forward logic)
            if model.use_memory and mem_k is not None and mem_k.shape[1] > 0:
                mem_ctx = model.memory_layer(x, mem_k, mem_v, mem_pos, L0)
                x = x + mem_ctx

    x = model.ln_f(x)
    logits = model.lm_head(x)  # [B, L1, vocab]

    # ── LM loss at query position ────────────────────────────────────
    batch_idx = torch.arange(B, device=logits.device)
    query_logits = logits[batch_idx, query_token_idx]  # [B, vocab]
    lm_loss = F.cross_entropy(query_logits, target_token_ids)

    with torch.no_grad():
        lm_pred = query_logits.argmax(dim=-1)
        lm_acc = (lm_pred == target_token_ids).float().mean()

    # ── Supervised retrieval loss ────────────────────────────────────
    ret = model.memory_layer.supervised_retrieval_loss(
        x_at_mem,
        mem_k,
        mem_v,
        mem_pos,
        L0,
        target_mem_idx=target_mem_idx,
        query_token_idx=query_token_idx,
    )

    # ── Combined loss ────────────────────────────────────────────────
    total = (
        loss_weights.get("lm_ce", 1.0) * lm_loss
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
        "niah_lm_acc": float(lm_acc.item()),
        "niah_router_ce": float(ret["router_ce"].item()),
        "niah_topm_hinge": float(ret["topm_hinge"].item()),
        "niah_recall_at_m": float(ret["recall_at_m"].item()),
        "niah_pointer_acc": float(ret["pointer_acc"].item()),
        "niah_fused_acc": float(ret["fused_acc"].item()),
    }
    return float(total.item()), metrics

"""
RFT-LM: Transformer Language Model with RFT Sparse Memory Layer + OCR

Architecture:
  - Standard causal Transformer decoder (sliding window attention)
  - RFT Memory Layer inserted at a configurable depth
  - Memory bank stores KV representations from past chunks
  - Router selects top-M candidates from memory per query token
  - OCR (Occurrence-Contrastive Resolver) disambiguates among retrieved candidates
  - Retrieved context is merged with local attention output

This is designed to be competitive with Titans (MAC) at 125M-scale
on RULER S-NIAH and BABILong benchmarks.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# ─────────────────────────────────────────────
# Rotary Position Embedding (RoPE)
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 65536, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.cos_cached.shape[0]:
            self._build_cache(offset + seq_len)
        return (
            self.cos_cached[offset : offset + seq_len],
            self.sin_cached[offset : offset + seq_len],
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ─────────────────────────────────────────────
# Sliding Window Causal Attention
# ─────────────────────────────────────────────
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                pos_offset: int = 0) -> torch.Tensor:
        B, L, D = x.shape
        H, HD = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,L,HD]

        q, k = apply_rotary(q, k, rope_cos, rope_sin)

        # Scaled dot-product with sliding window causal mask
        scale = 1.0 / math.sqrt(HD)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,L,L]

        # Causal + sliding window mask
        row = torch.arange(L, device=x.device).unsqueeze(1)
        col = torch.arange(L, device=x.device).unsqueeze(0)
        mask = (col <= row) & (col >= row - self.window_size + 1)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,L,HD]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


# ─────────────────────────────────────────────
# RFT Memory Layer (the core contribution)
# ─────────────────────────────────────────────
class RFTMemoryLayer(nn.Module):
    """
    Sparse routed memory retrieval with OCR disambiguation.

    Given current hidden states [B, L, D] and a memory bank of past
    hidden states, this layer:
    1. Routes each query token to top-M candidates in the memory bank
    2. Applies OCR to resolve ambiguity among retrieved candidates
    3. Returns a context vector to be added to the hidden states
    """

    def __init__(self, d_model: int, top_m: int = 64, ocr_dim: int = 256,
                 ocr_alpha_init: float = 0.1, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.top_m = top_m
        self.ocr_dim = ocr_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Router projections
        self.router_q = nn.Linear(d_model, d_model, bias=False)
        self.router_k = nn.Linear(d_model, d_model, bias=False)

        # Value projection for retrieved context
        self.mem_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=True)

        # Layer norm for retrieved context
        self.mem_ln = nn.LayerNorm(d_model)

        # Recency bias (learnable, query-conditioned)
        self.recency_beta_raw = nn.Parameter(torch.tensor(-3.0))
        self.recency_beta_q = nn.Linear(d_model, 1, bias=True)

        # OCR head
        feat_dim = 4  # recency, reverse_recency, router_score_rel, rank_rel
        self.ocr_q = nn.Linear(d_model, ocr_dim, bias=False)
        self.ocr_cand = nn.Linear(d_model, ocr_dim, bias=False)
        self.ocr_feat = nn.Linear(feat_dim, ocr_dim, bias=False)
        self.ocr_ln = nn.LayerNorm(ocr_dim)
        self.ocr_pair = nn.Sequential(
            nn.Linear(ocr_dim * 4, ocr_dim),
            nn.GELU(),
            nn.Linear(ocr_dim, 1),
        )
        self.ocr_alpha = nn.Parameter(torch.tensor(float(ocr_alpha_init)))

        # Gating scalar for residual addition (init > 0 so memory is active from start)
        self.mem_gate_alpha = nn.Parameter(torch.tensor(0.1))

    def _router_beta(self, q: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.recency_beta_raw)
        bq = self.recency_beta_q(q).squeeze(-1)  # [B, L]
        return F.softplus(base + bq)

    def _ocr_scores(self, q: torch.Tensor, cand: torch.Tensor,
                    router_scores: torch.Tensor, recency: torch.Tensor) -> torch.Tensor:
        """
        q: [B*L, D]  (flattened query tokens)
        cand: [B*L, M, D]  (retrieved candidate hidden states)
        router_scores: [B*L, M]
        recency: [B*L, M]  (normalized position of each candidate)
        """
        BL, M, _ = cand.shape
        qv = self.ocr_q(q).unsqueeze(1).expand(-1, M, -1)
        cv = self.ocr_cand(cand)
        rev_rec = 1.0 - recency
        rel_rank = torch.linspace(0.0, 1.0, M, device=cand.device).view(1, M).expand(BL, M)
        max_router = router_scores.max(dim=1, keepdim=True).values
        feat = torch.stack([
            recency,
            rev_rec,
            router_scores - max_router,
            rel_rank,
        ], dim=-1)
        zv = self.ocr_ln(qv + cv + self.ocr_feat(feat))
        mean_all = zv.mean(dim=1, keepdim=True)
        others = (mean_all * M - zv) / max(M - 1, 1)
        pair_in = torch.cat([zv, others, zv - others, zv * others], dim=-1)
        return self.ocr_alpha * self.ocr_pair(pair_in).squeeze(-1)

    def forward(self, x: torch.Tensor, memory_keys: torch.Tensor,
                memory_vals: torch.Tensor, memory_positions: torch.Tensor,
                current_pos_start: int) -> torch.Tensor:
        """
        x: [B, L, D] current chunk hidden states
        memory_keys: [B, N_mem, D] past hidden states (key projections)
        memory_vals: [B, N_mem, D] past hidden states (value projections)
        memory_positions: [B, N_mem] absolute positions of memory entries
        current_pos_start: int, absolute position of first token in current chunk
        Returns: [B, L, D] context from memory (to be added residually)
        """
        B, L, D = x.shape
        N_mem = memory_keys.shape[1]

        if N_mem == 0:
            return torch.zeros_like(x)

        M = min(self.top_m, N_mem)

        # Router scores
        q = self.router_q(x)  # [B, L, D]
        scores = torch.einsum("bld,bnd->bln", q, memory_keys) / math.sqrt(self.d_model)  # [B, L, N_mem]

        # Recency bias
        max_pos = max(current_pos_start + L - 1, 1)
        recency_all = memory_positions.float() / max_pos  # [B, N_mem], normalized 0..1
        beta = self._router_beta(x)  # [B, L]
        scores = scores + beta.unsqueeze(-1) * recency_all.unsqueeze(1)  # bias toward recent

        # Top-M selection
        top_scores, top_idx = torch.topk(scores, k=M, dim=-1)  # [B, L, M]

        # Sort by position (for consistency) — memory-efficient gather
        top_positions = torch.zeros(B, L, M, device=x.device, dtype=memory_positions.dtype)
        for b in range(B):
            top_positions[b] = memory_positions[b][top_idx[b]]  # [L, M]
        sort_order = top_positions.argsort(dim=-1)
        top_idx = torch.gather(top_idx, 2, sort_order)
        top_scores = torch.gather(top_scores, 2, sort_order)
        top_positions = torch.gather(top_positions, 2, sort_order)

        # Gather candidate values — memory-efficient (no full expand)
        # top_idx is [B, L, M] with indices into dim=1 of memory_vals [B, N_mem, D]
        # We gather per-batch using the flattened index
        cand_vals = torch.zeros(B, L, M, D, device=x.device, dtype=memory_vals.dtype)
        for b in range(B):
            # top_idx[b] is [L, M], memory_vals[b] is [N_mem, D]
            idx_b = top_idx[b]  # [L, M]
            cand_vals[b] = memory_vals[b][idx_b]  # [L, M, D] via advanced indexing

        # Recency features for OCR
        recency_feat = top_positions.float() / max(max_pos, 1)

        # OCR scoring
        BL = B * L
        q_flat = x.reshape(BL, D)
        cand_flat = cand_vals.reshape(BL, M, D)
        scores_flat = top_scores.reshape(BL, M)
        recency_flat = recency_feat.reshape(BL, M)

        ocr_s = self._ocr_scores(q_flat, cand_flat, scores_flat, recency_flat)
        attn_logits = scores_flat + ocr_s  # [BL, M]

        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [BL, M, 1]
        retrieved = (attn_weights * cand_flat).sum(dim=1)  # [BL, D]
        retrieved = retrieved.reshape(B, L, D)

        # Gate and project
        gate = torch.sigmoid(self.gate(x))
        retrieved = self.mem_ln(retrieved)
        out = self.out_proj(gate * retrieved)

        return torch.tanh(self.mem_gate_alpha) * out

    def ocr_contrastive_loss(
        self, x: torch.Tensor, memory_keys: torch.Tensor,
        memory_vals: torch.Tensor, memory_positions: torch.Tensor,
        current_pos_start: int, margin: float = 0.10,
    ) -> torch.Tensor:
        """
        Contrastive loss for OCR: encourages OCR to assign higher scores
        to candidates with higher router scores (proxy for correctness).

        Returns scalar loss (0 if no memory available).
        """
        B, L, D = x.shape
        N_mem = memory_keys.shape[1]
        if N_mem < 2:
            return torch.tensor(0.0, device=x.device)

        M = min(self.top_m, N_mem)
        if M < 2:
            return torch.tensor(0.0, device=x.device)

        # Router scores (same as forward)
        q = self.router_q(x)
        scores = torch.einsum("bld,bnd->bln", q, memory_keys) / math.sqrt(self.d_model)
        max_pos = max(current_pos_start + L - 1, 1)
        recency_all = memory_positions.float() / max_pos
        beta = self._router_beta(x)
        scores = scores + beta.unsqueeze(-1) * recency_all.unsqueeze(1)
        top_scores, top_idx = torch.topk(scores, k=M, dim=-1)

        # Gather candidate values
        cand_vals = torch.zeros(B, L, M, D, device=x.device, dtype=memory_vals.dtype)
        for b in range(B):
            cand_vals[b] = memory_vals[b][top_idx[b]]

        # Recency features
        top_positions = torch.zeros(B, L, M, device=x.device, dtype=memory_positions.dtype)
        for b in range(B):
            top_positions[b] = memory_positions[b][top_idx[b]]
        recency_feat = top_positions.float() / max(max_pos, 1)

        # OCR scores
        BL = B * L
        q_flat = x.reshape(BL, D)
        cand_flat = cand_vals.reshape(BL, M, D)
        scores_flat = top_scores.reshape(BL, M)
        recency_flat = recency_feat.reshape(BL, M)

        ocr_s = self._ocr_scores(q_flat, cand_flat, scores_flat, recency_flat)

        # Contrastive: top-1 router candidate should beat others by margin
        # Use router rank as proxy for "correct" candidate
        best_ocr = ocr_s[:, 0:1]  # highest router-score candidate
        others_ocr = ocr_s[:, 1:]  # rest
        violations = F.relu(margin - (best_ocr - others_ocr))
        return violations.mean()

    def supervised_retrieval_loss(
        self,
        x: torch.Tensor,                # [B, L, D] query-chunk hidden states
        memory_keys: torch.Tensor,      # [B, N_mem, D]
        memory_vals: torch.Tensor,      # [B, N_mem, D]
        memory_positions: torch.Tensor, # [B, N_mem]
        current_pos_start: int,
        target_mem_idx: torch.Tensor,   # [B] index into the N_mem dim (the correct slot)
        query_token_idx: torch.Tensor,  # [B] which token in L is the query
        topm_margin: float = 0.2,
        ocr_margin: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Supervised retrieval loss for synthetic KV-retrieval batches.

        Given that we know the correct memory slot per example, this trains
        router_q/router_k (router_ce + top-M hinge) and the OCR head
        (pointer_ce + OCR hinge vs hardest negative). This mirrors the
        multi-term loss from RFT_ocr.py that produced recall ~1.00.
        """
        B, L, D = x.shape
        N_mem = memory_keys.shape[1]
        device = x.device
        batch_idx = torch.arange(B, device=device)

        # Gather the query token per example: [B, D]
        q_tok = x[batch_idx, query_token_idx]               # [B, D]
        q = self.router_q(q_tok)                            # [B, D]

        # Router scores over the full memory bank: [B, N_mem], scaled like attention.
        scores = torch.einsum("bd,bnd->bn", q, memory_keys) / math.sqrt(self.d_model)
        max_pos = max(current_pos_start + L - 1, 1)
        recency_all = memory_positions.float() / max_pos    # [B, N_mem]
        beta = self._router_beta(q_tok.unsqueeze(1)).squeeze(1)  # [B]
        scores = scores + beta.unsqueeze(-1) * recency_all

        # (1) Router CE: highest score must be the target slot
        router_ce = F.cross_entropy(scores, target_mem_idx)

        # (2) Top-M hinge: target must be inside top-M by at least topm_margin
        M = min(self.top_m, N_mem)
        topk_scores = torch.topk(scores, k=M, dim=-1).values      # [B, M]
        cutoff = topk_scores[:, -1]                                # [B]
        s_true = scores.gather(1, target_mem_idx.unsqueeze(1)).squeeze(1)
        topm_hinge = F.relu(cutoff - s_true + topm_margin).mean()

        # Natural top-M set (before any replacement) — used for honest recall@M
        top_scores_nat, top_idx_nat = torch.topk(scores, k=M, dim=-1)    # [B, M]
        target_in_top_nat = (top_idx_nat == target_mem_idx.unsqueeze(1)).any(dim=1)  # [B]

        # Force target into candidate set at the correct (true) score so OCR can
        # learn to rank it; do NOT demote it by keeping a stale lower score.
        top_idx = top_idx_nat.clone()
        top_scores = top_scores_nat.clone()
        repl_rows = (~target_in_top_nat).nonzero(as_tuple=True)[0]
        if repl_rows.numel() > 0:
            top_idx[repl_rows, -1] = target_mem_idx[repl_rows]
            top_scores[repl_rows, -1] = scores[repl_rows].gather(
                1, target_mem_idx[repl_rows].unsqueeze(1)
            ).squeeze(1)

        # Gather candidate values & positions
        cand_vals = torch.zeros(B, M, D, device=device, dtype=memory_vals.dtype)
        top_positions = torch.zeros(B, M, device=device, dtype=memory_positions.dtype)
        for b in range(B):
            cand_vals[b] = memory_vals[b][top_idx[b]]
            top_positions[b] = memory_positions[b][top_idx[b]]
        recency_feat = top_positions.float() / max(max_pos, 1)

        # OCR scoring over the candidate set
        ocr_s = self._ocr_scores(q_tok, cand_vals, top_scores, recency_feat)  # [B, M]
        # Inference-time retrieval weight (for metric only)
        attn_logits = top_scores + ocr_s                                       # [B, M]
        # Training signal: OCR alone must rank target among top-M. If we add
        # top_scores here, OCR can never reorder candidates — router dominates
        # because ocr_alpha*ocr_s << top_scores in magnitude.
        ptr_logits = ocr_s                                                     # [B, M]

        # Locate the target inside top-M (guaranteed by replacement above)
        target_pos_in_top = (top_idx == target_mem_idx.unsqueeze(1)).float().argmax(dim=1)  # [B]

        # Pointer CE and OCR contrastive: only meaningful on rows where the
        # target was NATURALLY in top-M — otherwise the top-M hinge must first
        # pull the target up before OCR can discriminate it.
        if target_in_top_nat.any():
            good_rows = target_in_top_nat.nonzero(as_tuple=True)[0]
            pl = ptr_logits[good_rows]
            tp = target_pos_in_top[good_rows]
            pointer_ce = F.cross_entropy(pl, tp)
            s_true_final = pl.gather(1, tp.unsqueeze(1)).squeeze(1)
            nm = (top_idx[good_rows] != target_mem_idx[good_rows].unsqueeze(1))
            s_neg = pl.masked_fill(~nm, float("-inf"))
            s_hard = s_neg.max(dim=1).values
            ocr_contrastive = F.relu(ocr_margin - s_true_final + s_hard).mean()
        else:
            pointer_ce = torch.tensor(0.0, device=device)
            ocr_contrastive = torch.tensor(0.0, device=device)

        # Accuracy metrics (for logging) — honest, computed on natural top-M
        with torch.no_grad():
            router_top1_correct = (scores.argmax(dim=-1) == target_mem_idx).float().mean()
            router_topm_hit = target_in_top_nat.float().mean()
            # pointer_acc uses OCR-only ranking (what the loss actually trains)
            pointer_correct = (ptr_logits.argmax(dim=-1) == target_pos_in_top).float().mean()
            # fused_acc = what inference actually produces (router+OCR)
            fused_correct = (attn_logits.argmax(dim=-1) == target_pos_in_top).float().mean()

        return {
            "router_ce": router_ce,
            "topm_hinge": topm_hinge,
            "pointer_ce": pointer_ce,
            "ocr_contrastive": ocr_contrastive,
            "router_top1_acc": router_top1_correct,
            "recall_at_m": router_topm_hit,
            "pointer_acc": pointer_correct,
            "fused_acc": fused_correct,
        }


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int,
                 ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, window_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin, pos_offset)
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────
# RFT-LM: Full Language Model
# ─────────────────────────────────────────────
class RFTLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        window_size: int = 1024,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_len: int = 65536,
        # RFT Memory config
        memory_layer_idx: int = 6,  # insert memory layer after this layer
        use_memory: bool = True,
        mem_top_m: int = 64,
        ocr_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.window_size = window_size
        self.memory_layer_idx = memory_layer_idx
        self.use_memory = use_memory
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_len)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size, ff_mult, dropout)
            for _ in range(n_layers)
        ])

        if use_memory:
            self.memory_layer = RFTMemoryLayer(
                d_model=d_model,
                top_m=mem_top_m,
                ocr_dim=ocr_dim,
            )
            # Projections to create memory entries from hidden states
            self.mem_key_proj = nn.Linear(d_model, d_model, bias=False)
            self.mem_val_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        memory_keys: Optional[torch.Tensor] = None,  # [B, N_mem, D]
        memory_vals: Optional[torch.Tensor] = None,  # [B, N_mem, D]
        memory_positions: Optional[torch.Tensor] = None,  # [B, N_mem]
        pos_offset: int = 0,
        return_memory_state: bool = False,
        compute_ocr_loss: bool = False,
        ocr_margin: float = 0.10,
        detach_memory: bool = True,
    ) -> dict:
        B, L = input_ids.shape
        D = self.d_model

        x = self.embed(input_ids) * self.embed_scale
        x = self.drop(x)

        rope_cos, rope_sin = self.rope(L, offset=pos_offset)

        new_mem_keys = None
        new_mem_vals = None
        ocr_loss = None

        for i, layer in enumerate(self.layers):
            x = layer(x, rope_cos, rope_sin, pos_offset)

            # Insert memory layer after the designated layer
            if (i == self.memory_layer_idx) and self.use_memory:
                # Generate memory entries from current hidden state
                if return_memory_state:
                    x_mem = x.detach() if detach_memory else x
                    new_mem_keys = self.mem_key_proj(x_mem)  # [B, L, D]
                    new_mem_vals = self.mem_val_proj(x_mem)  # [B, L, D]

                # Retrieve from memory if available
                if memory_keys is not None and memory_keys.shape[1] > 0:
                    mem_ctx = self.memory_layer(
                        x, memory_keys, memory_vals,
                        memory_positions, pos_offset,
                    )
                    x = x + mem_ctx

                    # OCR contrastive loss (trains OCR to disambiguate)
                    if compute_ocr_loss:
                        ocr_loss = self.memory_layer.ocr_contrastive_loss(
                            x.detach(), memory_keys, memory_vals,
                            memory_positions, pos_offset, margin=ocr_margin,
                        )

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]

        out = {"logits": logits}
        if return_memory_state and new_mem_keys is not None:
            out["new_mem_keys"] = new_mem_keys
            out["new_mem_vals"] = new_mem_vals
        if ocr_loss is not None:
            out["ocr_loss"] = ocr_loss
        return out


# ─────────────────────────────────────────────
# Memorizing Transformers kNN Memory Layer (Wu et al. 2022)
# ─────────────────────────────────────────────
#
# Plain top-K dot-product retrieval over a memory of past (k, v) pairs,
# followed by softmax attention and a gated residual add. NO recency
# bias, NO OCR, NO supervised routing. Intentionally minimal so we can
# test whether embedding alignment loss is the unlock, not the router.
class MTMemoryLayer(nn.Module):
    """
    Memorizing-Transformers-style kNN memory (Wu et al., 2022).

    For each query position, retrieves the top-K nearest past hidden
    states by dot-product similarity, attends over their values, and
    merges the result into the residual stream through a learned gate.
    """

    def __init__(self, d_model: int, top_k: int = 32, gate_alpha_init: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k

        # Separate Q/K/V projections for the kNN pathway
        self.mem_q = nn.Linear(d_model, d_model, bias=False)
        self.mem_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Gated residual scalar (same parameterization as RFTMemoryLayer)
        self.mem_gate_alpha = nn.Parameter(torch.tensor(float(gate_alpha_init)))

    def forward(self, x: torch.Tensor, memory_keys: torch.Tensor,
                memory_vals: torch.Tensor, memory_positions: torch.Tensor,
                current_pos_start: int) -> torch.Tensor:
        """
        x: [B, L, D] current chunk hidden states
        memory_keys: [B, N_mem, D]
        memory_vals: [B, N_mem, D]
        memory_positions: [B, N_mem]  (unused — MT has no recency bias)
        current_pos_start: int
        Returns: [B, L, D] memory context to be added residually.
        """
        B, L, D = x.shape
        N_mem = memory_keys.shape[1]
        if N_mem == 0:
            return torch.zeros_like(x)

        K = min(self.top_k, N_mem)

        q = self.mem_q(x)  # [B, L, D]
        scores = torch.einsum("bld,bnd->bln", q, memory_keys) / math.sqrt(D)  # [B, L, N_mem]

        # Guard against NaN/Inf from diverged models
        scores = scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        top_scores, top_idx = torch.topk(scores, k=K, dim=-1)  # [B, L, K]
        top_idx = top_idx.clamp(0, N_mem - 1)

        # Gather candidate values — reshape to [B, L*K] and gather along dim=1
        # of memory_vals [B, N_mem, D] directly (avoids expand on stride-0 dim).
        idx_flat = top_idx.reshape(B, L * K, 1).expand(B, L * K, D)  # [B, L*K, D]
        cand_vals = torch.gather(memory_vals, dim=1, index=idx_flat)  # [B, L*K, D]
        cand_vals = cand_vals.reshape(B, L, K, D)

        attn_weights = F.softmax(top_scores, dim=-1).unsqueeze(-1)  # [B, L, K, 1]
        retrieved = (attn_weights * cand_vals).sum(dim=2)  # [B, L, D]

        # Final safety: clamp any residual NaN from corrupted weights
        retrieved = self.mem_ln(retrieved)
        out = self.out_proj(retrieved)
        out = out.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tanh(self.mem_gate_alpha) * out


# ─────────────────────────────────────────────
# MTLM: Memorizing-Transformers-style baseline (for Gate 1A head-to-head)
# ─────────────────────────────────────────────
class MTLM(nn.Module):
    """
    Sliding-window transformer LM with a Memorizing-Transformers-style
    kNN memory layer inserted at `memory_layer_idx`.

    Interface-compatible with RFTLM:
      - forward(...) returns a dict with 'logits' and (optionally)
        'new_mem_keys', 'new_mem_vals'
      - supports MemoryBank + train_step_chunked

    Intentionally does NOT include:
      - OCR head / supervised retrieval loss
      - Recency bias
      - Synthetic KV objectives

    The `niah_decode_w` alignment loss (applied externally) is the only
    training-time difference between the "with alignment" and "without
    alignment" variants used for Gate 1A.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        window_size: int = 512,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_len: int = 65536,
        memory_layer_idx: int = 6,
        use_memory: bool = True,
        mem_top_k: int = 32,
        mem_gate_alpha_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.window_size = window_size
        self.memory_layer_idx = memory_layer_idx
        self.use_memory = use_memory
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_len)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size, ff_mult, dropout)
            for _ in range(n_layers)
        ])

        if use_memory:
            self.memory_layer = MTMemoryLayer(
                d_model=d_model,
                top_k=mem_top_k,
                gate_alpha_init=mem_gate_alpha_init,
            )
            # Memory entry projections — matches RFTLM naming so MemoryBank
            # and train_step_chunked work unchanged.
            self.mem_key_proj = nn.Linear(d_model, d_model, bias=False)
            self.mem_val_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        self._init_weights()

        # Expose mem_gate_alpha at top level so --mem_gate_alpha_init CLI
        # override can find it without special-casing.
        if use_memory:
            # (already on self.memory_layer)
            pass

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_keys: Optional[torch.Tensor] = None,
        memory_vals: Optional[torch.Tensor] = None,
        memory_positions: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
        return_memory_state: bool = False,
        compute_ocr_loss: bool = False,  # accepted for API compat, ignored
        ocr_margin: float = 0.10,
        detach_memory: bool = True,
    ) -> dict:
        B, L = input_ids.shape
        D = self.d_model

        x = self.embed(input_ids) * self.embed_scale
        x = self.drop(x)

        rope_cos, rope_sin = self.rope(L, offset=pos_offset)

        new_mem_keys = None
        new_mem_vals = None

        for i, layer in enumerate(self.layers):
            x = layer(x, rope_cos, rope_sin, pos_offset)

            if (i == self.memory_layer_idx) and self.use_memory:
                if return_memory_state:
                    x_mem = x.detach() if detach_memory else x
                    new_mem_keys = self.mem_key_proj(x_mem).nan_to_num(0.0)
                    new_mem_vals = self.mem_val_proj(x_mem).nan_to_num(0.0)

                if memory_keys is not None and memory_keys.shape[1] > 0:
                    mem_ctx = self.memory_layer(
                        x, memory_keys, memory_vals,
                        memory_positions, pos_offset,
                    )
                    x = x + mem_ctx

        x = self.ln_f(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if return_memory_state and new_mem_keys is not None:
            out["new_mem_keys"] = new_mem_keys
            out["new_mem_vals"] = new_mem_vals
        return out


# ─────────────────────────────────────────────
# Baseline: Standard Transformer LM (no memory)
# ─────────────────────────────────────────────
class BaselineTransformerLM(nn.Module):
    """Same architecture but with full causal attention and no memory layer."""

    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, ff_mult: int = 4, dropout: float = 0.0,
                 max_len: int = 65536):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_len)
        self.drop = nn.Dropout(dropout)

        # Full attention (window_size = max_len effectively)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size=max_len,
                             ff_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, pos_offset: int = 0, **kwargs) -> dict:
        B, L = input_ids.shape
        x = self.embed(input_ids) * self.embed_scale
        x = self.drop(x)
        rope_cos, rope_sin = self.rope(L, offset=pos_offset)
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, pos_offset)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return {"logits": logits}


# ─────────────────────────────────────────────
# Memory Bank Manager
# ─────────────────────────────────────────────
class MemoryBank:
    """Manages growing memory bank across chunks during inference/training."""

    def __init__(self, max_entries: int = 32768):
        self.max_entries = max_entries
        self.keys: Optional[torch.Tensor] = None   # [B, N, D]
        self.vals: Optional[torch.Tensor] = None    # [B, N, D]
        self.positions: Optional[torch.Tensor] = None  # [B, N]

    def reset(self):
        self.keys = None
        self.vals = None
        self.positions = None

    def add(self, new_keys: torch.Tensor, new_vals: torch.Tensor,
            chunk_start_pos: int, chunk_len: int):
        """Add a chunk's hidden states to the memory bank."""
        B, L, D = new_keys.shape
        device = new_keys.device

        new_positions = torch.arange(chunk_start_pos, chunk_start_pos + chunk_len,
                                     device=device).unsqueeze(0).expand(B, -1)

        if self.keys is None:
            self.keys = new_keys
            self.vals = new_vals
            self.positions = new_positions
        else:
            self.keys = torch.cat([self.keys, new_keys], dim=1)
            self.vals = torch.cat([self.vals, new_vals], dim=1)
            self.positions = torch.cat([self.positions, new_positions], dim=1)

        # Evict oldest if over limit
        if self.keys.shape[1] > self.max_entries:
            keep = self.keys.shape[1] - self.max_entries
            self.keys = self.keys[:, keep:]
            self.vals = self.vals[:, keep:]
            self.positions = self.positions[:, keep:]

    def get_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.keys, self.vals, self.positions

    @property
    def size(self) -> int:
        return 0 if self.keys is None else self.keys.shape[1]


# ─────────────────────────────────────────────
# Chunked Training Step
# ─────────────────────────────────────────────
def train_step_chunked(
    model: RFTLM,
    input_ids: torch.Tensor,  # [B, total_len]
    chunk_size: int,
    memory_bank: MemoryBank,
    do_backward: bool = True,
    ocr_loss_weight: float = 0.05,
    ocr_margin: float = 0.10,
) -> Tuple[torch.Tensor, dict]:
    """
    Process a long sequence in chunks, building memory as we go.
    Each chunk does its own backward pass to avoid OOM from graph accumulation.
    Returns a detached average loss and metrics.
    """
    B, total_len = input_ids.shape
    device = input_ids.device

    memory_bank.reset()
    total_loss_val = 0.0
    total_ocr_loss_val = 0.0
    total_tokens = 0
    n_chunks = 0

    for start in range(0, total_len - 1, chunk_size):
        end = min(start + chunk_size, total_len - 1)
        chunk_input = input_ids[:, start:end]
        chunk_target = input_ids[:, start + 1:end + 1]
        L = chunk_input.shape[1]

        mem_k, mem_v, mem_pos = memory_bank.get_state()

        out = model(
            chunk_input,
            memory_keys=mem_k,
            memory_vals=mem_v,
            memory_positions=mem_pos,
            pos_offset=start,
            return_memory_state=True,
            compute_ocr_loss=(ocr_loss_weight > 0),
            ocr_margin=ocr_margin,
        )

        logits = out["logits"]  # [B, L, V]
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk_target.reshape(-1),
            ignore_index=-100,
        )

        # Combine LM loss with OCR contrastive loss
        loss = lm_loss
        ocr_loss_val = 0.0
        if "ocr_loss" in out and ocr_loss_weight > 0:
            loss = loss + ocr_loss_weight * out["ocr_loss"]
            ocr_loss_val = float(out["ocr_loss"].item())

        # Backward per chunk — keeps memory constant
        if do_backward:
            n_total_chunks = max((total_len - 1) // chunk_size, 1)
            (loss / n_total_chunks).backward()

        total_loss_val += float(lm_loss.item()) * L
        total_ocr_loss_val += ocr_loss_val * L
        total_tokens += L
        n_chunks += 1

        # Update memory bank (detached — no gradient through memory)
        if "new_mem_keys" in out:
            memory_bank.add(
                out["new_mem_keys"].detach(),
                out["new_mem_vals"].detach(),
                start, L,
            )

    avg_loss = total_loss_val / max(total_tokens, 1)
    avg_ocr_loss = total_ocr_loss_val / max(total_tokens, 1)
    # Return a dummy tensor for API compatibility (already backwarded)
    loss_out = torch.tensor(avg_loss, device=device)
    metrics = {
        "loss": avg_loss,
        "ocr_loss": avg_ocr_loss,
        "n_chunks": n_chunks,
        "memory_size": memory_bank.size,
        "total_tokens": total_tokens,
    }
    return loss_out, metrics


# ─────────────────────────────────────────────
# Simple Random Text Dataset (for pilot testing)
# ─────────────────────────────────────────────
class RandomTextDataset(IterableDataset):
    """Generates random token sequences for pilot testing."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        for _ in range(self.num_samples):
            tokens = torch.randint(1, self.vocab_size, (self.seq_len,), generator=rng)
            yield tokens


# ─────────────────────────────────────────────
# Pilot Experiment: Verify Architecture Works
# ─────────────────────────────────────────────
def run_pilot(args):
    """Quick sanity check that RFTLM trains and memory works."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[PILOT] device={device}")
    print(f"[PILOT] Building RFT-LM with d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, window={args.window_size}, mem_top_m={args.mem_top_m}")

    model = RFTLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        window_size=args.window_size,
        ff_mult=args.ff_mult,
        dropout=0.0,
        memory_layer_idx=args.memory_layer_idx,
        use_memory=True,
        mem_top_m=args.mem_top_m,
        ocr_dim=args.ocr_dim,
    ).to(device)

    n_params = model.count_params()
    print(f"[PILOT] RFT-LM params: {n_params:,}")

    # Also build baseline for comparison
    baseline = BaselineTransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        max_len=args.total_seq_len,
    ).to(device)

    n_params_baseline = baseline.count_params()
    print(f"[PILOT] Baseline params: {n_params_baseline:,}")

    # Test forward pass
    print(f"\n[PILOT] === Forward pass test ===")
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

    # RFT-LM chunked forward
    test_seq = torch.randint(1, args.vocab_size, (2, args.total_seq_len), device=device)
    memory_bank = MemoryBank(max_entries=args.total_seq_len)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.perf_counter()
    opt.zero_grad()
    loss, metrics = train_step_chunked(model, test_seq, args.chunk_size, memory_bank, do_backward=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    t1 = time.perf_counter()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
    print(f"[PILOT] RFT-LM: loss={metrics['loss']:.4f} chunks={metrics['n_chunks']} "
          f"mem_size={metrics['memory_size']} time={t1 - t0:.2f}s peak_MB={peak_mb:.0f}")

    # Baseline forward (will OOM on long sequences — that's the point)
    if args.total_seq_len <= 4096:  # only test baseline on short sequences
        torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
        baseline.train()
        opt_b = torch.optim.AdamW(baseline.parameters(), lr=args.lr, weight_decay=0.01)
        t0 = time.perf_counter()
        out_b = baseline(test_seq[:, :-1])
        loss_b = F.cross_entropy(out_b["logits"].reshape(-1, args.vocab_size), test_seq[:, 1:].reshape(-1))
        loss_b.backward()
        opt_b.step()
        opt_b.zero_grad()
        t1 = time.perf_counter()
        peak_mb_b = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
        print(f"[PILOT] Baseline: loss={loss_b.item():.4f} time={t1 - t0:.2f}s peak_MB={peak_mb_b:.0f}")
    else:
        print(f"[PILOT] Baseline: SKIPPED (seq_len={args.total_seq_len} too long for full attention)")

    # Multi-step training pilot
    print(f"\n[PILOT] === Training for {args.pilot_steps} steps ===")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    losses = []
    for step in range(args.pilot_steps):
        seq = torch.randint(1, args.vocab_size, (args.batch_size, args.total_seq_len), device=device)
        memory_bank.reset()
        opt.zero_grad()

        loss, metrics = train_step_chunked(model, seq, args.chunk_size, memory_bank, do_backward=True)
        # backward already done per-chunk inside train_step_chunked

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(metrics["loss"])
        if (step + 1) % 10 == 0 or step == 0:
            avg = sum(losses[-10:]) / len(losses[-10:])
            print(f"  step {step + 1}/{args.pilot_steps}: loss={metrics['loss']:.4f} avg10={avg:.4f} "
                  f"mem_size={metrics['memory_size']}")

    print(f"\n[PILOT] Final avg loss: {sum(losses[-10:]) / len(losses[-10:]):.4f}")
    print(f"[PILOT] Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({'DECREASING ✓' if losses[-1] < losses[0] else 'NOT DECREASING ✗'})")

    # Save pilot results
    os.makedirs(args.outdir, exist_ok=True)
    results = {
        "model": "rft_lm",
        "n_params": n_params,
        "n_params_baseline": n_params_baseline,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "window_size": args.window_size,
        "chunk_size": args.chunk_size,
        "total_seq_len": args.total_seq_len,
        "mem_top_m": args.mem_top_m,
        "ocr_dim": args.ocr_dim,
        "pilot_steps": args.pilot_steps,
        "final_loss": losses[-1],
        "loss_decreasing": losses[-1] < losses[0],
        "all_losses": losses,
        "peak_mem_mb": peak_mb,
    }
    with open(os.path.join(args.outdir, "pilot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[PILOT] Saved to {args.outdir}/pilot_results.json")
    print("[PILOT] DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./runs_lm/pilot")

    # Model config (small for pilot, scale up for real training)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--window_size", type=int, default=512)
    ap.add_argument("--memory_layer_idx", type=int, default=4)
    ap.add_argument("--mem_top_m", type=int, default=64)
    ap.add_argument("--ocr_dim", type=int, default=256)

    # Training config
    ap.add_argument("--total_seq_len", type=int, default=4096)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pilot_steps", type=int, default=100)

    args = ap.parse_args()
    run_pilot(args)
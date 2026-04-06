"""
synth_batch.py — Synthetic KV-retrieval batches for mixed-objective RFT-LM training.

Purpose
-------
The LM training on C4 provides no signal to router_q/router_k about which
past memory slot is correct. This module restores the supervised retrieval
objective from RFT_ocr.py (which achieved recall@M ~1.00) and applies it
directly to the RFT-LM memory layer.

Layout per example:
  - Chunk 0 (length = chunk_size): KV-pair tokens [k1,v1,k2,v2,...] placed
    into random slots; the rest is zero-padded. Processed through RFT-LM
    and written into the memory bank (no gradient on this chunk).
  - Chunk 1 (length = chunk_size): [QUERY_ID, target_key, 0, 0, ...].
    Run through the transformer layers up to the memory layer and the
    supervised_retrieval_loss is computed on the token at position 1
    (the target_key, whose next-token prediction is the target value).

The correct memory slot index is known a priori (position of v_target
inside chunk 0), so router_ce/topm_hinge/pointer_ce/ocr_contrastive can
all be computed exactly as in RFT_ocr.py.
"""

import random
from dataclasses import dataclass
from typing import Tuple

import torch

from RFT_LM import MemoryBank


@dataclass
class SynthBatchConfig:
    chunk_size: int = 512
    key_vocab: int = 2048
    val_vocab: int = 2048
    num_facts: int = 8
    num_decoys: int = 64
    batch_size: int = 4
    base_vocab_offset: int = 0   # reserve IDs [offset+1 .. offset+key_vocab+val_vocab+1]


class SyntheticKVBatchGen:
    """Infinite iterator producing synthetic KV-retrieval batches."""

    def __init__(self, cfg: SynthBatchConfig, device: torch.device, seed: int):
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(seed)
        self.num_pair_slots = cfg.chunk_size // 2
        total_pairs = cfg.num_facts + cfg.num_decoys
        assert total_pairs <= self.num_pair_slots, (
            f"need {total_pairs} pair slots but chunk fits only {self.num_pair_slots} "
            f"(chunk_size={cfg.chunk_size})"
        )
        # Token id layout (0 reserved as PAD):
        #   keys: [KEY_BASE .. KEY_BASE+key_vocab-1]
        #   vals: [VAL_BASE .. VAL_BASE+val_vocab-1]
        #   QUERY_ID: VAL_BASE + val_vocab
        self.KEY_BASE = cfg.base_vocab_offset + 1
        self.VAL_BASE = self.KEY_BASE + cfg.key_vocab
        self.QUERY_ID = self.VAL_BASE + cfg.val_vocab

    def max_token_id(self) -> int:
        return self.QUERY_ID

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.cfg.batch_size
        cs = self.cfg.chunk_size
        pairs_chunk = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        query_chunk = torch.zeros(B, cs, dtype=torch.long, device=self.device)
        target_mem_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        # Query token is at position 1 in chunk 1 (after the QUERY_ID marker).
        query_token_idx = torch.full((B,), 1, dtype=torch.long, device=self.device)

        total_pairs = self.cfg.num_facts + self.cfg.num_decoys
        for b in range(B):
            keys = self.rng.sample(range(self.cfg.key_vocab), total_pairs)
            vals = [self.rng.randrange(self.cfg.val_vocab) for _ in range(total_pairs)]
            tgt_pair = self.rng.randrange(self.cfg.num_facts)
            pair_positions = self.rng.sample(range(self.num_pair_slots), total_pairs)
            for p, (k, v) in enumerate(zip(keys, vals)):
                slot = pair_positions[p]
                pairs_chunk[b, 2 * slot]     = self.KEY_BASE + k
                pairs_chunk[b, 2 * slot + 1] = self.VAL_BASE + v
            tgt_slot = pair_positions[tgt_pair]
            # Target is the VALUE position in chunk 0 (= its index in the memory bank).
            target_mem_idx[b] = 2 * tgt_slot + 1
            query_chunk[b, 0] = self.QUERY_ID
            query_chunk[b, 1] = self.KEY_BASE + keys[tgt_pair]

        return pairs_chunk, query_chunk, target_mem_idx, query_token_idx


def synthetic_retrieval_step(
    raw_model,
    batch,
    loss_weights: dict,
    do_backward: bool = True,
) -> Tuple[float, dict]:
    """
    One synthetic retrieval training step through the RFT-LM.

    Returns (total_loss_value, metrics_dict). If do_backward=True the
    gradients are already accumulated into model parameters.
    """
    pairs_chunk, query_chunk, target_mem_idx, query_token_idx = batch
    B, L0 = pairs_chunk.shape
    _, L1 = query_chunk.shape

    memory_bank = MemoryBank(max_entries=L0)

    # Chunk 0: populate memory bank WITH gradients so that mem_key_proj,
    # mem_val_proj, and the shared transformer layers all receive gradient
    # signal from the supervised retrieval loss. Without this, only router_q
    # gets gradient — memory keys/vals are frozen random projections and the
    # router can never converge (as observed in v1/v2 pilots).
    out0 = raw_model(
        pairs_chunk,
        memory_keys=None, memory_vals=None, memory_positions=None,
        pos_offset=0,
        return_memory_state=True,
        compute_ocr_loss=False,
        detach_memory=False,  # keep gradient graph alive
    )
    memory_bank.add(
        out0["new_mem_keys"],   # NOT detached
        out0["new_mem_vals"],   # NOT detached
        0, L0,
    )
    mem_k, mem_v, mem_pos = memory_bank.get_state()

    # Chunk 1: partial forward up to memory layer with gradients enabled
    model = raw_model
    x = model.embed(query_chunk) * model.embed_scale
    x = model.drop(x)
    rope_cos, rope_sin = model.rope(L1, offset=L0)
    for i, layer in enumerate(model.layers):
        x = layer(x, rope_cos, rope_sin, L0)
        if i == model.memory_layer_idx:
            break

    losses = model.memory_layer.supervised_retrieval_loss(
        x, mem_k, mem_v, mem_pos, L0,
        target_mem_idx=target_mem_idx,
        query_token_idx=query_token_idx,
    )

    total = (
        loss_weights.get("router_ce", 1.0)      * losses["router_ce"]
        + loss_weights.get("topm_hinge", 0.25)    * losses["topm_hinge"]
        + loss_weights.get("pointer_ce", 0.5)     * losses["pointer_ce"]
        + loss_weights.get("ocr_contrastive", 0.2) * losses["ocr_contrastive"]
    )

    if do_backward:
        total.backward()

    metrics = {
        "synth_loss": float(total.item()),
        "synth_router_ce": float(losses["router_ce"].item()),
        "synth_topm_hinge": float(losses["topm_hinge"].item()),
        "synth_pointer_ce": float(losses["pointer_ce"].item()),
        "synth_ocr_contrastive": float(losses["ocr_contrastive"].item()),
        "synth_router_top1_acc": float(losses["router_top1_acc"].item()),
        "synth_recall_at_m": float(losses["recall_at_m"].item()),
        "synth_pointer_acc": float(losses["pointer_acc"].item()),
        "synth_fused_acc": float(losses["fused_acc"].item()),
    }
    return float(total.item()), metrics

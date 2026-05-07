"""
rft_graft.py — Insert RFT memory layer into a pretrained HF causal LM.

Trainable: memory_layer, mem_key_proj, mem_val_proj.
Frozen by default: the base transformer (LoRA can be added later via peft).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from RFT_LM import RFTMemoryLayer


class RFTGraftLM(nn.Module):
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B",
        memory_layer_idx: int = 12,
        mem_top_m: int = 64,
        ocr_dim: int = 256,
        mem_gate_alpha_init: float = 1.0,
        freeze_base: bool = True,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        cfg = self.base.config
        assert getattr(cfg, "tie_word_embeddings", False), (
            "Alignment loss requires tied embeddings"
        )

        self.d_model = cfg.hidden_size
        self.n_layers = cfg.num_hidden_layers
        self.vocab_size = cfg.vocab_size
        self.memory_layer_idx = memory_layer_idx
        self.use_memory = True
        self.mem_top_m = mem_top_m
        self.embed_scale = 1.0

        # Locate base parts (works for Llama/Qwen2/Mistral families)
        self._layers = self.base.model.layers
        self._embed = self.base.model.embed_tokens
        self._final_norm = self.base.model.norm
        self._lm_head = self.base.lm_head
        self._rotary = getattr(self.base.model, "rotary_emb", None)

        # Projections used to build memory entries from layer-N hidden states
        self.mem_key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.mem_val_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # The operator
        self.memory_layer = RFTMemoryLayer(
            d_model=self.d_model,
            top_m=mem_top_m,
            ocr_dim=ocr_dim,
            ocr_alpha_init=0.1,
        )
        with torch.no_grad():
            self.memory_layer.mem_gate_alpha.fill_(mem_gate_alpha_init)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    # Aliases so niah_batch.py code finds what it expects
    @property
    def embed(self):
        return self._embed

    @property
    def lm_head(self):
        return self._lm_head

    @property
    def ln_f(self):
        return self._final_norm

    @property
    def layers(self):
        return self._layers

    def drop(self, x):
        return x

    def rope(self, L, offset=0):
        return None, None

    def _build_position_ids(self, B, L, pos_offset, device):
        return torch.arange(pos_offset, pos_offset + L, device=device).unsqueeze(0).expand(B, -1)

    def _build_position_embeddings(self, hidden, position_ids):
        if self._rotary is None:
            return None
        return self._rotary(hidden, position_ids)

    def _build_causal_mask(self, B, L, dtype, device):
        if L == 1:
            return torch.zeros((B, 1, 1, 1), dtype=dtype, device=device)
        mask = torch.full((L, L), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    def forward(
        self,
        input_ids,
        memory_keys=None,
        memory_vals=None,
        memory_positions=None,
        pos_offset: int = 0,
        return_memory_state: bool = False,
        compute_ocr_loss: bool = False,
        detach_memory: bool = False,
    ):
        B, L = input_ids.shape
        device = input_ids.device

        x = self._embed(input_ids) * self.embed_scale
        position_ids = self._build_position_ids(B, L, pos_offset, device)
        position_embeddings = self._build_position_embeddings(x, position_ids)

        new_mem_keys = None
        new_mem_vals = None

        causal_mask = self._build_causal_mask(B, x.shape[1], x.dtype, device)
        for i, layer in enumerate(self._layers):
            kwargs = dict(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            try:
                out = layer(x, **kwargs)
            except TypeError:
                out = layer(
                    x,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
            x = out[0] if isinstance(out, tuple) else out

            if i == self.memory_layer_idx:
                k = self.mem_key_proj(x)
                v = self.mem_val_proj(x)
                new_mem_keys = k.detach() if detach_memory else k
                new_mem_vals = v.detach() if detach_memory else v

                if self.use_memory and memory_keys is not None and memory_keys.shape[1] > 0:
                    mem_ctx = self.memory_layer(
                        x, memory_keys, memory_vals, memory_positions, pos_offset
                    )
                    x = x + mem_ctx

        x = self._final_norm(x)
        logits = self._lm_head(x)

        out = {"logits": logits}
        if return_memory_state:
            out["new_mem_keys"] = new_mem_keys
            out["new_mem_vals"] = new_mem_vals
        return out

    def forward_manual(self, qry_chunk, mem_k, mem_v, mem_pos, pos_offset: int):
        """Manual layer-by-layer forward used by niah_retrieval_step.

        Returns (x_at_mem, mem_ctx, logits). HF-compatible layer call.
        """
        B, L1 = qry_chunk.shape
        device = qry_chunk.device
        x = self._embed(qry_chunk) * self.embed_scale
        position_ids = self._build_position_ids(B, L1, pos_offset, device)
        position_embeddings = self._build_position_embeddings(x, position_ids)

        x_at_mem = None
        mem_ctx = None

        causal_mask = self._build_causal_mask(B, x.shape[1], x.dtype, device)
        for i, layer in enumerate(self._layers):
            kwargs = dict(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            try:
                out = layer(x, **kwargs)
            except TypeError:
                out = layer(
                    x,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
            x = out[0] if isinstance(out, tuple) else out

            if i == self.memory_layer_idx:
                x_at_mem = x
                if self.use_memory and mem_k is not None and mem_k.shape[1] > 0:
                    mem_ctx = self.memory_layer(x, mem_k, mem_v, mem_pos, pos_offset)
                    x = x + mem_ctx

        x = self._final_norm(x)
        logits = self._lm_head(x)
        return x_at_mem, mem_ctx, logits

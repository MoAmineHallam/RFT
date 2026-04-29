"""Test: does AR generation work if we feed MORE context at each step?

If we pass [last 64 tokens of prompt + accumulated generated], does memory fire better?
"""
import json, random
import torch
from transformers import AutoTokenizer
from niah_batch import NIAHBatchGen, NIAHBatchConfig
from eval_ruler_niah import _forward_chunk, load_rft_model
from RFT_LM import MemoryBank

torch.manual_seed(42); random.seed(42)
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2_tokenizer", use_fast=True)
tokenizer.model_max_length = 10**9

pg = json.load(open("PaulGrahamEssays.json"))
blob = pg["text"] if isinstance(pg, dict) else pg
words = blob.split(); chunk = 500
texts = [" ".join(words[i:i+chunk]) for i in range(0, min(len(words), 2000*chunk), chunk)]
dist_tokens = tokenizer.encode(" ".join(texts), add_special_tokens=False)

cfg = NIAHBatchConfig(chunk_size=512, batch_size=1, num_needles=3, value_type="numbers")
gen = NIAHBatchGen(cfg, tokenizer, dist_tokens, device=device, seed=42)

model = load_rft_model("mixed_pilot_v12_ruler_full_seed42/rft_lm/checkpoint_step35000.pt", device)
model.eval()

correct = 0
for t in range(10):
    ctx, qry, tgt_mem, qry_tok, tgt_id, val_targets = next(gen)
    ctx_ids = ctx[0].tolist()
    qry_ids = qry[0].tolist()
    probe_pos = int(qry_tok[0].item())
    val_tok_ids = [int(v) for v in val_targets[0].tolist() if v != -100]
    ref_val = tokenizer.decode(val_tok_ids).strip()

    # Setup: write context to memory
    mb = MemoryBank(max_entries=65536); mb.reset()
    ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long, device=device)
    _ = _forward_chunk(model, ctx_tensor, 0, True, mb, True)

    # Query setup: process query up to probe position, writing to memory
    qry_prefix = qry_ids[:probe_pos + 1]
    qry_tensor = torch.tensor([qry_prefix], dtype=torch.long, device=device)
    out = _forward_chunk(model, qry_tensor, 512, True, mb, True)
    logits = out["logits"][0, -1]

    # Greedy AR generation, but at each step feed the FULL growing query
    generated = []
    for step in range(8):
        next_id = int(logits.argmax().item())
        generated.append(next_id)
        # Feed the FULL growing sequence (qry prefix + generated) as a single chunk
        full_seq = qry_prefix + generated
        full_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
        # Reset mem bank but keep context memory
        mb2 = MemoryBank(max_entries=65536); mb2.reset()
        ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        _ = _forward_chunk(model, ctx_tensor, 0, True, mb2, True)
        out = _forward_chunk(model, full_tensor, 512, True, mb2, False)
        logits = out["logits"][0, -1]

    pred = tokenizer.decode(generated).strip()
    ok = ref_val in pred
    if ok: correct += 1
    print(f"[{t}] ref={ref_val} pred='{pred[:40]}' ok={ok}")

print(f"\nFull-context AR: {correct}/10")
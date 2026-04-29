import re

with open("eval_ruler_niah.py") as f:
    src = f.read()

OLD = '''    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    total_len = ids.shape[1]
    mb = MemoryBank(max_entries=65536) if use_memory else None
    if mb is not None:
        mb.reset()
    # Phase 1: process body in full chunks, writing to memory
    body_end = max(0, total_len - tail_chunk_len) if tail_chunk_len > 0 else total_len
    last_logits = None
    for s in range(0, body_end, chunk_size):
        e = min(body_end, s + chunk_size)
        out = _forward_chunk(model, ids[:, s:e], s, use_memory, mb, True)
        last_logits = out["logits"]
    # Phase 2: process the tail as a single mini-chunk (memory now contains the body)
    if body_end < total_len:
        out = _forward_chunk(model, ids[:, body_end:total_len], body_end, use_memory, mb, True)
        last_logits = out["logits"]
    if last_logits is None:
        return []
    logits = last_logits[:, -1, :]
    out_ids = []
    for i in range(max_new_tokens):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        tid = int(nxt.item())
        out_ids.append(tid)
        if eos_token_id is not None and tid == eos_token_id:
            break
        acc_tensor = torch.tensor([out_ids], dtype=torch.long, device=device)
        out = _forward_chunk(model, acc_tensor, total_len, use_memory, mb, False)
        logits = out["logits"][:, -1, :]
    return out_ids'''

NEW = '''    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    total_len = ids.shape[1]
    mb = MemoryBank(max_entries=65536) if use_memory else None
    if mb is not None:
        mb.reset()
    # Phase 1: process body in full chunks, writing to memory
    body_end = max(0, total_len - tail_chunk_len) if tail_chunk_len > 0 else total_len
    last_logits = None
    for s in range(0, body_end, chunk_size):
        e = min(body_end, s + chunk_size)
        out = _forward_chunk(model, ids[:, s:e], s, use_memory, mb, True)
        last_logits = out["logits"]
    # Phase 2: process the tail (memory now contains the body)
    # We also remember the tail tokens so AR generation can re-feed the query context.
    tail_start = body_end
    tail_tokens: List[int] = []
    if body_end < total_len:
        tail_tokens = input_ids[body_end:total_len]
        out = _forward_chunk(model, ids[:, body_end:total_len], body_end, use_memory, mb, True)
        last_logits = out["logits"]
    if last_logits is None:
        return []
    logits = last_logits[:, -1, :]
    out_ids: List[int] = []
    # Figure out a suffix window of the prompt to keep reattaching at each AR step,
    # so the memory router has local context matching the training distribution.
    # We pick min(chunk_size-1, len(input_ids)) so the combined input stays within chunk_size.
    window_len = min(chunk_size - max_new_tokens - 1, total_len)
    window_len = max(window_len, 0)
    prompt_suffix = input_ids[total_len - window_len:] if window_len > 0 else []
    suffix_offset = total_len - window_len  # absolute position of prompt_suffix[0]
    for i in range(max_new_tokens):
        nxt_id = int(torch.argmax(logits, dim=-1).item())
        out_ids.append(nxt_id)
        if eos_token_id is not None and nxt_id == eos_token_id:
            break
        # Feed the running window: prompt_suffix + generated_so_far
        # Memory bank already has the full body; we just need proper local context.
        # IMPORTANT: write=False so we don't pollute memory with our own outputs.
        seq = prompt_suffix + out_ids
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        # Reset the body-memory read each step by using a temporary memory bank view.
        # Actually, we keep mb as-is since body memory should stay; we just skip the write.
        out = _forward_chunk(model, seq_tensor, suffix_offset, use_memory, mb, False)
        logits = out["logits"][:, -1, :]
    return out_ids'''

if OLD not in src:
    print("❌ Could not find the old generate_greedy body. Applying manually needed.")
    raise SystemExit(1)

src = src.replace(OLD, NEW)
with open("eval_ruler_niah.py", "w") as f:
    f.write(src)
print("✅ generate_greedy patched — AR path now feeds windowed prompt+generated at correct pos_offset")
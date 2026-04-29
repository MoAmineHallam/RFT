"""Patch niah_sanity_ruler.py to use 2-chunk layout matching training."""

with open("niah_sanity_ruler.py") as f:
    src = f.read()

# Find the function boundaries
start = src.find("def build_ruler_sample(")
assert start >= 0, "build_ruler_sample not found"
end = src.find("def run_sanity(", start)
assert end >= 0, "run_sanity not found"

old_func = src[start:end]

new_func = '''def build_ruler_sample(
    tokenizer,
    wordlist,
    distractor_tokens,
    seq_len: int,
    seed: int,
    value_type: str,
    needle_depth,
    num_needle_k: int,
    num_needle_q: int,
    chunk_size: int = 512,
    add_instruction: bool = False,
):
    """
    Two-chunk layout matching niah_batch.py training exactly:
      chunk 1 (ctx, size=chunk_size): distractor with needles interleaved
      chunk 2 (qry, size=chunk_size): padding-distractor + query + answer prefix

    Total returned input length = 2 * chunk_size (plus optional trailing trim).
    add_instruction defaults to False since most training samples had no prefix.
    """
    rng = random.Random(seed)
    num_needle_k = max(num_needle_k, num_needle_q)

    keys = [f"{rng.choice(wordlist)}-{rng.choice(wordlist)}" for _ in range(num_needle_k)]
    values = [generate_value(rng, value_type) for _ in range(num_needle_k)]

    _pl = {"numbers": "numbers", "short_int": "numbers", "digit2": "numbers",
           "words": "words", "uuids": "uuids"}.get(value_type, value_type)
    _sg = {"numbers": "number", "short_int": "number", "digit2": "number",
           "words": "word", "uuids": "uuid"}.get(value_type, value_type)

    needle_sentences = [
        f" The special magic {_sg} for {k} is: {v}."
        for k, v in zip(keys, values)
    ]
    needle_tok_lists = [tokenizer.encode(s, add_special_tokens=False) for s in needle_sentences]

    q_idx = rng.sample(range(num_needle_k), k=num_needle_q)
    query_keys = [keys[i] for i in q_idx]
    ref_values = [values[i] for i in q_idx]

    if len(query_keys) == 1:
        query_str = query_keys[0]
    elif len(query_keys) == 2:
        query_str = f"{query_keys[0]} and {query_keys[1]}"
    else:
        query_str = ", ".join(query_keys[:-1]) + f", and {query_keys[-1]}"

    # ---- Chunk 1: ctx (distractor + needles interleaved) ----
    if add_instruction:
        instr_str = (
            f"Some special magic {_pl} are hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about the {_pl} afterwards.\\n"
        )
        instr_toks = tokenizer.encode(instr_str, add_special_tokens=False)
    else:
        instr_toks = []

    total_needle_len = sum(len(t) for t in needle_tok_lists)
    ctx_dist_budget = max(16, chunk_size - len(instr_toks) - total_needle_len)

    max_start = max(0, len(distractor_tokens) - ctx_dist_budget - 1)
    d_start = rng.randint(0, max_start) if max_start > 0 else 0
    ctx_dist = distractor_tokens[d_start:d_start + ctx_dist_budget]
    while len(ctx_dist) < ctx_dist_budget:
        extra_start = rng.randint(0, max(0, len(distractor_tokens) - 128))
        ctx_dist = ctx_dist + distractor_tokens[extra_start:extra_start + 128]
    ctx_dist = ctx_dist[:ctx_dist_budget]

    n_needles = len(needle_tok_lists)
    seg_len = len(ctx_dist) // (n_needles + 1)
    ctx_toks = list(instr_toks)
    for i, nt in enumerate(needle_tok_lists):
        seg_start = i * seg_len
        ctx_toks.extend(ctx_dist[seg_start:seg_start + seg_len])
        ctx_toks.extend(nt)
    ctx_toks.extend(ctx_dist[n_needles * seg_len:])

    # Truncate or pad ctx to chunk_size
    ctx_toks = ctx_toks[:chunk_size]
    if len(ctx_toks) < chunk_size:
        pad_start = rng.randint(0, max(0, len(distractor_tokens) - 128))
        ctx_toks.extend(distractor_tokens[pad_start:pad_start + (chunk_size - len(ctx_toks))])
        ctx_toks = ctx_toks[:chunk_size]

    # ---- Chunk 2: qry (padding-distractor + query + answer prefix) ----
    query_full_str = (
        f"\\nWhat are all the special magic {_pl} for {query_str} "
        f"mentioned in the provided text? "
        f"The special magic {_pl} for {query_str} mentioned in "
        f"the provided text are"
    )
    query_toks = tokenizer.encode(query_full_str, add_special_tokens=False)

    qry_dist_budget = max(16, chunk_size - len(query_toks))
    q_start = rng.randint(0, max(0, len(distractor_tokens) - qry_dist_budget - 1))
    qry_dist = distractor_tokens[q_start:q_start + qry_dist_budget]
    while len(qry_dist) < qry_dist_budget:
        extra_start = rng.randint(0, max(0, len(distractor_tokens) - 128))
        qry_dist = qry_dist + distractor_tokens[extra_start:extra_start + 128]
    qry_dist = qry_dist[:qry_dist_budget]

    qry_toks = list(qry_dist) + list(query_toks)
    qry_toks = qry_toks[:chunk_size]
    if len(qry_toks) < chunk_size:
        # Pad at the front with extra distractor
        pad_needed = chunk_size - len(qry_toks)
        pad_start = rng.randint(0, max(0, len(distractor_tokens) - pad_needed))
        qry_toks = list(distractor_tokens[pad_start:pad_start + pad_needed]) + qry_toks
        qry_toks = qry_toks[:chunk_size]

    input_ids = ctx_toks + qry_toks
    return {
        "input_ids": input_ids,
        "refs": ref_values,
        "query_keys": query_keys,
    }


'''

src_new = src[:start] + new_func + src[end:]
with open("niah_sanity_ruler.py", "w") as f:
    f.write(src_new)
print("OK: build_ruler_sample replaced with 2-chunk layout")
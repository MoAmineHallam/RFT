"""
niah_sanity_traindist.py — Sanity test that uses NIAHBatchGen directly,
so the eval sample is constructed by the EXACT same code path as training.

If v12 fails even on this, generalization is broken.
If v12 passes here but fails on niah_sanity_ruler.py, the eval has a bug.
"""
import argparse
import json
import random
import torch
from transformers import AutoTokenizer

from niah_batch import NIAHBatchGen, NIAHBatchConfig
from eval_ruler_niah import (
    generate_greedy,
    load_rft_model,
    string_match_all_binary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default="gpt2_tokenizer")
    ap.add_argument("--distractor_path", type=str, default="PaulGrahamEssays.json")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--num_needles", type=int, default=3)
    ap.add_argument("--value_type", type=str, default="numbers")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9

    # Load distractor same way train_overnight.py does for PG essays
    pg = json.load(open(args.distractor_path))
    blob = pg["text"] if isinstance(pg, dict) else pg
    words = blob.split()
    chunk = 500
    texts = [" ".join(words[i:i+chunk]) for i in range(0, min(len(words), 2000*chunk), chunk)]
    dist_tokens = tokenizer.encode(" ".join(texts), add_special_tokens=False)
    print(f"[TRAINDIST] distractor tokens: {len(dist_tokens):,}")

    # Build NIAHBatchGen exactly as train_overnight.py does
    cfg = NIAHBatchConfig(
        chunk_size=args.chunk_size,
        batch_size=1,
        num_needles=args.num_needles,
        value_type=args.value_type,
    )
    gen = NIAHBatchGen(cfg, tokenizer, dist_tokens, device=device, seed=args.seed)

    # Load model
    model = load_rft_model(args.rft_ckpt, device)
    model.eval()

    hits_full = 0
    hits_first = 0
    print(f"\n=== RFT-LM on TRAINING-DISTRIBUTION samples ===")
    for t in range(args.n_trials):
        # Get a training-format batch (batch size 1)
        ctx, qry, tgt_mem, qry_tok, tgt_id, val_targets = next(gen)

        # Construct input_ids as the model would see during generation:
        # [context_chunk] + [query_chunk_up_to_answer_prefix]
        # query chunk has format: [distractor_pad][query_text][answer_prefix][value_tokens]
        # We want to stop right after answer_prefix ("are"), before value tokens.
        ctx_ids = ctx[0].tolist()
        qry_ids = qry[0].tolist()
        # The probe position qry_tok[0] is the last token of answer prefix ("are")
        # Input should be ctx + qry[:qry_tok+1], then model generates the value
        probe_pos = int(qry_tok[0].item())
        input_ids = ctx_ids + qry_ids[:probe_pos + 1]

        # Reference value: decode from val_targets, stripping -100 pads
        val_tok_ids = [int(v) for v in val_targets[0].tolist() if v != -100]
        ref_val = tokenizer.decode(val_tok_ids).strip()

        gen_ids = generate_greedy(
            model=model,
            input_ids=input_ids,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            use_memory=True,
            tail_chunk_len=8,
        )
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        full_ok = ref_val in pred
        ref_first = tokenizer.decode([val_tok_ids[0]]).strip()
        first_ok = pred.lstrip().startswith(ref_first)
        hits_full += int(full_ok)
        hits_first += int(first_ok)

        mark_full = "✅" if full_ok else "❌"
        mark_first = "✅" if first_ok else "❌"
        tail = tokenizer.decode(input_ids[-60:], skip_special_tokens=True)
        print(f"  [{t}] full={mark_full} first={mark_first} ref={ref_val} (first='{ref_first}')")
        print(f"      tail=...{tail!r}")
        print(f"      pred={pred!r}")

    print(f"\n  RFT-LM train-dist full: {hits_full}/{args.n_trials}")
    print(f"  RFT-LM train-dist first: {hits_first}/{args.n_trials}")


if __name__ == "__main__":
    main()
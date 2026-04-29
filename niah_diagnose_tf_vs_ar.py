"""
Compare teacher-forced prediction vs autoregressive generation on the same sample.

If teacher-forced predicts the value tokens correctly but autoregressive fails,
the training loss is measuring something that doesn't transfer to inference.
"""
import argparse, json, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from niah_batch import NIAHBatchGen, NIAHBatchConfig
from eval_ruler_niah import generate_greedy, load_rft_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_ckpt", required=True)
    ap.add_argument("--tokenizer_path", default="gpt2_tokenizer")
    ap.add_argument("--distractor_path", default="PaulGrahamEssays.json")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9

    pg = json.load(open(args.distractor_path))
    blob = pg["text"] if isinstance(pg, dict) else pg
    words = blob.split()
    chunk = 500
    texts = [" ".join(words[i:i+chunk]) for i in range(0, min(len(words), 2000*chunk), chunk)]
    dist_tokens = tokenizer.encode(" ".join(texts), add_special_tokens=False)

    cfg = NIAHBatchConfig(chunk_size=512, batch_size=1, num_needles=3, value_type="numbers")
    gen = NIAHBatchGen(cfg, tokenizer, dist_tokens, device=device, seed=args.seed)

    model = load_rft_model(args.rft_ckpt, device)
    model.eval()

    tf_correct = 0
    ar_correct_first = 0
    ar_correct_full = 0

    for t in range(args.n_trials):
        ctx, qry, tgt_mem, qry_tok, tgt_id, val_targets = next(gen)
        ctx_ids = ctx[0].tolist()
        qry_ids = qry[0].tolist()
        probe_pos = int(qry_tok[0].item())
        val_tok_ids = [int(v) for v in val_targets[0].tolist() if v != -100]
        ref_val = tokenizer.decode(val_tok_ids).strip()
        ref_first = tokenizer.decode([val_tok_ids[0]]).strip()

        # === AR path: generate from probe position ===
        ar_input = ctx_ids + qry_ids[:probe_pos + 1]
        ar_gen = generate_greedy(
            model=model, input_ids=ar_input, chunk_size=512,
            max_new_tokens=20, eos_token_id=tokenizer.eos_token_id,
            device=device, use_memory=True, tail_chunk_len=8,
        )
        ar_pred = tokenizer.decode(ar_gen, skip_special_tokens=True).strip()
        ar_first = ar_pred.lstrip().startswith(ref_first)
        ar_full = ref_val in ar_pred

        # === TF path: give the WHOLE input including value tokens,
        # then check what the model predicts AT each value token position ===
        # This replicates what training computes (at every j in val_targets,
        # get the logits at probe_pos+j given inputs up to probe_pos+j).
        tf_input_ids = ctx_ids + qry_ids  # full context + query with value baked in
        tf_input = torch.tensor([tf_input_ids], dtype=torch.long, device=device)
        # Run in 2 chunks of 512, mimicking training
        from eval_ruler_niah import _forward_chunk
        from RFT_LM import MemoryBank
        mb = MemoryBank(max_entries=65536)
        mb.reset()
        _ = _forward_chunk(model, tf_input[:, :512], 0, True, mb, True)
        out = _forward_chunk(model, tf_input[:, 512:1024], 512, True, mb, True)
        logits = out["logits"][0]  # [L, V]
        # Predicted token at position 512+probe_pos predicts the token at 512+probe_pos+1,
        # which is the first value token. That's val_tok_ids[0].
        tf_preds = []
        for j in range(len(val_tok_ids)):
            pred_pos = probe_pos + j  # within the second chunk
            if pred_pos >= logits.shape[0]:
                break
            pred_id = int(logits[pred_pos].argmax().item())
            tf_preds.append(pred_id)
        tf_match_first = (len(tf_preds) > 0 and tf_preds[0] == val_tok_ids[0])
        tf_match_full = (tf_preds == val_tok_ids[:len(tf_preds)])

        if tf_match_first: tf_correct += 1
        if ar_first: ar_correct_first += 1
        if ar_full: ar_correct_full += 1

        tf_pred_str = tokenizer.decode(tf_preds)
        print(f"[{t}] ref={ref_val}")
        print(f"    TF first-tok: {tf_match_first}  TF full: {tf_match_full}  TF_pred='{tf_pred_str}'")
        print(f"    AR first-tok: {ar_first}  AR full: {ar_full}  AR_pred='{ar_pred[:40]}'")

    print(f"\n  TF first-tok correct: {tf_correct}/{args.n_trials}")
    print(f"  AR first-tok correct: {ar_correct_first}/{args.n_trials}")
    print(f"  AR full-match correct: {ar_correct_full}/{args.n_trials}")
    if tf_correct >= args.n_trials * 0.7 and ar_correct_first < args.n_trials * 0.3:
        print("\n  VERDICT: Training loss is teacher-forced and does NOT transfer to AR inference.")
        print("           Model learned to predict value GIVEN context up to that point is intact,")
        print("           but the memory/retrieval mechanism doesn't fire during AR generation.")

if __name__ == "__main__":
    main()
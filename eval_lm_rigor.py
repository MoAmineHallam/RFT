import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from RFT_LM import BaselineTransformerLM, MemoryBank, RFTLM


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ci95_mean(vals: List[float]) -> Tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(vals) / n
    if n == 1:
        return mean, mean, mean
    sd = statistics.stdev(vals)
    m = 1.96 * sd / math.sqrt(n)
    return mean, mean - m, mean + m


def load_val_tokens(path: str, tokenizer_path: str, max_docs: int) -> Tuple[torch.Tensor, object]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.model_max_length = 10**9

    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            obj = json.loads(line)
            txt = obj.get("text", "")
            if txt:
                texts.append(txt)

    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer.encode(t, add_special_tokens=False))

    return torch.tensor(all_tokens, dtype=torch.long), tokenizer


def load_rft(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = RFTLM(
        vocab_size=a["vocab_size"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        window_size=a["window_size"],
        ff_mult=a["ff_mult"],
        dropout=0.0,
        memory_layer_idx=a["memory_layer_idx"],
        use_memory=True,
        mem_top_m=a["mem_top_m"],
        ocr_dim=a["ocr_dim"],
    ).to(device)
    sd = ck["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_baseline(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = BaselineTransformerLM(
        vocab_size=a["vocab_size"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        ff_mult=a["ff_mult"],
        dropout=0.0,
        max_len=65536,
    ).to(device)
    sd = ck["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def apply_rft_ablation(model: RFTLM, mode: str):
    if mode == "none":
        return
    if mode == "disable_memory":
        model.use_memory = False
        return
    if mode == "disable_ocr":
        if hasattr(model, "memory_layer") and hasattr(model.memory_layer, "ocr_alpha"):
            with torch.no_grad():
                model.memory_layer.ocr_alpha.fill_(0.0)
        return
    raise ValueError(f"unknown ablation: {mode}")


@torch.no_grad()
def eval_sequences(model, tokens: torch.Tensor, starts: List[int], seq_len: int, chunk_size: int, device: torch.device, use_memory: bool):
    total_loss = 0.0
    total_tok = 0
    per_seq_losses = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()

    for s0 in starts:
        seq = tokens[s0:s0 + seq_len + 1]
        if len(seq) < seq_len + 1:
            seq = F.pad(seq, (0, seq_len + 1 - len(seq)))
        ids = seq.unsqueeze(0).to(device)

        mem = MemoryBank(max_entries=32768) if use_memory else None
        if mem is not None:
            mem.reset()

        seq_loss = 0.0
        seq_tok = 0
        for st in range(0, seq_len, chunk_size):
            ed = min(st + chunk_size, seq_len)
            x = ids[:, st:ed]
            y = ids[:, st + 1:ed + 1]
            if use_memory:
                mk, mv, mp = mem.get_state()
                out = model(
                    x,
                    memory_keys=mk,
                    memory_vals=mv,
                    memory_positions=mp,
                    pos_offset=st,
                    return_memory_state=True,
                )
                if "new_mem_keys" in out:
                    mem.add(out["new_mem_keys"].detach(), out["new_mem_vals"].detach(), st, x.shape[1])
            else:
                out = model(x, pos_offset=st)

            logits = out["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            seq_loss += float(loss.item())
            seq_tok += x.shape[1]

        per_seq_losses.append(seq_loss / max(seq_tok, 1))
        total_loss += seq_loss
        total_tok += seq_tok

    wall = time.time() - t0
    avg_loss = total_loss / max(total_tok, 1)
    ppl = math.exp(avg_loss)
    tps = total_tok / wall if wall > 0 else float("inf")

    peak_alloc = 0.0
    peak_reserved = 0.0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "tokens": total_tok,
        "wall_s": wall,
        "tokens_per_s": tps,
        "peak_vram_alloc_gb": peak_alloc,
        "peak_vram_reserved_gb": peak_reserved,
        "per_sequence_loss": per_seq_losses,
    }


def make_starts(total_tokens: int, seq_len: int, n: int, seed: int) -> List[int]:
    n_possible = max(0, (total_tokens - 1) // seq_len)
    n_eval = min(n_possible, n)
    rng = random.Random(seed)
    picks = list(range(n_possible))
    rng.shuffle(picks)
    picks = sorted(picks[:n_eval])
    return [p * seq_len for p in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--rft_ckpt", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_docs", type=int, default=2000)
    ap.add_argument("--max_seqs", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "val_path": args.val_path,
        "tokenizer_path": args.tokenizer_path,
        "rft_ckpt": args.rft_ckpt,
        "baseline_ckpt": args.baseline_ckpt,
        "seq_lens": args.seq_lens,
        "seeds": args.seeds,
        "chunk_size": args.chunk_size,
        "max_docs": args.max_docs,
        "max_seqs": args.max_seqs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokens, tokenizer = load_val_tokens(args.val_path, args.tokenizer_path, args.max_docs)

    variants = [
        ("baseline", "baseline"),
        ("rft_none", "none"),
        ("rft_disable_memory", "disable_memory"),
        ("rft_disable_ocr", "disable_ocr"),
    ]

    results = {"config": config, "config_hash": config_hash, "runs": {}, "aggregates": {}}
    failures = []
    error_lines = []

    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_key = str(seed)
        results["runs"][seed_key] = {}

        starts_by_len = {sl: make_starts(len(tokens), sl, args.max_seqs, seed + sl * 9973) for sl in args.seq_lens}

        for variant_name, ablation in variants:
            results["runs"][seed_key][variant_name] = {}

            try:
                if variant_name == "baseline":
                    model = load_baseline(args.baseline_ckpt, device)
                    use_memory = False
                else:
                    model = load_rft(args.rft_ckpt, device)
                    apply_rft_ablation(model, ablation)
                    use_memory = ablation != "disable_memory"

                for sl in args.seq_lens:
                    starts = starts_by_len[sl]
                    if not starts:
                        error_lines.append(f"seed={seed} variant={variant_name} seq_len={sl} no sequences available")
                        continue
                    out = eval_sequences(
                        model=model,
                        tokens=tokens,
                        starts=starts,
                        seq_len=sl,
                        chunk_size=args.chunk_size,
                        device=device,
                        use_memory=use_memory,
                    )
                    out["n_sequences"] = len(starts)
                    out["start_positions"] = starts
                    results["runs"][seed_key][variant_name][str(sl)] = out

            except torch.cuda.OutOfMemoryError as e:
                msg = f"OOM seed={seed} variant={variant_name}: {str(e)}"
                error_lines.append(msg)
                torch.cuda.empty_cache()
            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()

        # failure collection for full RFT vs baseline
        for sl in args.seq_lens:
            s = str(sl)
            b = results["runs"][seed_key].get("baseline", {}).get(s)
            r = results["runs"][seed_key].get("rft_none", {}).get(s)
            if not b or not r:
                continue
            b_losses = b["per_sequence_loss"]
            r_losses = r["per_sequence_loss"]
            starts = b["start_positions"]
            for i, (bl, rl, st) in enumerate(zip(b_losses, r_losses, starts)):
                d = rl - bl
                if d > 0:
                    tok_slice = tokens[st:st + min(sl, 128)].tolist()
                    snippet = tokenizer.decode(tok_slice, skip_special_tokens=True)
                    failures.append({
                        "seed": seed,
                        "seq_len": sl,
                        "sample_idx": i,
                        "start_token": st,
                        "baseline_loss": bl,
                        "rft_loss": rl,
                        "delta_loss_rft_minus_baseline": d,
                        "text_snippet": snippet[:240],
                    })

    # aggregate
    for sl in args.seq_lens:
        s = str(sl)
        results["aggregates"][s] = {}
        baseline_seed_losses = []
        baseline_seed_ppl = []

        for seed in args.seeds:
            rec = results["runs"].get(str(seed), {}).get("baseline", {}).get(s)
            if rec:
                baseline_seed_losses.append(rec["loss"])
                baseline_seed_ppl.append(rec["perplexity"])

        for variant_name, _ in variants:
            seed_rows = []
            for seed in args.seeds:
                rec = results["runs"].get(str(seed), {}).get(variant_name, {}).get(s)
                if rec:
                    seed_rows.append(rec)
            if not seed_rows:
                continue

            loss_vals = [r["loss"] for r in seed_rows]
            ppl_vals = [r["perplexity"] for r in seed_rows]
            tps_vals = [r["tokens_per_s"] for r in seed_rows]
            wall_vals = [r["wall_s"] for r in seed_rows]
            vram_vals = [r["peak_vram_alloc_gb"] for r in seed_rows]

            loss_mean, loss_lo, loss_hi = ci95_mean(loss_vals)
            ppl_mean, ppl_lo, ppl_hi = ci95_mean(ppl_vals)
            tps_mean, tps_lo, tps_hi = ci95_mean(tps_vals)
            wall_mean, wall_lo, wall_hi = ci95_mean(wall_vals)
            vram_mean, vram_lo, vram_hi = ci95_mean(vram_vals)

            delta_loss_vals = []
            delta_ppl_vals = []
            if variant_name != "baseline":
                for seed in args.seeds:
                    vr = results["runs"].get(str(seed), {}).get(variant_name, {}).get(s)
                    br = results["runs"].get(str(seed), {}).get("baseline", {}).get(s)
                    if vr and br:
                        delta_loss_vals.append(vr["loss"] - br["loss"])
                        delta_ppl_vals.append(vr["perplexity"] - br["perplexity"])

            dloss_mean = dloss_lo = dloss_hi = None
            dppl_mean = dppl_lo = dppl_hi = None
            sig = None
            if delta_loss_vals:
                dloss_mean, dloss_lo, dloss_hi = ci95_mean(delta_loss_vals)
                dppl_mean, dppl_lo, dppl_hi = ci95_mean(delta_ppl_vals)
                sig = (dloss_hi < 0.0) or (dloss_lo > 0.0)

            results["aggregates"][s][variant_name] = {
                "n_seeds": len(loss_vals),
                "loss_mean": loss_mean,
                "loss_ci95": [loss_lo, loss_hi],
                "ppl_mean": ppl_mean,
                "ppl_ci95": [ppl_lo, ppl_hi],
                "tokens_per_s_mean": tps_mean,
                "tokens_per_s_ci95": [tps_lo, tps_hi],
                "wall_s_mean": wall_mean,
                "wall_s_ci95": [wall_lo, wall_hi],
                "peak_vram_alloc_gb_mean": vram_mean,
                "peak_vram_alloc_gb_ci95": [vram_lo, vram_hi],
                "delta_vs_baseline_loss_mean": dloss_mean,
                "delta_vs_baseline_loss_ci95": [dloss_lo, dloss_hi] if dloss_mean is not None else None,
                "delta_vs_baseline_ppl_mean": dppl_mean,
                "delta_vs_baseline_ppl_ci95": [dppl_lo, dppl_hi] if dppl_mean is not None else None,
                "delta_significant_ci_excludes_zero": sig,
            }

    failures_sorted = sorted(failures, key=lambda x: x["delta_loss_rft_minus_baseline"], reverse=True)
    top_failures = failures_sorted[:20]
    results["failure_analysis"] = {
        "count_total_rft_underperform_cases": len(failures_sorted),
        "top20": top_failures,
    }

    outdir = Path(args.outdir)
    summary_path = outdir / "results_summary.json"
    table_path = outdir / "results_table.csv"
    error_path = outdir / "error_log_summary.txt"
    verdict_path = outdir / "verdict.md"
    runbook_path = outdir / "runbook.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seq_len", "variant", "n_seeds",
            "loss_mean", "loss_ci95_low", "loss_ci95_high",
            "ppl_mean", "ppl_ci95_low", "ppl_ci95_high",
            "delta_loss_vs_baseline_mean", "delta_loss_ci95_low", "delta_loss_ci95_high", "delta_significant",
            "tokens_per_s_mean", "wall_s_mean", "peak_vram_alloc_gb_mean",
        ])
        for s, var_map in results["aggregates"].items():
            for vname, rec in var_map.items():
                dci = rec.get("delta_vs_baseline_loss_ci95")
                w.writerow([
                    s, vname, rec["n_seeds"],
                    rec["loss_mean"], rec["loss_ci95"][0], rec["loss_ci95"][1],
                    rec["ppl_mean"], rec["ppl_ci95"][0], rec["ppl_ci95"][1],
                    rec.get("delta_vs_baseline_loss_mean"),
                    dci[0] if dci else None,
                    dci[1] if dci else None,
                    rec.get("delta_significant_ci_excludes_zero"),
                    rec["tokens_per_s_mean"], rec["wall_s_mean"], rec["peak_vram_alloc_gb_mean"],
                ])

    if error_lines:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("\n".join(error_lines) + "\n")
    else:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("none\n")

    # verdict
    wins = 0
    total = 0
    for s, vm in results["aggregates"].items():
        rec = vm.get("rft_none")
        if not rec:
            continue
        total += 1
        d = rec.get("delta_vs_baseline_loss_mean")
        if d is not None and d < 0:
            wins += 1

    if total == 0:
        verdict = "Insufficient completed runs to decide."
        conf = "low"
    elif wins == total:
        verdict = "RFT-LM appears better than baseline on LM loss across tested context lengths."
        conf = "medium" if len(args.seeds) == 3 else "low"
    elif wins >= total // 2 + 1:
        verdict = "RFT-LM shows mixed but mostly positive LM gains vs baseline."
        conf = "medium" if len(args.seeds) == 3 else "low"
    else:
        verdict = "RFT-LM does not show robust LM advantage over baseline under this protocol."
        conf = "medium" if len(args.seeds) == 3 else "low"

    caveats = [
        "CI is seed-level with n=3; statistical power is limited.",
        "No retraining performed; conclusions apply to current checkpoints.",
        "Deterministic token slicing used for fairness; external benchmarks not included in this verdict.",
    ]

    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write("# Is RFT-LM worth it?\n\n")
        f.write(f"**Verdict:** {verdict}\n\n")
        f.write(f"**Confidence:** {conf}\n\n")
        f.write("## Caveats\n")
        for c in caveats:
            f.write(f"- {c}\n")
        f.write("\n## Next best experiment\n")
        f.write("- Run matched retrains (3 seeds) for baseline and RFT ablations with equal token budget to isolate architecture from checkpoint luck.\n")

    script_hash = sha256_file(__file__)
    rft_hash = sha256_file(args.rft_ckpt)
    baseline_hash = sha256_file(args.baseline_ckpt)
    tok_hash = sha256_file(os.path.join(args.tokenizer_path, "tokenizer.json")) if os.path.exists(os.path.join(args.tokenizer_path, "tokenizer.json")) else "missing"

    with open(runbook_path, "a", encoding="utf-8") as f:
        f.write("\n[LM-RIGOR RUN]\n")
        f.write(f"command: {' '.join(os.sys.argv)}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
        f.write(f"config_hash={config_hash}\n")
        f.write(f"script_sha256={script_hash}\n")
        f.write(f"rft_ckpt_sha256={rft_hash}\n")
        f.write(f"baseline_ckpt_sha256={baseline_hash}\n")
        f.write(f"tokenizer_json_sha256={tok_hash}\n")
        f.write(f"results_summary={summary_path}\n")
        f.write(f"results_table={table_path}\n")
        f.write(f"error_log_summary={error_path}\n")
        f.write(f"verdict={verdict_path}\n")

    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {table_path}")
    print(f"[SAVED] {runbook_path}")
    print(f"[SAVED] {error_path}")
    print(f"[SAVED] {verdict_path}")


if __name__ == "__main__":
    main()

"""
aggregate_results.py — Parse RULER S-NIAH JSON results into paper tables.

Reads eval JSON files from multi-seed runs and produces:
  1. Mean ± std accuracy across seeds for each (variant, seq_len, depth)
  2. LaTeX-formatted table for the paper
  3. CSV for further analysis / plotting

Usage:
    python aggregate_results.py --results_dir /path/to/niah_eval/
    python aggregate_results.py --results_dir /path/to/eval_v4b_niah/  # single-seed
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import math


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))
    return max(0.0, center - margin), min(1.0, center + margin)


def load_results(results_dir):
    """Load all niah_*.json files from results_dir."""
    results = {}
    for f in Path(results_dir).glob("niah_*_full.json"):
        results[f.stem] = json.loads(f.read_text())
    # Also try per-seed format
    for f in Path(results_dir).glob("niah_*.json"):
        if f.stem not in results:
            results[f.stem] = json.loads(f.read_text())
    return results


def extract_accuracies(results_dict):
    """Extract (variant, seed, seq_len, depth) -> accuracy mappings."""
    data = defaultdict(list)  # (variant, seq_len, depth) -> [acc1, acc2, ...]

    for name, result in results_dict.items():
        # Try to extract seed and variant from filename
        # Format: niah_SEED_VARIANT or niah_v4b_full
        parts = name.replace("niah_", "").split("_")

        res = result.get("results", {})
        for sl_str, depth_map in res.items():
            for depth_str, rec in depth_map.items():
                for model_key in ["rft_lm", "baseline"]:
                    if model_key in rec:
                        acc = rec[model_key].get("accuracy", 0.0)
                        key = (model_key, sl_str, depth_str)
                        data[key].append(acc)

    return data


def print_table(data, title="NIAH Accuracy"):
    """Print a formatted table of results."""
    # Collect all seq_lens and depths
    seq_lens = sorted(set(k[1] for k in data.keys()))
    depths = sorted(set(k[2] for k in data.keys()))
    variants = sorted(set(k[0] for k in data.keys()))

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    for variant in variants:
        print(f"\n  --- {variant} ---")
        header = f"  {'seq_len':<10}"
        for d in depths:
            header += f"  d={d:>6}"
        print(header)
        print("  " + "-" * (10 + 10 * len(depths)))

        for sl in seq_lens:
            row = f"  {sl:<10}"
            for d in depths:
                accs = data.get((variant, sl, d), [])
                if not accs:
                    row += f"  {'---':>6}"
                elif len(accs) == 1:
                    row += f"  {100*accs[0]:5.1f}%"
                else:
                    mean = sum(accs) / len(accs)
                    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
                    row += f" {100*mean:4.1f}±{100*std:3.1f}"
            print(row)


def write_csv(data, outpath):
    """Write results to CSV."""
    rows = []
    for (variant, sl, depth), accs in sorted(data.items()):
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / max(len(accs), 1)) ** 0.5
        rows.append({
            "variant": variant,
            "seq_len": sl,
            "depth": depth,
            "mean_acc": f"{mean:.4f}",
            "std_acc": f"{std:.4f}",
            "n_seeds": len(accs),
            "raw_accs": ";".join(f"{a:.4f}" for a in accs),
        })

    with open(outpath, "w") as f:
        header = list(rows[0].keys()) if rows else []
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    print(f"\n[SAVED] {outpath}")


def write_latex(data, outpath):
    """Write a LaTeX table for the paper."""
    seq_lens = sorted(set(k[1] for k in data.keys()))
    depths = sorted(set(k[2] for k in data.keys()))
    variants = sorted(set(k[0] for k in data.keys()))

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{RULER S-NIAH accuracy (\\%) across sequence lengths and needle depths.}")
    lines.append("\\label{tab:niah_results}")

    ncols = 2 + len(depths)
    lines.append("\\begin{tabular}{ll" + "c" * len(depths) + "}")
    lines.append("\\toprule")

    header = "Model & Seq Len"
    for d in depths:
        header += f" & $d$={d}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for variant in variants:
        label = {
            "rft_lm": "RFT-LM (ours)",
            "baseline": "Baseline",
        }.get(variant, variant)

        for i, sl in enumerate(seq_lens):
            vname = label if i == 0 else ""
            row = f"{vname} & {sl}"
            for d in depths:
                accs = data.get((variant, sl, d), [])
                if not accs:
                    row += " & ---"
                elif len(accs) == 1:
                    row += f" & {100*accs[0]:.1f}"
                else:
                    mean = sum(accs) / len(accs)
                    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
                    row += f" & {100*mean:.1f}$\\pm${100*std:.1f}"
            row += " \\\\"
            lines.append(row)

        if variant != variants[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SAVED] {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True,
                    help="Directory containing niah_*.json result files")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory for CSV/LaTeX (default: same as results_dir)")
    args = ap.parse_args()

    outdir = args.outdir or args.results_dir

    results = load_results(args.results_dir)
    if not results:
        print(f"No niah_*.json files found in {args.results_dir}")
        return

    print(f"[LOADED] {len(results)} result files from {args.results_dir}")
    for name in sorted(results.keys()):
        print(f"  - {name}")

    data = extract_accuracies(results)
    print_table(data)
    write_csv(data, os.path.join(outdir, "niah_summary.csv"))
    write_latex(data, os.path.join(outdir, "niah_table.tex"))


if __name__ == "__main__":
    main()

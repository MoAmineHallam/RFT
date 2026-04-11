"""
collect_all_results.py — Walk a directory tree and consolidate every
RULER S-NIAH eval JSON into one place.

Output:
  - RESULTS.md    one human-readable markdown table per experiment
  - results.csv   flat (experiment, seq_len, depth, model, accuracy, ...)
  - results.json  merged dump keyed by experiment

Finds files matching any of: niah_*.json, niah_*_full.json
Skips the *_details.json sidecars (they're huge and redundant).

Handles both eval_ruler_niah.py formats:
  - old: {"config", "results": {sl: {d: {"rft_lm":..., "baseline":...}}}}
  - new: {"config", "primary_kind", "primary_label", "results": {...}}

Usage:
  python collect_all_results.py --root "/zeng_gk/Amine/Huawei Challenge/RFT/AmineHL"
  python collect_all_results.py --root "/data3/adam_transfer/AmineHL"
  python collect_all_results.py --root /path/to/root --outdir /path/to/RESULTS
"""

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] could not parse {path}: {e}")
        return None


def find_result_files(root: Path):
    """Find every niah_*.json under root, excluding *_details.json."""
    files = []
    for p in root.rglob("niah_*.json"):
        if p.name.endswith("_details.json"):
            continue
        files.append(p)
    return sorted(files)


def experiment_tag(path: Path, root: Path) -> str:
    """
    Derive a short experiment tag from the path.
    Uses the first directory under root as the tag, which matches
    our layout: <root>/<experiment_name>/<subdir>/niah_*.json
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = rel.parts
    return parts[0] if parts else path.parent.name


def normalise_one(path: Path, root: Path):
    """Return a list of flat rows from one JSON file."""
    raw = load_json(path)
    if raw is None or "results" not in raw:
        return []

    exp = experiment_tag(path, root)
    cfg = raw.get("config", {})
    primary_label = raw.get("primary_label", "RFT-LM")
    tail_k = cfg.get("tail_chunk_len", 0)
    ablation = cfg.get("rft_ablation", "none")
    n_trials = cfg.get("n_trials", None)
    outfile = path.name

    rows = []
    for sl, depth_map in raw["results"].items():
        for d, rec in depth_map.items():
            if not isinstance(rec, dict):
                continue
            primary = rec.get("rft_lm")
            if primary is not None:
                rows.append({
                    "experiment": exp,
                    "file": outfile,
                    "path": str(path),
                    "model": primary_label,
                    "seq_len": int(sl),
                    "depth": d,
                    "n_trials": primary.get("total", n_trials),
                    "correct": primary.get("correct"),
                    "accuracy": primary.get("accuracy"),
                    "ci95_low": primary.get("ci95_low"),
                    "ci95_high": primary.get("ci95_high"),
                    "tail_chunk_len": tail_k,
                    "ablation": ablation,
                })
            bl = rec.get("baseline")
            if bl is not None:
                rows.append({
                    "experiment": exp,
                    "file": outfile,
                    "path": str(path),
                    "model": "Baseline",
                    "seq_len": int(sl),
                    "depth": d,
                    "n_trials": bl.get("total", n_trials),
                    "correct": bl.get("correct"),
                    "accuracy": bl.get("accuracy"),
                    "ci95_low": bl.get("ci95_low"),
                    "ci95_high": bl.get("ci95_high"),
                    "tail_chunk_len": tail_k,
                    "ablation": ablation,
                })
    return rows


def group_by(rows, *keys):
    out = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        out.setdefault(k, []).append(r)
    return out


def write_csv(rows, path: Path):
    if not rows:
        return
    cols = [
        "experiment", "model", "seq_len", "depth",
        "accuracy", "ci95_low", "ci95_high",
        "correct", "n_trials",
        "tail_chunk_len", "ablation", "file", "path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_markdown(rows, path: Path, root: Path):
    lines = []
    lines.append("# RULER S-NIAH — consolidated results\n")
    lines.append(f"_root_: `{root}`  \n")
    lines.append(f"_files scanned_: {len(set(r['file'] for r in rows))}  \n")
    lines.append(f"_rows_: {len(rows)}\n")

    by_exp = group_by(rows, "experiment")
    for exp in sorted(by_exp.keys()):
        exp_rows = by_exp[exp]
        lines.append(f"\n## {exp[0]}\n")
        # one table per (file, tail_chunk_len) so mixing tail_k=0 and tail_k=8 is visible
        by_file = group_by(exp_rows, "file", "tail_chunk_len", "ablation")
        for (fname, tk, abl), file_rows in sorted(by_file.items()):
            depths = sorted({r["depth"] for r in file_rows},
                            key=lambda d: float(d) if d != "random" else 2.0)
            seq_lens = sorted({r["seq_len"] for r in file_rows})
            models = sorted({r["model"] for r in file_rows})

            header = (f"\n**{fname}**  "
                      f"(tail_chunk_len={tk}"
                      + (f", ablation={abl}" if abl and abl != "none" else "")
                      + ")\n")
            lines.append(header)
            lines.append("| model | L | " + " | ".join(f"d={d}" for d in depths) + " | mean |")
            lines.append("|" + "---|" * (3 + len(depths)))
            for m in models:
                for sl in seq_lens:
                    cells = []
                    accs = []
                    for d in depths:
                        match = [r for r in file_rows
                                 if r["model"] == m and r["seq_len"] == sl and r["depth"] == d]
                        if match:
                            a = match[0]["accuracy"]
                            if a is not None:
                                accs.append(a)
                                cells.append(f"{100*a:.1f}")
                            else:
                                cells.append("—")
                        else:
                            cells.append("—")
                    mean = f"{100*sum(accs)/len(accs):.1f}" if accs else "—"
                    lines.append(f"| {m} | {sl} | " + " | ".join(cells) + f" | **{mean}** |")

    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root dir to scan recursively for niah_*.json files.")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Where to write RESULTS.md / results.csv / results.json. "
                         "Defaults to <root>/_consolidated_results.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")

    outdir = Path(args.outdir) if args.outdir else root / "_consolidated_results"
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_result_files(root)
    print(f"[SCAN] found {len(files)} result files under {root}")
    for p in files:
        print(f"  - {p.relative_to(root)}")

    all_rows = []
    merged = {}
    for p in files:
        rows = normalise_one(p, root)
        all_rows.extend(rows)
        tag = experiment_tag(p, root)
        merged.setdefault(tag, {})[p.name] = load_json(p)

    if not all_rows:
        print("[WARN] no result rows extracted")
        return

    md_path = outdir / "RESULTS.md"
    csv_path = outdir / "results.csv"
    json_path = outdir / "results.json"

    write_markdown(all_rows, md_path, root)
    write_csv(all_rows, csv_path)
    json_path.write_text(json.dumps(merged, indent=2))

    print(f"\n[WROTE] {md_path}")
    print(f"[WROTE] {csv_path}")
    print(f"[WROTE] {json_path}")
    print(f"\n{len(all_rows)} rows from {len(files)} files, "
          f"{len(set(r['experiment'] for r in all_rows))} experiments.")


if __name__ == "__main__":
    main()

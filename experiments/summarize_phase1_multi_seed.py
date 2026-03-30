"""
Aggregate multiple phase1_results.json (one per random seed) into a single report.

Reads only scalar blocks: retrieval, support_pages, and compact analysis counters.
Skips heavy nested arrays (per_query lists).

Usage
-----
python -m experiments.summarize_phase1_multi_seed \\
    --glob 'multi_seed_runs/seed_*/eval/phase1_results.json' \\
    --out_json multi_seed_runs/aggregated_results.json \\
    --out_md multi_seed_runs/aggregated_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple, Union


Scalar = Union[int, float, bool, None]


def _collect_numeric_leaves(d: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested dict to paths -> float (int coerced)."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping) and not isinstance(v, list):
            out.update(_collect_numeric_leaves(v, path))
        elif isinstance(v, bool) or v is None:
            continue
        elif isinstance(v, (int, float)):
            out[path] = float(v)
    return out


def _unflatten_to_nested(flat: Dict[str, float]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for path, val in flat.items():
        parts = path.split(".")
        cur: MutableMapping[str, Any] = root
        for p in parts[:-1]:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]  # type: ignore[assignment]
        cur[parts[-1]] = val
    return root


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values), "n": float(n)}


def aggregate_scalar_blocks(
    runs: List[Dict[str, Any]],
    block_keys: Tuple[str, ...] = ("retrieval", "support_pages"),
    include_raw_values: bool = False,
) -> Dict[str, Any]:
    """For each path in nested dicts, compute mean/std/min/max over runs."""
    per_run_flats: List[Dict[str, float]] = []
    for r in runs:
        flat: Dict[str, float] = {}
        for bk in block_keys:
            if bk in r and isinstance(r[bk], Mapping):
                flat.update(_collect_numeric_leaves(r[bk], prefix=bk))
        # Pairwise summary counts (small, useful across seeds)
        if "analysis" in r and isinstance(r["analysis"], Mapping):
            pair = r["analysis"].get("pairwise_no_graph_vs_gat")
            if isinstance(pair, Mapping):
                for k in ("n_queries", "n_no_graph_better_recall1", "n_equal_recall1"):
                    if k in pair and isinstance(pair[k], (int, float)):
                        flat[f"analysis.pairwise_no_graph_vs_gat.{k}"] = float(pair[k])
        per_run_flats.append(flat)

    all_paths = set()
    for f in per_run_flats:
        all_paths.update(f.keys())

    agg_flat: Dict[str, Dict[str, float]] = {}
    for path in sorted(all_paths):
        vals = [f[path] for f in per_run_flats if path in f]
        if vals:
            st = _stats(vals)
            agg_flat[path] = {**st, "values": vals}  # type: ignore[dict-item]

    out: Dict[str, Any] = {
        "by_metric": _unflatten_to_nested(
            {p: {k: v for k, v in st.items() if k != "values"} for p, st in agg_flat.items()}
        ),
    }
    if include_raw_values:
        out["raw_values_per_path"] = {p: st["values"] for p, st in agg_flat.items()}
    return out


def _md_table_for_block(
    by_metric: Dict[str, Any],
    block: str,
    rows: List[str],
) -> str:
    """Build a markdown table for selected metric suffixes (e.g. Recall@1) for each method."""
    methods = sorted(by_metric.get(block, {}).keys()) if isinstance(by_metric.get(block), dict) else []
    if not methods:
        return f"_(no `{block}` in aggregate)_\n"

    lines = [
        f"### {block}",
        "",
        "| Method | " + " | ".join(rows) + " |",
        "| --- | " + " | ".join(["---"] * len(rows)) + " |",
    ]
    blk = by_metric[block]
    for m in methods:
        cells = []
        for row in rows:
            sub = blk[m]
            for part in row.split("."):
                if isinstance(sub, dict) and part in sub:
                    sub = sub[part]
                else:
                    sub = None
                    break
            if isinstance(sub, dict) and "mean" in sub and "std" in sub:
                cells.append(f"{sub['mean']:.4f} ± {sub['std']:.4f}")
            elif isinstance(sub, (int, float)):
                cells.append(f"{float(sub):.4f}")
            else:
                cells.append("—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_markdown(summary: Dict[str, Any]) -> str:
    by_metric = summary.get("by_metric", {})
    retrieval_rows = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
    ]
    support_rows = [
        "Coverage@5",
        "Coverage@10",
        "AllSupportHit@10",
        "SupportF1@10",
    ]
    parts = [
        "# Multi-seed Phase 1 summary",
        "",
        f"Runs: **{summary.get('n_runs', '?')}**  |  Seeds: `{summary.get('seeds', [])}`",
        "",
        _md_table_for_block(by_metric, "retrieval", retrieval_rows),
        _md_table_for_block(by_metric, "support_pages", support_rows),
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate phase1_results.json over multiple seeds.")
    ap.add_argument("--glob", default=None, help="Glob for phase1_results.json paths")
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit list of phase1_results.json files (overrides --glob)",
    )
    ap.add_argument("--out_json", required=True, help="Output aggregated JSON path")
    ap.add_argument("--out_md", default=None, help="Optional Markdown report path")
    ap.add_argument(
        "--include_raw_values",
        action="store_true",
        help="Include per-seed lists for every metric (larger JSON).",
    )
    args = ap.parse_args()

    paths: List[str]
    if args.inputs:
        paths = list(args.inputs)
    elif args.glob:
        paths = sorted(glob.glob(args.glob))
    else:
        ap.error("Provide --glob or --inputs")

    if len(paths) < 1:
        raise SystemExit("No result files matched.")

    runs: List[Dict[str, Any]] = []
    seeds_meta: List[int] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
        pth = Path(p)
        seed_dir = pth.parent.parent.name if pth.parent.name == "eval" else pth.parent.name
        if seed_dir.startswith("seed_"):
            try:
                seeds_meta.append(int(seed_dir.split("_", 1)[1]))
            except ValueError:
                seeds_meta.append(-1)
        else:
            seeds_meta.append(-1)

    nested = aggregate_scalar_blocks(runs, include_raw_values=args.include_raw_values)
    summary = {
        "n_runs": len(runs),
        "seeds": seeds_meta,
        "result_paths": paths,
        **nested,
    }

    out_j = Path(args.out_json)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_j}")

    if args.out_md:
        out_m = Path(args.out_md)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        with open(out_m, "w", encoding="utf-8") as f:
            f.write(format_markdown(summary))
        print(f"Wrote {out_m}")


if __name__ == "__main__":
    main()

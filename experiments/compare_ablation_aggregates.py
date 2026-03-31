"""
Merge per-experiment aggregated_results.json files (from summarize_phase1_multi_seed) into one comparison table.

Scans:
    {battery_root}/*/aggregated_results.json
    {battery_root}/_summaries/*/aggregated_results.json   (after --archive_slug_then_delete)

Outputs:
    {battery_root}/ablation_comparison.md
    {battery_root}/ablation_comparison.json

Usage
-----
python -m experiments.compare_ablation_aggregates --battery_root multi_seed_runs/paper_ablation_battery
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RETRIEVAL_METRICS = ["Recall@1", "Recall@5", "MRR@10", "nDCG@10"]

# Prefer these methods when building the "primary reranker" column
METHOD_PRIORITY = [
    "EvidenceGraph-RAG (page+region)",
    "X-PageRerank (GAT)",
    "X-PageRerank (ablation: no graph)",
]


def _get_metric(
    retrieval: Dict[str, Any],
    method: str,
    metric: str,
) -> Optional[Tuple[float, float]]:
    sub = retrieval.get(method)
    if not isinstance(sub, dict):
        return None
    m = sub.get(metric)
    if not isinstance(m, dict):
        return None
    mean, std = m.get("mean"), m.get("std")
    if mean is None:
        return None
    return float(mean), float(std or 0.0)


def _pick_primary_method(retrieval: Dict[str, Any]) -> Optional[str]:
    for m in METHOD_PRIORITY:
        if m in retrieval and isinstance(retrieval[m], dict):
            if _get_metric(retrieval, m, "Recall@1") is not None:
                return m
    return None


def _pack(ms: Optional[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if ms is None:
        return None
    return {"mean": ms[0], "std": ms[1]}


def _fmt(ms: Any) -> str:
    if ms is None:
        return "—"
    if isinstance(ms, dict):
        mean, std = float(ms["mean"]), float(ms.get("std", 0.0))
    else:
        mean, std = ms[0], ms[1]
    return f"{mean:.4f} ± {std:.4f}"


def _slug_from_path(p: Path) -> str:
    return p.parent.name


def load_battery_runs(battery_root: Path) -> List[Tuple[str, Path, Dict[str, Any]]]:
    """Prefer live slug dir over _summaries when both exist (same slug name)."""
    by_slug: Dict[str, Path] = {}
    for agg in sorted(battery_root.glob("*/aggregated_results.json")):
        slug = _slug_from_path(agg)
        if slug.startswith(".") or slug == "_summaries":
            continue
        by_slug[slug] = agg
    arch = battery_root / "_summaries"
    if arch.is_dir():
        for agg in sorted(arch.glob("*/aggregated_results.json")):
            slug = _slug_from_path(agg)
            if slug.startswith("."):
                continue
            if slug not in by_slug:
                by_slug[slug] = agg
    runs: List[Tuple[str, Path, Dict[str, Any]]] = []
    for slug in sorted(by_slug.keys()):
        agg = by_slug[slug]
        with open(agg, encoding="utf-8") as f:
            data = json.load(f)
        runs.append((slug, agg, data))
    return runs


def build_comparison(battery_root: Path) -> Dict[str, Any]:
    loaded = load_battery_runs(battery_root)
    rows: List[Dict[str, Any]] = []

    for slug, path, data in loaded:
        by_metric = data.get("by_metric") or {}
        retrieval = by_metric.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        primary = _pick_primary_method(retrieval)
        row: Dict[str, Any] = {
            "slug": slug,
            "aggregated_results": str(path),
            "n_seeds": data.get("n_runs"),
            "seeds": data.get("seeds"),
            "primary_method": primary,
        }
        for metric in RETRIEVAL_METRICS:
            key = f"primary_{metric.replace('@', '_')}"
            if primary:
                row[key] = _pack(_get_metric(retrieval, primary, metric))
            else:
                row[key] = None
        if "ColPali (stage-1)" in retrieval:
            row["baseline_colpali_stage1_recall1"] = _pack(
                _get_metric(retrieval, "ColPali (stage-1)", "Recall@1")
            )
        if "ColPali + MLP reranker" in retrieval:
            row["baseline_mlp_recall1"] = _pack(
                _get_metric(retrieval, "ColPali + MLP reranker", "Recall@1")
            )
        nog = _get_metric(retrieval, "X-PageRerank (ablation: no graph)", "Recall@1")
        gat = _get_metric(retrieval, "X-PageRerank (GAT)", "Recall@1")
        if nog is not None:
            row["page_nograph_recall1"] = _pack(nog)
        if gat is not None:
            row["page_gat_recall1"] = _pack(gat)
        rows.append(row)

    return {"battery_root": str(battery_root), "runs": rows}


def format_markdown(comp: Dict[str, Any]) -> str:
    runs: List[Dict[str, Any]] = comp.get("runs") or []
    lines = [
        "# Ablation comparison (multi-seed aggregates)",
        "",
        f"Battery root: `{comp.get('battery_root')}`",
        "",
        "## Primary reranker per experiment",
        "",
        "| Experiment | Method | " + " | ".join(RETRIEVAL_METRICS) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(RETRIEVAL_METRICS)) + " |",
    ]
    for r in runs:
        slug = r.get("slug", "?")
        pm = r.get("primary_method") or "—"
        cells = []
        for metric in RETRIEVAL_METRICS:
            key = f"primary_{metric.replace('@', '_')}"
            cells.append(_fmt(r.get(key)))
        lines.append(f"| {slug} | {pm} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Page-branch: GAT vs no-graph (Recall@1)")
    lines.append("")
    lines.append("| Experiment | X-PageRerank (GAT) | no-graph |")
    lines.append("| --- | --- | --- |")
    for r in runs:
        if r.get("page_gat_recall1") is None and r.get("page_nograph_recall1") is None:
            continue
        lines.append(
            f"| {r.get('slug')} | {_fmt(r.get('page_gat_recall1'))} | {_fmt(r.get('page_nograph_recall1'))} |"
        )
    lines.append("")

    lines.append("## ColPali stage-1 Recall@1 (sanity / split effects)")
    lines.append("")
    lines.append("| Experiment | ColPali R@1 |")
    lines.append("| --- | --- |")
    for r in runs:
        key = "baseline_colpali_stage1_recall1"
        if r.get(key) is not None:
            lines.append(f"| {r.get('slug')} | {_fmt(r.get(key))} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare aggregated multi-seed runs across ablation folders.")
    ap.add_argument(
        "--battery_root",
        required=True,
        help="Directory containing one subfolder per experiment with aggregated_results.json",
    )
    args = ap.parse_args()

    root = Path(args.battery_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    comp = build_comparison(root)
    out_j = root / "ablation_comparison.json"
    out_m = root / "ablation_comparison.md"

    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)
    print(f"Wrote {out_j}")

    with open(out_m, "w", encoding="utf-8") as f:
        f.write(format_markdown(comp))
    print(f"Wrote {out_m}")


if __name__ == "__main__":
    main()

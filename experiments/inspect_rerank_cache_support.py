"""
Inspect how many positive support labels appear in a RerankDataset cache (per *_mask.pt).

Phase 1 eval builds gold pages only from pages that are both (a) in ColPali top-K
and (b) marked as support — see train_reranker.RerankDataset.build_from_index.
So for each cached row, ``mask.sum()`` is what metrics see, not necessarily the
full ``evidence_pages`` list from MP-DocVQA.

Usage
-----
python -m experiments.inspect_rerank_cache_support \\
    --cache_dir multi_seed_runs/mpdoc_val_disjoint/seed_42/splits/test

Optional: compare question_ids in this cache to raw parquet annotation counts
(same window as ``run_phase1 cache``: --sample_offset / --num_samples).

python -m experiments.inspect_rerank_cache_support \\
    --cache_dir path/to/splits/test \\
    --parquet_dir ~/.cache/huggingface/hub/datasets--lmms-lab--MP-DocVQA/snapshots/<hash>/data \\
    --split_prefix val \\
    --sample_offset 0 \\
    --num_samples 1500
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


def _mask_support_counts(cache_dir: str) -> List[int]:
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = len(meta)
    counts: List[int] = []
    for i in range(n):
        mp = os.path.join(cache_dir, f"{i}_mask.pt")
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"Missing {mp}")
        m = torch.load(mp, map_location="cpu")
        counts.append(int(m.float().sum().item()))
    return counts


def _n_pages_from_metadata(row: Dict[str, Any]) -> int:
    """Document length without decoding images (page_ids JSON or upper bound)."""
    pids = row.get("page_ids")
    if isinstance(pids, str):
        try:
            pids = json.loads(pids)
        except Exception:
            pids = None
    if isinstance(pids, list) and pids:
        return len(pids)
    return 60  # align with loader default max_pages; only bounds evidence filter


def _support_count_from_metadata_row(row: Dict[str, Any], evidence_col: Optional[str]) -> Tuple[int, Optional[str]]:
    qid = str(row.get("questionId") or row.get("question_id") or row.get("id") or "")
    n_pages = _n_pages_from_metadata(row)

    raw_idx = row.get("answer_page_idx")
    if raw_idx in (None, "", [], "null"):
        return 0, qid or None
    if isinstance(raw_idx, str):
        try:
            parsed = json.loads(raw_idx)
            raw_idx = parsed[0] if isinstance(parsed, list) and parsed else 0
        except Exception:
            raw_idx = 0
    answer_page_idx = min(int(raw_idx), max(n_pages - 1, 0))

    evidence = row.get(evidence_col) if evidence_col else None
    if evidence is not None and evidence_col:
        if isinstance(evidence, (int, float)):
            support_idxs = [int(evidence)]
        else:
            support_idxs = [int(e) for e in evidence if int(e) < max(n_pages, 1)]
        support_idxs = support_idxs or [answer_page_idx]
    else:
        support_idxs = [answer_page_idx]
    return len(set(support_idxs)), qid or None


def _parquet_qid_to_raw_support(
    parquet_dir: str,
    split_prefix: str,
    sample_offset: int,
    num_samples: Optional[int],
) -> Tuple[Dict[str, int], Optional[str]]:
    """
    Fast path: pyarrow reads only metadata columns (no image decode).

    Returns:
        (qid -> raw_support_count, evidence_column_name_or_None)
    """
    import glob

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pip install pyarrow") from e

    pattern = os.path.join(os.path.expanduser(parquet_dir), f"{split_prefix}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching {pattern}")

    schema = pq.read_schema(files[0])
    names = schema.names
    evidence_col: Optional[str] = None
    for c in ("evidence_pages", "supporting_pages", "support_pages"):
        if c in names:
            evidence_col = c
            break

    base_cols = ["questionId", "answer_page_idx"]
    if "page_ids" in names:
        base_cols.append("page_ids")
    read_cols = base_cols + ([evidence_col] if evidence_col else [])

    rows: List[Dict[str, Any]] = []
    for fpath in files:
        sch_f = pq.read_schema(fpath)
        avail = [c for c in read_cols if c in sch_f.names]
        table = pq.read_table(fpath, columns=avail)
        names_f = table.column_names
        for i in range(table.num_rows):
            rows.append({nm: table.column(nm)[i].as_py() for nm in names_f})

    total = len(rows)
    start = sample_offset
    end = min(start + num_samples, total) if num_samples is not None else total
    if start > 0 or end < total:
        rows = rows[start:end]

    out: Dict[str, int] = {}
    for row in rows:
        ns, qid = _support_count_from_metadata_row(row, evidence_col)
        if qid:
            out[qid] = ns
    return out, evidence_col


def _print_histogram(title: str, counts: List[int]) -> None:
    c = Counter(counts)
    print(f"\n=== {title} ===")
    print(f"Samples: {len(counts)}")
    for k in sorted(c.keys()):
        print(f"  support_count={k}: {c[k]} ({100.0 * c[k] / max(len(counts), 1):.2f}%)")
    multi = sum(1 for x in counts if x > 1)
    print(f"  Rows with >1 support (eval-relevant): {multi} / {len(counts)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect support-page counts in a rerank cache (mask labels)."
    )
    ap.add_argument("--cache_dir", required=True, help="Directory with meta.json and *_mask.pt")
    ap.add_argument("--parquet_dir", default=None, help="Optional: MP-DocVQA parquet dir for raw annotations")
    ap.add_argument("--split_prefix", default="val", help="Parquet file prefix (e.g. val, train)")
    ap.add_argument("--sample_offset", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=None)
    args = ap.parse_args()

    cache_dir = os.path.expanduser(args.cache_dir)
    counts = _mask_support_counts(cache_dir)
    _print_histogram("Cached mask (what Phase 1 eval uses)", counts)

    if args.parquet_dir:
        meta_path = os.path.join(cache_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        qids: Set[str] = {str(m.get("question_id", "")) for m in meta}
        qids.discard("")

        pmap, ev_col = _parquet_qid_to_raw_support(
            args.parquet_dir,
            args.split_prefix,
            args.sample_offset,
            args.num_samples,
        )
        if ev_col:
            print(f"\nParquet evidence column detected: {ev_col}")
        else:
            print(
                "\nNote: parquet has no evidence_pages / support_* column "
                "(common for lmms-lab/MP-DocVQA shards). "
                "Raw annotation is then answer_page_idx only ⇒ 1 support page per row "
                "in loader logic — multi-page *documents* still exist."
            )
        raw_counts: List[int] = []
        missing: List[str] = []
        for m in meta:
            q = str(m.get("question_id", ""))
            if q not in pmap:
                missing.append(q)
                continue
            raw_counts.append(pmap[q])

        if raw_counts:
            _print_histogram("Raw parquet annotation (unique evidence pages, same row window)", raw_counts)
        print(f"\nquestion_ids in cache meta: {len(meta)}")
        print(f"Matched to parquet rows: {len(raw_counts)}")
        if missing:
            print(
                f"Unmatched question_ids: {len(missing)} "
                "(check --split_prefix / --sample_offset / --num_samples vs cache build)"
            )


if __name__ == "__main__":
    main()

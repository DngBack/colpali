"""
Filter a pre-built RerankDataset cache by excluding documents (doc_id).

This avoids re-encoding embeddings: it copies the required *_q.pt, *_p.pt, *_s0.pt, *_mask.pt
files for samples that remain after filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Set


def _load_meta(cache_dir: str) -> List[Dict]:
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json in: {cache_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_excluded_doc_ids(exclude_dirs: List[str]) -> Set[str]:
    excluded: Set[str] = set()
    for d in exclude_dirs:
        meta = _load_meta(d)
        for m in meta:
            # Be defensive: meta.json should always include doc_id
            if "doc_id" in m:
                excluded.add(str(m["doc_id"]))
    return excluded


def filter_cache_by_doc_id(
    candidate_cache: str,
    exclude_caches: List[str],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    candidate_meta = _load_meta(candidate_cache)
    excluded_doc_ids = _collect_excluded_doc_ids(exclude_caches)

    kept_meta: List[Dict] = []
    for i, m in enumerate(candidate_meta):
        doc_id = str(m.get("doc_id", ""))
        if doc_id in excluded_doc_ids:
            continue
        kept_meta.append(m)

    # Copy required pt files and write new meta.json
    # Note: RerankDataset.save() names samples by enumeration index within that cache.
    # Since we filter, indices change; we copy from original index -> new index.
    kept_src_indices: List[int] = []
    for i, m in enumerate(candidate_meta):
        doc_id = str(m.get("doc_id", ""))
        if doc_id in excluded_doc_ids:
            continue
        kept_src_indices.append(i)

    for new_i, src_i in enumerate(kept_src_indices):
        for suffix in ("q", "p", "s0", "mask"):
            src = os.path.join(candidate_cache, f"{src_i}_{suffix}.pt")
            dst = os.path.join(output_dir, f"{new_i}_{suffix}.pt")
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing tensor file: {src}")
            shutil.copyfile(src, dst)

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(kept_meta, f)

    print(
        f"Filtered rerank cache: kept {len(kept_meta)}/{len(candidate_meta)} samples "
        f"into {output_dir}"
    )
    print(f"Excluded doc_ids: {len(excluded_doc_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter reranker cache by excluding doc_id overlaps (no re-encoding)."
    )
    parser.add_argument("--candidate_cache", required=True, help="Cache dir containing meta.json")
    parser.add_argument(
        "--exclude_cache",
        action="append",
        default=[],
        help="Cache dirs to exclude doc_ids from (can be provided multiple times).",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not args.exclude_cache:
        raise ValueError("Provide at least one --exclude_cache (e.g. train/val caches).")

    filter_cache_by_doc_id(
        candidate_cache=args.candidate_cache,
        exclude_caches=args.exclude_cache,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


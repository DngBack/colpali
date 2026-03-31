"""
Create strict train/val/test splits from one RerankDataset cache.

Guarantees:
  - No doc_id overlap across splits
  - No question_id overlap across splits

Use case:
  Build one large candidate cache from MP-DocVQA "val" rows, then split
  deterministically into train/val/test for reranker experiments.

CLI defaults: train_ratio=0.12, val_ratio=0.08, test_ratio=0.80 (by doc_id).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def _load_meta(cache_dir: str) -> List[Dict]:
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta.json in {cache_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_sample(candidate_cache: str, src_idx: int, out_dir: str, dst_idx: int) -> None:
    for suffix in ("q", "p", "s0", "mask"):
        src = os.path.join(candidate_cache, f"{src_idx}_{suffix}.pt")
        dst = os.path.join(out_dir, f"{dst_idx}_{suffix}.pt")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing tensor file: {src}")
        shutil.copyfile(src, dst)


def _write_split(
    candidate_cache: str,
    meta: List[Dict],
    src_indices: Sequence[int],
    out_dir: str,
) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    out_meta: List[Dict] = []

    for dst_idx, src_idx in enumerate(src_indices):
        _copy_sample(candidate_cache, src_idx, out_dir, dst_idx)
        out_meta.append(meta[src_idx])

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f)

    doc_ids = {str(m.get("doc_id", "")) for m in out_meta}
    qids = {str(m.get("question_id", "")) for m in out_meta}
    return {
        "n_samples": len(out_meta),
        "n_doc_ids": len(doc_ids),
        "n_question_ids": len(qids),
    }


def _validate_no_overlap(split_meta: Dict[str, List[Dict]]) -> None:
    split_docs: Dict[str, set] = {}
    split_qids: Dict[str, set] = {}
    for name, rows in split_meta.items():
        split_docs[name] = {str(r.get("doc_id", "")) for r in rows}
        split_qids[name] = {str(r.get("question_id", "")) for r in rows}

    names = list(split_meta.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            doc_overlap = split_docs[a] & split_docs[b]
            qid_overlap = split_qids[a] & split_qids[b]
            if doc_overlap:
                raise RuntimeError(f"doc_id overlap detected: {a} vs {b} ({len(doc_overlap)} overlaps)")
            if qid_overlap:
                raise RuntimeError(
                    f"question_id overlap detected: {a} vs {b} ({len(qid_overlap)} overlaps)"
                )


def make_disjoint_splits(
    candidate_cache: str,
    output_root: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict:
    meta = _load_meta(candidate_cache)
    if not meta:
        raise RuntimeError("Candidate cache is empty.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Group sample indices by doc_id to enforce document disjointness.
    by_doc: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(meta):
        doc_id = str(row.get("doc_id", ""))
        if not doc_id:
            raise ValueError(f"Missing doc_id at meta index {i}")
        qid = str(row.get("question_id", ""))
        if not qid:
            raise ValueError(f"Missing question_id at meta index {i}")
        by_doc[doc_id].append(i)

    docs = list(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(docs)

    n_docs = len(docs)
    n_train_docs = int(round(train_ratio * n_docs))
    n_val_docs = int(round(val_ratio * n_docs))
    n_train_docs = max(1, min(n_train_docs, n_docs - 2)) if n_docs >= 3 else max(1, n_docs - 1)
    n_val_docs = max(1, min(n_val_docs, n_docs - n_train_docs - 1)) if n_docs >= 3 else max(0, n_docs - n_train_docs)
    n_test_docs = n_docs - n_train_docs - n_val_docs
    if n_test_docs <= 0:
        raise RuntimeError("Not enough doc_ids to create non-empty test split.")

    train_docs = set(docs[:n_train_docs])
    val_docs = set(docs[n_train_docs:n_train_docs + n_val_docs])
    test_docs = set(docs[n_train_docs + n_val_docs:])

    split_indices = {"train": [], "val": [], "test": []}
    for d in train_docs:
        split_indices["train"].extend(by_doc[d])
    for d in val_docs:
        split_indices["val"].extend(by_doc[d])
    for d in test_docs:
        split_indices["test"].extend(by_doc[d])

    # Keep original order (stable) inside each split for reproducibility.
    for name in split_indices:
        split_indices[name].sort()

    split_meta = {name: [meta[i] for i in idxs] for name, idxs in split_indices.items()}
    _validate_no_overlap(split_meta)

    os.makedirs(output_root, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}
    for name in ("train", "val", "test"):
        out_dir = os.path.join(output_root, name)
        summary[name] = _write_split(candidate_cache, meta, split_indices[name], out_dir)

    summary_full = {
        "candidate_cache": candidate_cache,
        "output_root": output_root,
        "seed": seed,
        "ratios": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "num_docs": {
            "train": len(train_docs),
            "val": len(val_docs),
            "test": len(test_docs),
            "total": n_docs,
        },
        "num_samples": summary,
        "constraints": {
            "doc_id_overlap": "none",
            "question_id_overlap": "none",
        },
    }

    with open(os.path.join(output_root, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_full, f, indent=2)

    return summary_full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test caches with strict doc/question disjointness."
    )
    parser.add_argument("--candidate_cache", required=True, help="Source cache with meta.json and *_*.pt files")
    parser.add_argument("--output_root", required=True, help="Output directory containing train/ val/ test/")
    parser.add_argument("--train_ratio", type=float, default=0.12)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = make_disjoint_splits(
        candidate_cache=args.candidate_cache,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


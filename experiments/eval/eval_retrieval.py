"""
Standard IR retrieval metrics for X-PageRerank.

Metrics implemented:
  - Recall@k
  - MRR@k  (Mean Reciprocal Rank)
  - nDCG@k (Normalised Discounted Cumulative Gain)
  - Precision@k
  - MAP@k  (Mean Average Precision)

All functions follow a consistent interface:
    predictions:  List[List[int]]  — ranked list of page indices per query
    ground_truth: List[List[int]]  — relevant page indices per query

Example
-------
>>> preds = [[3, 7, 8, 12, 1], [5, 2, 9]]
>>> golds = [[7, 12], [2]]
>>> print(recall_at_k(preds, golds, k=5))
>>> results = evaluate_retrieval(preds, golds, k_values=[1, 5, 10])
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """
    Recall@k: fraction of queries where at least one relevant page
    appears in the top-k predictions.

    This is the "any-hit" variant used by MP-DocVQA.
    For multi-support queries, it requires at least one support page hit.
    """
    hits = 0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        if any(p in gold_set for p in pred[:k]):
            hits += 1
    return hits / max(len(predictions), 1)


def recall_at_k_all_support(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """
    Strict Recall@k: requires ALL support pages to be in the top-k.
    Used as a secondary metric for multi-hop questions.
    """
    hits = 0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        pred_set = set(pred[:k])
        if gold_set.issubset(pred_set):
            hits += 1
    return hits / max(len(predictions), 1)


def precision_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """Precision@k: fraction of top-k predictions that are relevant."""
    total_prec = 0.0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        top_k = pred[:k]
        hits = sum(1 for p in top_k if p in gold_set)
        total_prec += hits / max(k, 1)
    return total_prec / max(len(predictions), 1)


def mrr_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """
    MRR@k: Mean Reciprocal Rank at cutoff k.
    For each query, the reciprocal rank of the first relevant item in top-k.
    """
    total_rr = 0.0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        rr = 0.0
        for rank, p in enumerate(pred[:k], start=1):
            if p in gold_set:
                rr = 1.0 / rank
                break
        total_rr += rr
    return total_rr / max(len(predictions), 1)


def average_precision(pred: List[int], gold: List[int], k: int) -> float:
    """Average precision at k for a single query."""
    gold_set = set(gold)
    hits = 0
    sum_prec = 0.0
    for rank, p in enumerate(pred[:k], start=1):
        if p in gold_set:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / max(len(gold_set), 1)


def map_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """MAP@k: Mean Average Precision at cutoff k."""
    return sum(
        average_precision(pred, gold, k)
        for pred, gold in zip(predictions, ground_truth)
    ) / max(len(predictions), 1)


def _dcg_at_k(relevance: List[int], k: int) -> float:
    """DCG@k for a single query given a list of graded relevance scores."""
    dcg = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """
    nDCG@k: Normalised Discounted Cumulative Gain.
    Uses binary relevance (0 or 1).
    """
    total_ndcg = 0.0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        relevance = [1 if p in gold_set else 0 for p in pred[:k]]
        dcg = _dcg_at_k(relevance, k)
        ideal = [1] * min(len(gold_set), k)
        idcg = _dcg_at_k(ideal, k)
        total_ndcg += dcg / max(idcg, 1e-8)
    return total_ndcg / max(len(predictions), 1)


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k_values: Optional[List[int]] = None,
    dataset_name: str = "",
    method_name: str = "",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute all retrieval metrics at multiple k values.

    Args:
        predictions:  (Q, ranked_pages)  per-query ranked page indices
        ground_truth: (Q, relevant_pages) per-query relevant page indices
        k_values:     list of k cutoffs to evaluate (default: [1, 5, 10])
        dataset_name: for display
        method_name:  for display
        verbose:      print table to stdout

    Returns:
        Dict mapping metric names to float values.
    """
    k_values = k_values or [1, 5, 10]
    max_k = max(k_values)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        metrics[f"Recall_all@{k}"] = recall_at_k_all_support(predictions, ground_truth, k)
        metrics[f"P@{k}"] = precision_at_k(predictions, ground_truth, k)
        metrics[f"MAP@{k}"] = map_at_k(predictions, ground_truth, k)

    metrics["MRR@10"] = mrr_at_k(predictions, ground_truth, 10)
    metrics["nDCG@10"] = ndcg_at_k(predictions, ground_truth, 10)

    if verbose:
        tag = f"{dataset_name} / {method_name}" if dataset_name or method_name else "Results"
        print(f"\n=== Retrieval Metrics — {tag} ===")
        print(f"  Queries evaluated: {len(predictions)}")
        for name, val in metrics.items():
            print(f"  {name:<20}: {val:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Per-query analysis
# ---------------------------------------------------------------------------

def per_query_metrics(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    question_ids: Optional[List[str]] = None,
    k: int = 10,
) -> List[Dict]:
    """
    Compute per-query metrics for error analysis.

    Returns list of dicts with keys:
      question_id, rank_of_first_hit, recall@k, is_cross_page
    """
    results = []
    qids = question_ids or [str(i) for i in range(len(predictions))]

    for qid, pred, gold in zip(qids, predictions, ground_truth):
        gold_set = set(gold)
        # Rank of first hit
        rank = None
        for r, p in enumerate(pred[:k], start=1):
            if p in gold_set:
                rank = r
                break

        results.append({
            "question_id": qid,
            "rank_of_first_hit": rank,
            "recall_at_k": int(rank is not None),
            "num_support_pages": len(gold),
            "is_cross_page": len(gold) > 1,
        })
    return results


# ---------------------------------------------------------------------------
# Baseline comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Print a LaTeX-style comparison table.

    Args:
        results:  {"ColPali": {"Recall@1": 0.3, ...}, "X-PageRerank": {...}, ...}
        metrics:  list of metric names to display (default: standard set)
    """
    metrics = metrics or ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    methods = list(results.keys())

    # Header
    col_w = 15
    header = "Method".ljust(col_w) + "".join(m.ljust(col_w) for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for method in methods:
        row = method.ljust(col_w)
        for m in metrics:
            val = results[method].get(m, float("nan"))
            row += f"{val:.4f}".ljust(col_w)
        print(row)

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Tensor-based fast evaluation (for training loops)
# ---------------------------------------------------------------------------

def recall_at_k_tensor(
    scores: torch.Tensor,           # (Q, N) page scores
    support_masks: torch.Tensor,    # (Q, N) binary support
    k: int,
) -> float:
    """Fast tensor-based Recall@k, used in training loops."""
    _, top_idx = scores.topk(min(k, scores.shape[1]), dim=1, sorted=True)
    top_support = support_masks.gather(1, top_idx)   # (Q, k)
    any_hit = (top_support.sum(dim=1) > 0).float()
    return any_hit.mean().item()

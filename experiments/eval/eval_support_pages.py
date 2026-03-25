"""
Cross-page evidence metrics for X-PageRerank.

These metrics go beyond standard IR retrieval metrics to specifically
measure the ability to retrieve all evidence required for multi-hop questions.

Metrics:
  BothSupportHit@k    — ALL support pages appear in top-k (strict multi-hop)
  Support-Page F1     — micro F1 between predicted and gold support pages
  Evidence Coverage   — recall of support pages in top-k
  Page-Gap Accuracy   — whether the predicted pages span the correct page range

These are the "cross-page evidence metrics" described in Section 10 of idea.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# BothSupportHit@k  (and generalised AllSupportHit@k)
# ---------------------------------------------------------------------------

def both_support_hit_at_k(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
    min_support_pages: int = 2,
) -> float:
    """
    AllSupportHit@k: fraction of (multi-page) queries where ALL support pages
    appear in the top-k predicted pages.

    Only evaluated on queries with >= `min_support_pages` support pages.

    Args:
        predictions:       (Q, ranked_pages)
        ground_truth:      (Q, support_pages)
        k:                 retrieval cutoff
        min_support_pages: minimum # support pages for a query to be included

    Returns:
        float in [0, 1], or NaN if no eligible queries exist.
    """
    hits = 0
    eligible = 0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        if len(gold_set) < min_support_pages:
            continue
        eligible += 1
        pred_set = set(pred[:k])
        if gold_set.issubset(pred_set):
            hits += 1

    if eligible == 0:
        return float("nan")
    return hits / eligible


def support_hit_at_k_by_count(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> Dict[int, float]:
    """
    Stratify AllSupportHit@k by the number of support pages required.

    Returns dict: {num_support_pages: AllSupportHit@k}
    """
    by_count: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))  # (hits, total)

    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        n = len(gold_set)
        pred_set = set(pred[:k])
        hit = int(gold_set.issubset(pred_set))
        prev_h, prev_t = by_count[n]
        by_count[n] = (prev_h + hit, prev_t + 1)

    return {n: h / t for n, (h, t) in sorted(by_count.items())}


# ---------------------------------------------------------------------------
# Support-Page F1
# ---------------------------------------------------------------------------

def support_page_precision_recall_f1(
    predicted_pages: List[int],   # top-k predicted pages for one query
    gold_pages: List[int],        # gold support pages for one query
) -> Tuple[float, float, float]:
    """
    Compute P, R, F1 for a single query's support page prediction.
    """
    gold_set = set(gold_pages)
    pred_set = set(predicted_pages)

    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 1.0, 1.0, 1.0  # vacuously true if no gold pages

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def support_page_f1(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Macro-averaged Support-Page P, R, F1 over all queries.

    Args:
        predictions:  (Q, ranked_pages)
        ground_truth: (Q, support_pages)
        k:            truncate predictions at top-k before computing F1
                      (None = use full prediction list)

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    for pred, gold in zip(predictions, ground_truth):
        p_trunc = pred[:k] if k is not None else pred
        p, r, f1 = support_page_precision_recall_f1(p_trunc, gold)
        total_p += p
        total_r += r
        total_f1 += f1

    n = max(len(predictions), 1)
    return {
        "precision": total_p / n,
        "recall": total_r / n,
        "f1": total_f1 / n,
    }


# ---------------------------------------------------------------------------
# Evidence Coverage
# ---------------------------------------------------------------------------

def evidence_coverage(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """
    Average fraction of support pages retrieved in the top-k.

    Unlike BothSupportHit@k (which requires ALL), this gives partial credit
    for retrieving some of the required support pages.

    Returns float in [0, 1].
    """
    total_coverage = 0.0
    for pred, gold in zip(predictions, ground_truth):
        gold_set = set(gold)
        if not gold_set:
            continue
        pred_set = set(pred[:k])
        total_coverage += len(pred_set & gold_set) / len(gold_set)
    return total_coverage / max(len(predictions), 1)


# ---------------------------------------------------------------------------
# Page-Gap Accuracy (structural multi-hop metric)
# ---------------------------------------------------------------------------

def page_gap_accuracy(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
    gap_tolerance: int = 2,
) -> float:
    """
    For cross-page questions, check whether predicted pages span
    approximately the same page-number range as gold support pages.

    Only evaluated on queries with >= 2 support pages.

    Args:
        gap_tolerance: allow predicted span to be within ±gap_tolerance
                       of gold span size.
    """
    hits = 0
    eligible = 0
    for pred, gold in zip(predictions, ground_truth):
        if len(set(gold)) < 2:
            continue
        eligible += 1
        gold_span = max(gold) - min(gold)
        pred_top = pred[:k]
        if not pred_top:
            continue
        pred_span = max(pred_top) - min(pred_top)
        if abs(pred_span - gold_span) <= gap_tolerance:
            hits += 1

    return hits / max(eligible, 1) if eligible > 0 else float("nan")


# ---------------------------------------------------------------------------
# Query-type stratified evaluation
# ---------------------------------------------------------------------------

def stratified_evaluation(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    query_types: List[str],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately by query type.

    Args:
        query_types: list of query type strings per query
                     e.g. ["single-hop", "cross-page", "chart+text", ...]

    Returns:
        dict: {query_type: {metric_name: value}}
    """
    from .eval_retrieval import recall_at_k, mrr_at_k, ndcg_at_k

    # Group indices by type
    type_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, qt in enumerate(query_types):
        type_to_indices[qt].append(i)

    results = {}
    for qt, indices in type_to_indices.items():
        preds_q = [predictions[i] for i in indices]
        golds_q = [ground_truth[i] for i in indices]
        f1_res = support_page_f1(preds_q, golds_q, k=k)
        results[qt] = {
            f"Recall@{k}": recall_at_k(preds_q, golds_q, k),
            "MRR@10": mrr_at_k(preds_q, golds_q, 10),
            "nDCG@10": ndcg_at_k(preds_q, golds_q, 10),
            f"AllSupportHit@{k}": both_support_hit_at_k(preds_q, golds_q, k),
            f"Coverage@{k}": evidence_coverage(preds_q, golds_q, k),
            "SupportF1": f1_res["f1"],
            "n_queries": len(indices),
        }
    return results


# ---------------------------------------------------------------------------
# Comprehensive cross-page evaluation
# ---------------------------------------------------------------------------

def evaluate_support_pages(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    k_values: Optional[List[int]] = None,
    query_types: Optional[List[str]] = None,
    dataset_name: str = "",
    method_name: str = "",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Full cross-page evidence evaluation.

    Args:
        predictions:  (Q, ranked_pages)
        ground_truth: (Q, support_pages)
        k_values:     cutoffs to evaluate at
        query_types:  optional per-query type annotations
        verbose:      print results to stdout

    Returns:
        flat dict of all metric values
    """
    k_values = k_values or [5, 10]
    metrics: Dict[str, float] = {}

    for k in k_values:
        metrics[f"Coverage@{k}"] = evidence_coverage(predictions, ground_truth, k)
        metrics[f"AllSupportHit@{k}"] = both_support_hit_at_k(predictions, ground_truth, k)
        f1_res = support_page_f1(predictions, ground_truth, k=k)
        metrics[f"SupportF1@{k}"] = f1_res["f1"]
        metrics[f"SupportP@{k}"] = f1_res["precision"]
        metrics[f"SupportR@{k}"] = f1_res["recall"]

    # Cross-page specific (multi-support queries only)
    cross_pred = [p for p, g in zip(predictions, ground_truth) if len(set(g)) > 1]
    cross_gold = [g for g in ground_truth if len(set(g)) > 1]
    if cross_pred:
        for k in k_values:
            metrics[f"CrossPage_Coverage@{k}"] = evidence_coverage(cross_pred, cross_gold, k)
            metrics[f"CrossPage_AllHit@{k}"] = both_support_hit_at_k(cross_pred, cross_gold, k)

    if verbose:
        tag = f"{dataset_name} / {method_name}" if dataset_name or method_name else "Results"
        n_cross = len(cross_pred)
        print(f"\n=== Cross-Page Evidence Metrics — {tag} ===")
        print(f"  Total queries: {len(predictions)}")
        print(f"  Cross-page queries: {n_cross}")
        for name, val in metrics.items():
            v_str = f"{val:.4f}" if not math.isnan(val) else "N/A"
            print(f"  {name:<30}: {v_str}")

    return metrics

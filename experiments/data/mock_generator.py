"""
Mock data generator for pipeline smoke-testing without downloading real datasets.

Generates synthetic RerankDataset samples with random embeddings, so the
full train → eval pipeline can be tested locally in seconds.
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional

import torch


def generate_mock_rerank_dataset(
    num_samples: int = 50,
    top_k: int = 10,
    embedding_dim: int = 128,
    seq_len: int = 32,
    min_support: int = 1,
    max_support: int = 3,
    seed: int = 42,
) -> "RerankDataset":  # noqa: F821 — avoid circular import
    """
    Generate a synthetic RerankDataset for smoke-testing.

    Each sample has:
    - Random multi-vector query embedding  (1, seq_len, dim)
    - Random multi-vector page embeddings  (top_k, seq_len, dim)
    - Random stage-1 scores               (top_k,)
    - Random support_mask                 (top_k,) with 1–3 support pages

    Returns a RerankDataset ready for training/evaluation.
    """
    # Import here to avoid circular dependency
    from ..train.train_reranker import RerankDataset, RerankSample

    rng = random.Random(seed)
    torch.manual_seed(seed)

    samples: List[RerankSample] = []
    for i in range(num_samples):
        K = top_k

        # Random L2-normalised embeddings
        q_embs = torch.randn(1, seq_len, embedding_dim)
        q_embs = q_embs / q_embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        p_embs = torch.randn(K, seq_len, embedding_dim)
        p_embs = p_embs / p_embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Stage-1 scores: roughly decreasing (simulate rank order)
        stage1 = torch.linspace(1.0, 0.3, K) + torch.randn(K) * 0.05

        # Support pages: randomly pick 1–max_support
        n_support = rng.randint(min_support, min(max_support, K))
        support_indices = rng.sample(range(K), n_support)
        support_mask = torch.zeros(K)
        support_mask[support_indices] = 1.0

        samples.append(RerankSample(
            question_id=f"mock_{i:04d}",
            query_embs=q_embs,
            page_embs=p_embs,
            page_numbers=list(range(K)),
            stage1_scores=stage1,
            support_mask=support_mask,
            doc_id=f"doc_{i // 5:04d}",
        ))

    return RerankDataset(samples)


def save_mock_cache(
    output_dir: str,
    num_train: int = 200,
    num_val: int = 50,
    top_k: int = 10,
    embedding_dim: int = 128,
    seed: int = 42,
) -> None:
    """
    Generate and save mock train+val caches to disk.

    Usage:
        from experiments.data.mock_generator import save_mock_cache
        save_mock_cache("cache/mock_train", num_train=200)
        save_mock_cache("cache/mock_val",   num_train=50, seed=99)
    """
    train_ds = generate_mock_rerank_dataset(
        num_samples=num_train, top_k=top_k, embedding_dim=embedding_dim, seed=seed,
    )
    train_out = os.path.join(output_dir, "train")
    train_ds.save(train_out)

    val_ds = generate_mock_rerank_dataset(
        num_samples=num_val, top_k=top_k, embedding_dim=embedding_dim, seed=seed + 1,
    )
    val_out = os.path.join(output_dir, "val")
    val_ds.save(val_out)

    print(f"Mock caches saved:")
    print(f"  train: {train_out}  ({num_train} samples)")
    print(f"  val:   {val_out}  ({num_val} samples)")
    print(f"\nRun training with:")
    print(f"  python -m experiments.run_phase1 train \\")
    print(f"      --train_cache {train_out} --val_cache {val_out} \\")
    print(f"      --output_dir checkpoints/mock_test")

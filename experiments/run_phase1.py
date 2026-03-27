"""
run_phase1.py — End-to-end Phase 1 experiment for X-PageRerank.

Phase 1 design (from idea.md §12):
  - backbone:  ColPali  (or ColQwen2)
  - node:      page-level only
  - graph:     semantic edges + page adjacency
  - model:     2-layer GAT (PageGATReranker)
  - task:      page reranking on MP-DocVQA
  - metrics:   Recall@k, MRR@10, nDCG@10, AllSupportHit@k, SupportF1

The script has three subcommands:

  1. cache   — encode pages + queries, cache top-K embeddings to disk
  2. train   — train the GAT reranker on cached train split
  3. eval    — evaluate all baselines + X-PageRerank on cached val/test split

Usage examples
--------------
# Step 1: build embedding caches (run once, takes hours for full dataset)
python -m experiments.run_phase1 cache \\
    --split train \\
    --model_name vidore/colpali-v1.2 \\
    --model_type colpali \\
    --output_dir cache/phase1_train \\
    --top_k 20

python -m experiments.run_phase1 cache \\
    --split validation \\
    --model_name vidore/colpali-v1.2 \\
    --model_type colpali \\
    --output_dir cache/phase1_val \\
    --top_k 20

# Step 2: train GAT reranker
python -m experiments.run_phase1 train \\
    --train_cache cache/phase1_train \\
    --val_cache   cache/phase1_val \\
    --output_dir  checkpoints/gat_phase1 \\
    --num_epochs 10

# Step 3: evaluate (all baselines + ablations + X-PageRerank)
python -m experiments.run_phase1 eval \\
    --eval_cache  cache/phase1_val \\
    --checkpoint  checkpoints/gat_phase1/best.pt \\
    --output_dir  results/phase1
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

# Local imports
from .data.mpdocvqa_loader import MPDocVQADataset
from .retrieval.colpali_infer import ColVisionInferencer, maxsim_score, pool_multivector
from .graph.build_query_graph import EvidenceGraph, EvidenceGraphConfig, build_evidence_graph_from_retrieval
from .models.page_gat_reranker import GATConfig, PageGATReranker, MLPReranker
from .train.train_reranker import (
    RerankDataset, RerankSample, RerankerTrainer, TrainingConfig, collate_rerank,
)
from .eval.eval_retrieval import evaluate_retrieval, print_comparison_table
from .eval.eval_support_pages import evaluate_support_pages, both_support_hit_at_k

logger = logging.getLogger(__name__)


def _parquet_split_row_count(parquet_dir: str, split_prefix: str) -> int:
    """Total rows in split without decoding images (cheap)."""
    pattern = os.path.join(os.path.expanduser(parquet_dir), f"{split_prefix}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files matching '{pattern}'. "
            f"Directory: {os.listdir(os.path.expanduser(parquet_dir))}"
        )
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(sum(pq.read_metadata(f).num_rows for f in files))
    except Exception:
        from datasets import load_dataset

        raw = load_dataset("parquet", data_files={split_prefix: files}, split=split_prefix)
        return len(raw)


def _append_rerank_cache_files(
    output_dir: str,
    samples: List[RerankSample],
    start_index: int,
) -> Tuple[int, List[dict]]:
    """Write RerankSample tensors and return (next_index, meta rows)."""
    meta_rows: List[dict] = []
    for i, s in enumerate(samples):
        idx = start_index + i
        torch.save(s.query_embs, os.path.join(output_dir, f"{idx}_q.pt"))
        torch.save(s.page_embs, os.path.join(output_dir, f"{idx}_p.pt"))
        torch.save(s.stage1_scores, os.path.join(output_dir, f"{idx}_s0.pt"))
        torch.save(s.support_mask, os.path.join(output_dir, f"{idx}_mask.pt"))
        meta_rows.append({
            "question_id": s.question_id,
            "doc_id": s.doc_id,
            "page_numbers": s.page_numbers,
        })
    return start_index + len(samples), meta_rows


def _resolve_graph_hparams(args) -> Tuple[float, int]:
    """
    Resolve graph hyperparameters from optional preset + CLI overrides.

    Presets:
      - default: keep current CLI values (sem_threshold, adj_max_gap)
      - sparse-graph: fewer, stronger graph edges
    """
    sem_threshold = args.sem_threshold
    adj_max_gap = args.adj_max_gap

    if getattr(args, "graph_preset", "default") == "sparse-graph":
        sem_threshold = max(sem_threshold, 0.75)
        adj_max_gap = min(adj_max_gap, 1)

    return sem_threshold, adj_max_gap


def _doc_ids_from_rerank_cache_dir(cache_dir: str) -> set:
    """Unique doc_id values stored in a train RerankDataset cache (meta.json)."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Cannot load train doc ids: missing meta.json in {cache_dir}"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {str(m["doc_id"]) for m in meta if m.get("doc_id")}


def _filter_mpdoc_samples_doc_disjoint(dataset: MPDocVQADataset, banned_doc_ids: set) -> None:
    """In-place: keep only samples whose doc_id is not in banned_doc_ids."""
    before = len(dataset.samples)
    dataset.samples = [s for s in dataset.samples if str(s.doc_id) not in banned_doc_ids]
    after = len(dataset.samples)
    logger.info(
        "Doc-disjoint filter: kept %d/%d samples (%d rows dropped as doc_id overlap)",
        after,
        before,
        before - after,
    )
    if after == 0:
        raise RuntimeError(
            "After doc-disjoint filtering, no samples remain. "
            "Widen the val candidate window (--sample_offset / --num_samples)."
        )


# ===========================================================================
# Sub-command: cache
# ===========================================================================

def cmd_mock(args):
    """Generate synthetic mock cache — no HuggingFace download needed."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("=== Generating mock cache (no download) ===")

    from .data.mock_generator import generate_mock_rerank_dataset

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")

    train_ds = generate_mock_rerank_dataset(
        num_samples=args.num_train,
        top_k=args.top_k,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    train_ds.save(train_dir)

    val_ds = generate_mock_rerank_dataset(
        num_samples=args.num_val,
        top_k=args.top_k,
        embedding_dim=args.embedding_dim,
        seed=args.seed + 1,
    )
    val_ds.save(val_dir)

    logger.info("Mock train cache: %s (%d samples)", train_dir, args.num_train)
    logger.info("Mock val   cache: %s (%d samples)", val_dir, args.num_val)
    print("\n=== Next steps ===")
    print(f"python -m experiments.run_phase1 train \\")
    print(f"    --train_cache {train_dir} \\")
    print(f"    --val_cache   {val_dir} \\")
    print(f"    --output_dir  checkpoints/mock_test")


def cmd_cache(args):
    """Encode pages + queries and save top-K RerankDataset to disk."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("=== Phase 1 — Cache embeddings [%s] ===", args.split)

    # Force offline mode if requested
    if getattr(args, "offline", False):
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Offline mode enabled")

    # Resolve HF token: CLI arg > env var > .env file
    hf_token = getattr(args, "hf_token", None) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    hf_token = line.strip().split("=", 1)[1]
    if hf_token and not os.environ.get("HF_DATASETS_OFFLINE"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        logger.info("HuggingFace token loaded")

    # Load dataset
    # Normalise split name ("validation" → "val" for parquet files from lmms-lab)
    split_for_parquet = "val" if args.split in ("val", "validation") else args.split

    sample_offset = getattr(args, "sample_offset", 0)
    num_samples   = getattr(args, "num_samples", None)

    parquet_dir = getattr(args, "parquet_dir", None)
    if isinstance(parquet_dir, str):
        parquet_dir = parquet_dir.strip() or None
    chunk_rows = int(getattr(args, "cache_chunk_rows", 0) or 0)
    streaming = getattr(args, "streaming", False)

    if chunk_rows > 0 and streaming:
        logger.warning(
            "--cache_chunk_rows is ignored when --streaming is set (HF streaming iterator)."
        )
        chunk_rows = 0

    # Chunked mode needs a bounded row window: parquet (count rows) or explicit --num_samples.
    use_chunked = chunk_rows > 0 and (parquet_dir is not None or num_samples is not None)
    if chunk_rows > 0 and not use_chunked and not streaming:
        logger.warning(
            "--cache_chunk_rows ignored without --parquet_dir and without --num_samples "
            "(load full HF split in one pass; set --num_samples to enable chunking)."
        )

    train_cache_for_exclusion = getattr(args, "exclude_train_doc_ids_cache", None)
    banned: Optional[Set[str]] = None
    if train_cache_for_exclusion:
        banned = _doc_ids_from_rerank_cache_dir(train_cache_for_exclusion)
        logger.info(
            "Excluding %d doc_ids from train cache %s",
            len(banned),
            train_cache_for_exclusion,
        )

    inferencer = ColVisionInferencer(
        model_name_or_path=args.model_name,
        model_type=args.model_type,
        batch_size=args.batch_size,
        maxsim_doc_chunk=getattr(args, "maxsim_doc_chunk", 16),
    )

    if use_chunked:
        if parquet_dir:
            total_rows = _parquet_split_row_count(parquet_dir, split_for_parquet)
            if num_samples is not None:
                n_to_process = min(num_samples, max(0, total_rows - sample_offset))
            else:
                n_to_process = max(0, total_rows - sample_offset)
        else:
            if num_samples is None:
                raise ValueError(
                    "Chunked cache needs a row budget: pass a non-empty --parquet_dir "
                    "(offline MP-DocVQA parquet, recommended) or --num_samples for HF. "
                    "If you use --parquet_dir \"$PARQUET_DIR\", ensure the variable is "
                    "set (e.g. echo \"$PARQUET_DIR\") — an empty path is ignored and "
                    "falls back to HF."
                )
            n_to_process = num_samples

        if n_to_process <= 0:
            raise RuntimeError("Nothing to process (check --sample_offset / --num_samples).")

        os.makedirs(args.output_dir, exist_ok=True)
        meta_all: List[dict] = []
        global_idx = 0
        n_chunks = (n_to_process + chunk_rows - 1) // chunk_rows
        logger.info(
            "Chunked cache: %d dataset rows in %d chunks of up to %d "
            "(lower peak RAM — avoids loading all page images at once).",
            n_to_process,
            n_chunks,
            chunk_rows,
        )

        for chunk_i, local_start in enumerate(range(0, n_to_process, chunk_rows)):
            n_chunk = min(chunk_rows, n_to_process - local_start)
            abs_offset = sample_offset + local_start
            logger.info(
                "Chunk %d/%d: rows [%d, %d) (absolute offset=%d, n=%d)",
                chunk_i + 1,
                n_chunks,
                local_start,
                local_start + n_chunk,
                abs_offset,
                n_chunk,
            )

            if parquet_dir:
                dataset = MPDocVQADataset.from_parquet_dir(
                    parquet_dir=parquet_dir,
                    split_prefix=split_for_parquet,
                    max_pages_per_doc=args.max_pages,
                    num_samples=n_chunk,
                    sample_offset=abs_offset,
                )
            else:
                dataset = MPDocVQADataset(
                    split=args.split,
                    hf_dataset_id=getattr(args, "hf_dataset_id", None),
                    local_json_path=getattr(args, "local_json_path", None),
                    max_pages_per_doc=args.max_pages,
                    hf_token=hf_token,
                    streaming=False,
                    num_samples=n_chunk,
                    sample_offset=abs_offset,
                )

            if banned is not None:
                _filter_mpdoc_samples_doc_disjoint(dataset, banned)

            rerank_ds = RerankDataset.build_from_index(
                dataset=dataset,
                index=None,
                inferencer=inferencer,
                top_k=args.top_k,
                show_progress=True,
            )
            global_idx, rows = _append_rerank_cache_files(
                args.output_dir, rerank_ds.samples, global_idx,
            )
            meta_all.extend(rows)

            del dataset
            del rerank_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_all, f)
        logger.info(
            "Done (chunked). Saved %d cached samples to %s",
            len(meta_all),
            args.output_dir,
        )
        return

    if parquet_dir:
        dataset = MPDocVQADataset.from_parquet_dir(
            parquet_dir=parquet_dir,
            split_prefix=split_for_parquet,
            max_pages_per_doc=args.max_pages,
            num_samples=num_samples,
            sample_offset=sample_offset,
        )
    else:
        dataset = MPDocVQADataset(
            split=args.split,
            hf_dataset_id=getattr(args, "hf_dataset_id", None),
            local_json_path=getattr(args, "local_json_path", None),
            max_pages_per_doc=args.max_pages,
            hf_token=hf_token,
            streaming=streaming,
            num_samples=num_samples,
            sample_offset=sample_offset,
        )

    if banned is not None:
        _filter_mpdoc_samples_doc_disjoint(dataset, banned)

    dataset.print_stats()

    rerank_ds = RerankDataset.build_from_index(
        dataset=dataset,
        index=None,
        inferencer=inferencer,
        top_k=args.top_k,
        show_progress=True,
    )
    rerank_ds.save(args.output_dir)
    logger.info("Done. Saved to %s", args.output_dir)


# ===========================================================================
# Sub-command: train
# ===========================================================================

def cmd_train(args):
    """Train the GAT reranker on cached embeddings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("=== Phase 1 — Train GAT Reranker ===")

    train_ds = RerankDataset.load(args.train_cache)
    val_ds = RerankDataset.load(args.val_cache) if args.val_cache else None

    # Infer input_dim from first sample
    sem_threshold, adj_max_gap = _resolve_graph_hparams(args)
    logger.info(
        "Graph preset=%s -> sem_threshold=%.2f, adj_max_gap=%d",
        args.graph_preset,
        sem_threshold,
        adj_max_gap,
    )
    graph_cfg = EvidenceGraphConfig(
        sem_threshold=sem_threshold,
        adj_max_gap=adj_max_gap,
        include_query_node=True,
        inject_query_diff=True,
        concat_stage1_score=True,
    )
    sample0 = train_ds[0]
    g0 = build_evidence_graph_from_retrieval(
        page_embs=sample0.page_embs,
        query_embs=sample0.query_embs,
        page_numbers=sample0.page_numbers,
        stage1_scores=sample0.stage1_scores.tolist(),
        config=graph_cfg,
    )
    input_dim = g0.feat_dim
    logger.info("Input dim = %d", input_dim)

    model_cfg = GATConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=64,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.1,
        lambda_mix=0.5,
    )
    model = PageGATReranker(config=model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("PageGATReranker: %d parameters", n_params)

    train_cfg = TrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        top_k=args.top_k,
        sem_threshold=sem_threshold,
        adj_max_gap=adj_max_gap,
    )
    trainer = RerankerTrainer(model=model, config=train_cfg, graph_config=graph_cfg)
    results = trainer.train(train_ds, val_ds)

    print(f"\n=== Training complete ===")
    print(f"Best Recall@5: {results['best_recall@5']:.4f}")
    print(f"Checkpoint: {args.output_dir}/best.pt")


# ===========================================================================
# Sub-command: eval
# ===========================================================================

def _run_colpali_baseline(eval_ds: RerankDataset, k: int) -> Tuple[List, List]:
    """Stage-1 ColPali ranking (no reranking)."""
    all_preds, all_golds = [], []
    for sample in eval_ds:
        sorted_local = sample.stage1_scores.argsort(descending=True).tolist()
        pred = [sample.page_numbers[i] for i in sorted_local[:k]]
        gold = [sample.page_numbers[i] for i, v in enumerate(sample.support_mask.tolist()) if v > 0]
        all_preds.append(pred)
        all_golds.append(gold)
    return all_preds, all_golds


def _run_naive_concat_baseline(eval_ds: RerankDataset, k: int) -> Tuple[List, List]:
    """
    ColPali + naive concat: take all top-k pages from stage-1
    in order — same as ColPali baseline, but evaluated at higher k.
    This simulates providing k pages to a downstream model.
    """
    return _run_colpali_baseline(eval_ds, k)


def _run_mlp_baseline(
    eval_ds: RerankDataset,
    model: MLPReranker,
    device: torch.device,
    k: int,
) -> Tuple[List, List]:
    """MLP reranker baseline (no graph)."""
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for sample in eval_ds:
            p_vecs = pool_multivector(sample.page_embs).to(device)   # (K, D)
            q_vec = pool_multivector(sample.query_embs).squeeze(0).to(device)   # (D,)
            s0 = sample.stage1_scores.to(device)
            scores = model(p_vecs, q_vec, s0).cpu()
            sorted_local = scores.argsort(descending=True).tolist()
            pred = [sample.page_numbers[i] for i in sorted_local[:k]]
            gold = [sample.page_numbers[i] for i, v in enumerate(sample.support_mask.tolist()) if v > 0]
            all_preds.append(pred)
            all_golds.append(gold)
    return all_preds, all_golds


def _run_gat_reranker(
    eval_ds: RerankDataset,
    model: PageGATReranker,
    graph_cfg: EvidenceGraphConfig,
    device: torch.device,
    k: int,
) -> Tuple[List, List]:
    """X-PageRerank GAT reranker."""
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for sample in eval_ds:
            graph = build_evidence_graph_from_retrieval(
                page_embs=sample.page_embs,
                query_embs=sample.query_embs,
                page_numbers=sample.page_numbers,
                stage1_scores=sample.stage1_scores.tolist(),
                config=graph_cfg,
            )
            graph.to(device)
            s0 = sample.stage1_scores.to(device)
            scores = model.rerank(graph, stage1_scores=s0, device=device).cpu()
            sorted_local = scores.argsort(descending=True).tolist()
            pred = [sample.page_numbers[i] for i in sorted_local[:k]]
            gold = [sample.page_numbers[i] for i, v in enumerate(sample.support_mask.tolist()) if v > 0]
            all_preds.append(pred)
            all_golds.append(gold)
    return all_preds, all_golds


def _run_ablation_no_graph(
    eval_ds: RerankDataset,
    model: PageGATReranker,
    graph_cfg: EvidenceGraphConfig,
    device: torch.device,
    k: int,
) -> Tuple[List, List]:
    """
    Ablation: use proper node features (same as GAT) but replace the
    adjacency matrix with identity (self-loops only) — disabling all
    cross-page message passing while keeping the same scoring head.
    """
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for sample in eval_ds:
            # Build full graph to get correctly shaped node features
            graph = build_evidence_graph_from_retrieval(
                page_embs=sample.page_embs,
                query_embs=sample.query_embs,
                page_numbers=sample.page_numbers,
                stage1_scores=sample.stage1_scores.tolist(),
                config=graph_cfg,
            )
            node_feats = graph.node_features.to(device)
            N = node_feats.shape[0]
            # Identity adjacency = no cross-node message passing
            adj = torch.eye(N, device=device)
            s0 = sample.stage1_scores.to(device)
            scores = model(node_feats, adj, s0, num_pages=graph.K).cpu()
            sorted_local = scores.argsort(descending=True).tolist()
            pred = [sample.page_numbers[i] for i in sorted_local[:k]]
            gold = [sample.page_numbers[i] for i, v in enumerate(sample.support_mask.tolist()) if v > 0]
            all_preds.append(pred)
            all_golds.append(gold)
    return all_preds, all_golds


def cmd_eval(args):
    """Run all baselines + X-PageRerank and print a comparison table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("=== Phase 1 — Evaluation ===")

    eval_ds = RerankDataset.load(args.eval_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = args.k

    # --- Load models ---
    # Infer model dims from first sample
    sample0 = eval_ds[0]
    sem_threshold, adj_max_gap = _resolve_graph_hparams(args)
    logger.info(
        "Graph preset=%s -> sem_threshold=%.2f, adj_max_gap=%d",
        args.graph_preset,
        sem_threshold,
        adj_max_gap,
    )
    graph_cfg = EvidenceGraphConfig(
        sem_threshold=sem_threshold,
        adj_max_gap=adj_max_gap,
        include_query_node=True,
        inject_query_diff=True,
        concat_stage1_score=True,
    )
    g0 = build_evidence_graph_from_retrieval(
        page_embs=sample0.page_embs,
        query_embs=sample0.query_embs,
        page_numbers=sample0.page_numbers,
        stage1_scores=sample0.stage1_scores.tolist(),
        config=graph_cfg,
    )
    feat_dim = g0.feat_dim
    page_dim = sample0.page_embs.shape[-1]  # e.g. 128 for ColPali

    # MLP baseline
    mlp_model = MLPReranker(input_dim=page_dim).to(device)

    # GAT model
    gat_cfg = GATConfig(input_dim=feat_dim, hidden_dim=256, num_layers=2, num_heads=4)
    gat_model = PageGATReranker(config=gat_cfg).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        gat_model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded GAT checkpoint from %s", args.checkpoint)
    else:
        logger.warning("No checkpoint provided or not found; using random weights")

    # --- Run baselines ---
    logger.info("Running ColPali baseline …")
    t0 = time.time()
    colpali_preds, all_golds = _run_colpali_baseline(eval_ds, k)
    logger.info("  done in %.1fs", time.time() - t0)

    logger.info("Running MLP reranker baseline …")
    t0 = time.time()
    mlp_preds, _ = _run_mlp_baseline(eval_ds, mlp_model, device, k)
    logger.info("  done in %.1fs", time.time() - t0)

    logger.info("Running X-PageRerank (GAT) …")
    t0 = time.time()
    gat_preds, _ = _run_gat_reranker(eval_ds, gat_model, graph_cfg, device, k)
    logger.info("  done in %.1fs", time.time() - t0)

    logger.info("Running ablation: no-graph …")
    t0 = time.time()
    nograph_preds, _ = _run_ablation_no_graph(eval_ds, gat_model, graph_cfg, device, k)
    logger.info("  done in %.1fs", time.time() - t0)

    # --- Evaluate ---
    all_methods = {
        "ColPali (stage-1)": colpali_preds,
        "ColPali + MLP reranker": mlp_preds,
        "X-PageRerank (ablation: no graph)": nograph_preds,
        "X-PageRerank (GAT)": gat_preds,
    }

    retrieval_results: Dict[str, Dict] = {}
    support_results: Dict[str, Dict] = {}
    k_values = [1, 5, 10]

    # ------------------------------------------------------------------
    # Per-query analysis (export to results/.../phase1_results.json)
    # ------------------------------------------------------------------
    # Rationale: when ablation (no-graph) outperforms the full GAT on Recall@1,
    # we want to know *which queries* and *what support-page regimes*
    # cause the failure.
    #
    # This analysis does not affect the main metrics printed in TABLE 1/2.
    qids: List[str] = [eval_ds[i].question_id for i in range(len(eval_ds))]
    doc_ids: List[str] = [eval_ds[i].doc_id for i in range(len(eval_ds))]

    for name, preds in all_methods.items():
        retrieval_results[name] = evaluate_retrieval(
            preds, all_golds, k_values=k_values,
            method_name=name, verbose=False,
        )
        support_results[name] = evaluate_support_pages(
            preds, all_golds, k_values=k_values,
            method_name=name, verbose=False,
        )

    def per_query_first_hit(pred: List[int], gold: List[int], k: int = 10) -> Dict:
        import math

        gold_set = set(gold)
        pred_top = pred[:k]

        # Rank of the first relevant (gold support) page within top-k
        first_rank = None
        for r, p in enumerate(pred_top, start=1):
            if p in gold_set:
                first_rank = r
                break

        recall1 = int(any(p in gold_set for p in pred[:1]))
        recall5 = int(any(p in gold_set for p in pred[:5]))
        mrr10 = 0.0 if first_rank is None else 1.0 / float(first_rank)

        # nDCG@10 with binary relevance
        relevance = [1 if p in gold_set else 0 for p in pred_top]
        dcg = 0.0
        for i, rel in enumerate(relevance, start=1):
            dcg += rel / math.log2(i + 1)
        idcg = 0.0
        n_rel = min(len(gold_set), k)
        for i in range(1, n_rel + 1):
            idcg += 1 / math.log2(i + 1)
        ndcg10 = dcg / max(idcg, 1e-8)

        return {
            "num_support_pages": len(gold_set),
            "rank_first_hit_top10": first_rank,
            "Recall@1": recall1,
            "Recall@5": recall5,
            "MRR@10": mrr10,
            "nDCG@10": ndcg10,
        }

    # Collect per-query metrics for each method
    analysis: Dict[str, Dict] = {"per_query": {}, "by_support_count": {}}
    for name, preds in all_methods.items():
        pq = []
        by_count: Dict[int, List[Dict]] = {}
        for i, (pred, gold) in enumerate(zip(preds, all_golds)):
            m = per_query_first_hit(pred, gold, k=10)
            m["question_id"] = qids[i]
            m["doc_id"] = doc_ids[i]
            pq.append(m)

            by_count.setdefault(m["num_support_pages"], []).append(m)

        # Aggregate by number of support pages
        agg = {}
        for c, rows in by_count.items():
            agg[c] = {
                "n_queries": len(rows),
                "Recall@1": sum(r["Recall@1"] for r in rows) / max(len(rows), 1),
                "Recall@5": sum(r["Recall@5"] for r in rows) / max(len(rows), 1),
                "MRR@10": sum(r["MRR@10"] for r in rows) / max(len(rows), 1),
                "nDCG@10": sum(r["nDCG@10"] for r in rows) / max(len(rows), 1),
            }

        analysis["per_query"][name] = pq
        analysis["by_support_count"][name] = agg

    # Pairwise comparison: where no-graph beats GAT on Recall@1
    gat_name = "X-PageRerank (GAT)"
    nog_name = "X-PageRerank (ablation: no graph)"
    gat_preds = all_methods[gat_name]
    nog_preds = all_methods[nog_name]
    pair_rows = []
    for i, (gat_pred, nog_pred, gold) in enumerate(zip(gat_preds, nog_preds, all_golds)):
        gat_m = per_query_first_hit(gat_pred, gold, k=10)
        nog_m = per_query_first_hit(nog_pred, gold, k=10)
        pair_rows.append(
            {
                "question_id": qids[i],
                "doc_id": doc_ids[i],
                "num_support_pages": gat_m["num_support_pages"],
                "GAT Recall@1": gat_m["Recall@1"],
                "NoGraph Recall@1": nog_m["Recall@1"],
                "GAT first_hit_rank_top10": gat_m["rank_first_hit_top10"],
                "NoGraph first_hit_rank_top10": nog_m["rank_first_hit_top10"],
            }
        )
    analysis["pairwise_no_graph_vs_gat"] = {
        "n_queries": len(pair_rows),
        "n_no_graph_better_recall1": sum(
            1 for r in pair_rows if r["NoGraph Recall@1"] > r["GAT Recall@1"]
        ),
        "n_equal_recall1": sum(1 for r in pair_rows if r["NoGraph Recall@1"] == r["GAT Recall@1"]),
        "per_query": pair_rows,
    }

    # --- Print tables ---
    print("\n" + "=" * 70)
    print("TABLE 1 — Main Retrieval Results  (MP-DocVQA)")
    print("=" * 70)
    print_comparison_table(
        retrieval_results,
        metrics=["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"],
    )

    print("\n" + "=" * 70)
    print("TABLE 2 — Cross-Page Evidence Metrics  (MP-DocVQA)")
    print("=" * 70)
    print_comparison_table(
        support_results,
        metrics=["Coverage@5", "Coverage@10", "AllSupportHit@10", "SupportF1@10"],
    )

    # --- Save JSON results ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "retrieval": retrieval_results,
            "support_pages": support_results,
            "analysis": analysis,
        }
        out_path = os.path.join(args.output_dir, "phase1_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Results saved to %s", out_path)


# ===========================================================================
# Argument parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_phase1",
        description="X-PageRerank Phase 1 experiment pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- mock --
    p_mock = sub.add_parser(
        "mock",
        help="Generate synthetic cache (no HF download) for smoke-testing the pipeline",
    )
    p_mock.add_argument("--output_dir", default="cache/mock",
                        help="Root directory; train/ and val/ will be created inside")
    p_mock.add_argument("--num_train", type=int, default=200)
    p_mock.add_argument("--num_val", type=int, default=50)
    p_mock.add_argument("--top_k", type=int, default=10)
    p_mock.add_argument("--embedding_dim", type=int, default=128,
                        help="ColPali embedding dim (128 by default)")
    p_mock.add_argument("--seed", type=int, default=42)

    # -- cache --
    p_cache = sub.add_parser("cache", help="Encode + cache top-K embeddings")
    p_cache.add_argument("--split", default="train",
                         choices=["train", "val", "validation", "test"],
                         help="Dataset split. 'val' and 'validation' are treated as equivalent.")
    p_cache.add_argument("--model_name", default="vidore/colpali-v1.2")
    p_cache.add_argument("--model_type", default="colpali",
                         choices=["colpali", "colqwen2", "colqwen2_5", "colidefics3"])
    p_cache.add_argument("--output_dir", required=True)
    p_cache.add_argument("--top_k", type=int, default=20)
    p_cache.add_argument("--max_pages", type=int, default=60)
    p_cache.add_argument("--batch_size", type=int, default=4)
    p_cache.add_argument(
        "--maxsim_doc_chunk",
        type=int,
        default=16,
        help=(
            "When scoring query vs all pages (MaxSim), process this many pages at a time. "
            "Lowers RAM peak for long documents / high token counts; try 4–8 if cache still crashes."
        ),
    )
    # Dataset source options
    p_cache.add_argument("--hf_dataset_id", default=None,
                         help="HuggingFace dataset ID (default: lmms-lab/MP-DocVQA)")
    p_cache.add_argument("--hf_token", default=None,
                         help="HuggingFace token for gated datasets "
                              "(or set env var HUGGING_FACE_HUB_TOKEN)")
    p_cache.add_argument("--local_json_path", default=None,
                         help="Path to local JSON file instead of downloading from HF")
    p_cache.add_argument("--streaming", action="store_true",
                         help="Use HF streaming mode (no full download)")
    p_cache.add_argument("--num_samples", type=int, default=None,
                         help="Limit number of samples to process")
    p_cache.add_argument("--sample_offset", type=int, default=0,
                         help="Skip first N samples (use with --num_samples for non-overlapping splits). "
                              "E.g. train: --num_samples 800, val: --sample_offset 800 --num_samples 200")
    p_cache.add_argument("--parquet_dir", default=None,
                         help="Load directly from local parquet files (e.g. HF hub cache). "
                              "Example: ~/.cache/huggingface/hub/datasets--lmms-lab--MP-DocVQA/"
                              "snapshots/<hash>/data/")
    p_cache.add_argument("--offline", action="store_true",
                         help="Force offline mode (HF_DATASETS_OFFLINE=1 + TRANSFORMERS_OFFLINE=1). "
                              "Use when network is unavailable and data is already cached.")
    p_cache.add_argument(
        "--exclude_train_doc_ids_cache",
        default=None,
        help=(
            "Path to an existing train RerankDataset cache (meta.json). "
            "Before encoding, drop rows whose doc_id appears in that train cache so "
            "this split only contains documents not present in train. "
            "Use a larger --num_samples window after filtering so enough rows remain."
        ),
    )
    p_cache.add_argument(
        "--cache_chunk_rows",
        type=int,
        default=512,
        help=(
            "Process the dataset in row chunks of this size when building the cache "
            "(lower peak RAM). MP-DocVQA loads every page image per row into memory; "
            "large --num_samples without chunking can OOM or be killed by the OS. "
            "Use 0 for the legacy single-load path (all rows in RAM at once)."
        ),
    )

    # -- train --
    p_train = sub.add_parser("train", help="Train GAT reranker on cached data")
    p_train.add_argument("--train_cache", required=True)
    p_train.add_argument("--val_cache", default=None)
    p_train.add_argument("--output_dir", default="checkpoints/gat_phase1")
    p_train.add_argument("--num_epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--num_layers", type=int, default=2)
    p_train.add_argument("--num_heads", type=int, default=4)
    p_train.add_argument("--top_k", type=int, default=20)
    p_train.add_argument("--sem_threshold", type=float, default=0.65)
    p_train.add_argument("--adj_max_gap", type=int, default=1)
    p_train.add_argument(
        "--graph_preset",
        default="default",
        choices=["default", "sparse-graph"],
        help=(
            "Graph construction preset. 'sparse-graph' increases semantic threshold "
            "and keeps only immediate adjacency links to reduce noisy edges."
        ),
    )

    # -- eval --
    p_eval = sub.add_parser("eval", help="Evaluate all methods and print comparison table")
    p_eval.add_argument("--eval_cache", required=True)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--output_dir", default="results/phase1")
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.add_argument("--sem_threshold", type=float, default=0.65)
    p_eval.add_argument("--adj_max_gap", type=int, default=1)
    p_eval.add_argument(
        "--graph_preset",
        default="default",
        choices=["default", "sparse-graph"],
        help=(
            "Graph construction preset. 'sparse-graph' increases semantic threshold "
            "and keeps only immediate adjacency links to reduce noisy edges."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "mock": cmd_mock,
        "cache": cmd_cache,
        "train": cmd_train,
        "eval": cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

"""
Training pipeline for the X-PageRerank GAT reranker.

Training objective:
  Given a query q and top-K pages retrieved by ColPali, train the GAT to
  assign higher scores to support pages (gold evidence) than non-support pages.

Loss:
  Primary: listwise softmax cross-entropy over K pages
           (support pages form a positive set; all others are negatives)
  Auxiliary: pairwise margin loss between best positive and hardest negative

Training data:
  MP-DocVQA (and optionally DUDE) with ColPali-retrieved top-K pages
  cached as pre-computed embeddings.

Usage
-----
    python -m experiments.train.train_reranker \
        --index_dir  cache/mp_docvqa_index \
        --dataset_dir data/mp_docvqa \
        --output_dir  checkpoints/gat_reranker \
        --num_epochs  10  --top_k 20
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..graph.build_query_graph import EvidenceGraph, EvidenceGraphConfig, build_evidence_graph_from_retrieval
from ..models.page_gat_reranker import GATConfig, PageGATReranker
from ..eval.eval_retrieval import recall_at_k, mrr_at_k, ndcg_at_k

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Data
    top_k: int = 20                   # number of retrieved pages per query
    num_hard_negatives: int = 3       # forced negatives from bottom of stage-1 ranking

    # Optimiser
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 10
    batch_size: int = 32              # number of queries per batch
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Loss weights
    listwise_weight: float = 0.7
    pairwise_weight: float = 0.3
    margin: float = 0.1

    # Validation
    eval_every_n_epochs: int = 1
    eval_k: int = 10

    # Lambda mix schedule (for models with lambda_mix parameter)
    lambda_mix_start: float = 0.20
    lambda_mix_end: float = 0.60
    lambda_mix_warmup_steps: int = 800

    # Graph config (passed to EvidenceGraphConfig)
    sem_threshold: float = 0.65
    adj_max_gap: int = 1
    include_query_node: bool = True

    # Checkpoint
    output_dir: str = "checkpoints/gat_reranker"
    save_best: bool = True
    patience: int = 5               # early stopping on val Recall@5


# ---------------------------------------------------------------------------
# Pre-built training sample
# ---------------------------------------------------------------------------

@dataclass
class RerankSample:
    """
    One training sample: (query, top-K pages from ColPali, support labels).

    Stored as tensors for fast loading without re-running ColPali.
    """
    question_id: str
    query_embs: torch.Tensor          # (1, T, D) multi-vector query embedding
    page_embs: torch.Tensor           # (K, T, D) multi-vector page embeddings
    page_numbers: List[int]           # absolute page positions
    stage1_scores: torch.Tensor       # (K,) ColPali MaxSim scores
    support_mask: torch.Tensor        # (K,) bool: 1 if page supports answer
    doc_id: str


class RerankDataset(Dataset):
    """
    Dataset of pre-computed (query_emb, top-K page_embs, support_mask) tuples.

    Can be built from scratch via `RerankDataset.build_from_index()`
    or loaded from a pre-saved directory via `RerankDataset.load()`.
    """

    def __init__(self, samples: List[RerankSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RerankSample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # Build from ColPali index + dataset
    # ------------------------------------------------------------------

    @classmethod
    def build_from_index(
        cls,
        dataset,           # MPDocVQADataset or DUDEDataset
        index,             # PageIndex (loaded, with self._index_embs built)
        inferencer,        # ColVisionInferencer
        top_k: int = 20,
        num_hard_negatives: int = 3,
        show_progress: bool = True,
    ) -> "RerankDataset":
        """
        Build training samples by running ColPali retrieval over the dataset.

        For each question:
        1. Encode query → query_embs
        2. Retrieve top-K from the page index (restricted to the document)
        3. Build support_mask from ground-truth support page indices
        4. Optionally inject hard negatives from the bottom of stage-1 ranking

        Returns a RerankDataset.
        """
        samples = []
        pbar = tqdm(dataset, desc="Building RerankDataset", disable=not show_progress)

        for sample in pbar:
            pages = sample.images if hasattr(sample, "images") else []
            if not pages:
                continue

            support_set = set(sample.support_page_idxs)

            # Encode query
            q_embs = inferencer.encode_queries([sample.question], desc="")  # (1, T, D)

            # Encode pages (for this document)
            p_embs = inferencer.encode_pages(pages, desc="")               # (N_doc, T, D)
            N_doc = p_embs.shape[0]

            # Score: (1, N_doc)
            scores = inferencer.score(q_embs, p_embs).squeeze(0)           # (N_doc,)

            # Sort by stage-1 score descending, take top-K
            sorted_idx = scores.argsort(descending=True)
            topk_idx = sorted_idx[:top_k]

            # Pad with hard negatives from bottom if needed
            if num_hard_negatives > 0 and len(topk_idx) < top_k:
                bottom_idx = sorted_idx[-(num_hard_negatives):]
                combined = torch.cat([topk_idx, bottom_idx]).unique()
                topk_idx = combined[:top_k]

            K = topk_idx.shape[0]
            page_numbers = topk_idx.tolist()
            stage1_scores = scores[topk_idx]             # (K,)
            sel_page_embs = p_embs[topk_idx]             # (K, T, D)
            support_mask = torch.tensor(
                [int(idx.item() in support_set) for idx in topk_idx],
                dtype=torch.float,
            )

            # Skip samples where none of the top-K pages is a support page
            if support_mask.sum() == 0:
                continue

            samples.append(RerankSample(
                question_id=sample.question_id,
                query_embs=q_embs,
                page_embs=sel_page_embs,
                page_numbers=page_numbers,
                stage1_scores=stage1_scores,
                support_mask=support_mask,
                doc_id=sample.doc_id if hasattr(sample, "doc_id") else "",
            ))
            # Free decoded PIL pages before the next sample (large RAM for multi-page docs).
            if hasattr(sample, "images") and sample.images is not None:
                sample.images.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Built RerankDataset: %d samples", len(samples))
        return cls(samples)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        meta = []
        for i, s in enumerate(self.samples):
            torch.save(s.query_embs, os.path.join(directory, f"{i}_q.pt"))
            torch.save(s.page_embs, os.path.join(directory, f"{i}_p.pt"))
            torch.save(s.stage1_scores, os.path.join(directory, f"{i}_s0.pt"))
            torch.save(s.support_mask, os.path.join(directory, f"{i}_mask.pt"))
            meta.append({
                "question_id": s.question_id,
                "doc_id": s.doc_id,
                "page_numbers": s.page_numbers,
            })
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)
        logger.info("Saved RerankDataset (%d samples) to %s", len(self.samples), directory)

    @classmethod
    def load(cls, directory: str) -> "RerankDataset":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        samples = []
        for i, m in enumerate(meta):
            samples.append(RerankSample(
                question_id=m["question_id"],
                doc_id=m["doc_id"],
                page_numbers=m["page_numbers"],
                query_embs=torch.load(os.path.join(directory, f"{i}_q.pt"), map_location="cpu"),
                page_embs=torch.load(os.path.join(directory, f"{i}_p.pt"), map_location="cpu"),
                stage1_scores=torch.load(os.path.join(directory, f"{i}_s0.pt"), map_location="cpu"),
                support_mask=torch.load(os.path.join(directory, f"{i}_mask.pt"), map_location="cpu"),
            ))
        logger.info("Loaded RerankDataset: %d samples from %s", len(samples), directory)
        return cls(samples)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_rerank(batch: List[RerankSample]) -> List[RerankSample]:
    """Keep samples as a list (variable-length K per sample)."""
    return batch


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def listwise_loss(
    scores: torch.Tensor,      # (K,)
    support_mask: torch.Tensor,  # (K,) float, 1 = positive
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Softmax cross-entropy treating all support pages as the positive class.
    Averages over all support pages in the sample.
    """
    log_probs = F.log_softmax(scores / temperature, dim=0)   # (K,)
    n_pos = support_mask.sum().clamp(min=1)
    loss = -(log_probs * support_mask).sum() / n_pos
    return loss


def pairwise_margin_loss(
    scores: torch.Tensor,        # (K,)
    support_mask: torch.Tensor,  # (K,) float
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Pairwise margin loss: for each (positive, negative) pair,
    penalise if pos_score < neg_score + margin.
    """
    pos_scores = scores[support_mask.bool()]       # (P,)
    neg_scores = scores[(~support_mask.bool())]    # (N,)

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device)

    # All pairs
    p_exp = pos_scores.unsqueeze(1)   # (P, 1)
    n_exp = neg_scores.unsqueeze(0)   # (1, N)
    loss = F.relu(margin + n_exp - p_exp).mean()
    return loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RerankerTrainer:
    """
    Trains a PageGATReranker on pre-built RerankDatasets.

    Supports:
    - listwise + pairwise combined loss
    - CosineAnnealingLR with linear warmup
    - Best-model checkpoint saving
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        graph_config: Optional[EvidenceGraphConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.graph_config = graph_config or EvidenceGraphConfig(
            sem_threshold=config.sem_threshold,
            adj_max_gap=config.adj_max_gap,
            include_query_node=config.include_query_node,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0

    @staticmethod
    def _safe_logit(p: float) -> float:
        p = min(max(float(p), 1e-4), 1 - 1e-4)
        return float(torch.log(torch.tensor(p / (1 - p))))

    def _maybe_update_lambda_mix(self) -> None:
        if not hasattr(self.model, "lambda_mix"):
            return
        warm = max(int(self.config.lambda_mix_warmup_steps), 1)
        t = min(1.0, float(self.global_step) / float(warm))
        target = self.config.lambda_mix_start + t * (self.config.lambda_mix_end - self.config.lambda_mix_start)
        with torch.no_grad():
            self.model.lambda_mix.data.fill_(self._safe_logit(target))

    def _build_graph(self, sample: RerankSample) -> EvidenceGraph:
        """Build an EvidenceGraph for one training sample."""
        return build_evidence_graph_from_retrieval(
            page_embs=sample.page_embs,
            query_embs=sample.query_embs,
            page_numbers=sample.page_numbers,
            stage1_scores=sample.stage1_scores.tolist(),
            config=self.graph_config,
        )

    def _forward_one(self, sample: RerankSample) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one forward pass. Returns (pred_scores (K,), support_mask (K,))."""
        s0 = sample.stage1_scores.to(self.device)

        if hasattr(self.model, "rerank_from_multivector"):
            scores = self.model.rerank_from_multivector(
                page_embs=sample.page_embs.to(self.device),
                query_embs=sample.query_embs.to(self.device),
                page_numbers=sample.page_numbers,
                stage1_scores=s0,
            )
        else:
            graph = self._build_graph(sample)
            graph.to(self.device)
            scores = self.model.rerank(graph, stage1_scores=s0, device=self.device)

        support = sample.support_mask.to(self.device)
        return scores, support

    # ------------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            batch_loss = torch.tensor(0.0, device=self.device)
            self._maybe_update_lambda_mix()

            for sample in batch:
                scores, support = self._forward_one(sample)
                lw = listwise_loss(scores, support)
                pw = pairwise_margin_loss(scores, support, margin=self.config.margin)
                cfg = self.config
                loss = cfg.listwise_weight * lw + cfg.pairwise_weight * pw
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(batch)
            self.optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            scheduler.step()
            self.global_step += 1

            total_loss += batch_loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, k: int = 10) -> Dict:
        self.model.eval()
        all_preds: List[List[int]] = []    # predicted page_numbers at top-k
        all_golds: List[List[int]] = []    # gold support page_numbers

        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            for sample in batch:
                scores, support = self._forward_one(sample)
                sorted_local = scores.argsort(descending=True).cpu().tolist()
                pred_pages = [sample.page_numbers[i] for i in sorted_local[:k]]
                gold_pages = [sample.page_numbers[i]
                              for i, v in enumerate(sample.support_mask.tolist()) if v > 0]
                all_preds.append(pred_pages)
                all_golds.append(gold_pages)

        metrics = {
            f"Recall@{k}": recall_at_k(all_preds, all_golds, k),
            f"Recall@5": recall_at_k(all_preds, all_golds, 5),
            f"Recall@1": recall_at_k(all_preds, all_golds, 1),
            "MRR@10": mrr_at_k(all_preds, all_golds, 10),
            "nDCG@10": ndcg_at_k(all_preds, all_golds, 10),
        }
        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: RerankDataset,
        val_dataset: Optional[RerankDataset] = None,
    ) -> Dict:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_rerank,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_rerank,
        ) if val_dataset is not None else None

        total_steps = len(train_loader) * cfg.num_epochs
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)

        history = []
        for epoch in range(1, cfg.num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch, scheduler)
            row = {"epoch": epoch, "train_loss": round(train_loss, 5)}

            if val_loader is not None and epoch % cfg.eval_every_n_epochs == 0:
                metrics = self.evaluate(val_loader, k=cfg.eval_k)
                row.update(metrics)
                val_key = f"Recall@5"
                val_metric = metrics.get(val_key, 0.0)
                logger.info("Epoch %d | loss=%.4f | %s", epoch, train_loss,
                            " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.patience_counter = 0
                    if cfg.save_best:
                        self._save_checkpoint(os.path.join(cfg.output_dir, "best.pt"))
                        logger.info("  → Saved new best (Recall@5=%.4f)", val_metric)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= cfg.patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break
            else:
                logger.info("Epoch %d | train_loss=%.4f", epoch, train_loss)

            history.append(row)

        # Save final checkpoint
        self._save_checkpoint(os.path.join(cfg.output_dir, "last.pt"))
        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        return {"history": history, "best_recall@5": self.best_metric}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "best_metric": self.best_metric,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_metric = ckpt.get("best_metric", 0.0)
        logger.info("Loaded checkpoint from %s (best=%.4f)", path, self.best_metric)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train X-PageRerank GAT reranker")
    parser.add_argument("--train_cache_dir", required=True, help="RerankDataset train cache directory")
    parser.add_argument("--val_cache_dir", required=False, help="RerankDataset val cache directory")
    parser.add_argument("--output_dir", default="checkpoints/gat_reranker")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--input_dim", type=int, default=257)  # 128*2 + 1
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = RerankDataset.load(args.train_cache_dir)
    val_dataset = RerankDataset.load(args.val_cache_dir) if args.val_cache_dir else None

    # Infer input_dim from first sample's node features
    graph_cfg = EvidenceGraphConfig()
    sample0 = train_dataset[0]
    g0 = build_evidence_graph_from_retrieval(
        page_embs=sample0.page_embs,
        query_embs=sample0.query_embs,
        page_numbers=sample0.page_numbers,
        stage1_scores=sample0.stage1_scores.tolist(),
        config=graph_cfg,
    )
    input_dim = g0.feat_dim

    model_cfg = GATConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = PageGATReranker(config=model_cfg)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    train_cfg = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    trainer = RerankerTrainer(model=model, config=train_cfg, graph_config=graph_cfg)
    results = trainer.train(train_dataset, val_dataset)

    print("\n=== Training complete ===")
    print(f"Best Recall@5: {results['best_recall@5']:.4f}")


if __name__ == "__main__":
    main()

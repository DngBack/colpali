"""
EvidenceGraph — builds a query-conditioned graph over top-k retrieved pages.

Usage
-----
>>> config = EvidenceGraphConfig(sem_threshold=0.65, adj_max_gap=1)
>>> graph = EvidenceGraph(
...     page_vectors=pooled_embs,   # (K, D)
...     query_vector=q_vec,          # (D,)
...     page_numbers=[3, 7, 8, 12],
...     stage1_scores=[0.9, 0.8, 0.75, 0.6],
...     config=config,
... )
>>> graph.build()
>>> node_feats = graph.node_features     # (K+1, D)  pages + query node
>>> adj = graph.adj_matrix               # (K+1, K+1)
>>> edge_index = graph.edge_index        # (2, E)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .edges import (
    build_full_edge_index,
    compute_query_edges,
    edge_index_to_adj,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvidenceGraphConfig:
    """Hyperparameters for building the evidence graph."""

    # Semantic edges
    sem_threshold: float = 0.65
    sem_top_k: int = 5            # max neighbours per node

    # Adjacency edges
    adj_max_gap: int = 1          # connect pages within this page-gap

    # Graph options
    include_query_node: bool = True
    use_structural_edges: bool = False

    # Adjacency matrix
    add_self_loops: bool = True
    row_normalize: bool = True

    # Node feature options
    concat_stage1_score: bool = True  # append s0 as an extra feature dimension
    inject_query_diff: bool = True    # append (page_vec - query_vec) diff features


# ---------------------------------------------------------------------------
# EvidenceGraph
# ---------------------------------------------------------------------------

class EvidenceGraph:
    """
    Query-conditioned evidence graph over top-k retrieved pages.

    Node layout:
        0 … K-1   : page nodes (in order of stage-1 retrieval rank)
        K          : query node  (only if config.include_query_node)

    Edge types:
        semantic     : cosine similarity between page vectors
        adjacency    : consecutive page numbers
        query→pages  : query node to all page nodes (bidirectional)
        structural   : figure-caption / table-paragraph (optional)
    """

    def __init__(
        self,
        page_vectors: torch.Tensor,       # (K, D) pooled, L2-normalised
        query_vector: torch.Tensor,       # (D,)
        page_numbers: List[int],          # absolute page positions in the document
        stage1_scores: Optional[List[float]] = None,
        doc_ids: Optional[List[str]] = None,
        layout_info: Optional[List[Dict]] = None,
        config: Optional[EvidenceGraphConfig] = None,
    ):
        self.config = config or EvidenceGraphConfig()
        self.K = page_vectors.shape[0]
        self.D = page_vectors.shape[1]

        self.page_vectors = F.normalize(page_vectors.float(), dim=-1)   # (K, D)
        self.query_vector = F.normalize(query_vector.float(), dim=-1)   # (D,)

        self.page_numbers = page_numbers
        self.stage1_scores = stage1_scores or [1.0] * self.K
        self.doc_ids = doc_ids
        self.layout_info = layout_info

        self._built = False
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None
        self._adj: Optional[torch.Tensor] = None
        self._node_feats: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> "EvidenceGraph":
        """Construct all graph components. Idempotent."""
        if self._built:
            return self
        self._build_edges()
        self._build_node_features()
        self._built = True
        return self

    def _build_edges(self) -> None:
        cfg = self.config
        num_page_nodes = self.K
        query_node_idx = self.K if cfg.include_query_node else None

        self._edge_index, self._edge_weight = build_full_edge_index(
            page_vectors=self.page_vectors,
            page_numbers=self.page_numbers,
            query_node_idx=query_node_idx,
            doc_ids=self.doc_ids,
            layout_info=self.layout_info,
            sem_threshold=cfg.sem_threshold,
            sem_top_k=cfg.sem_top_k,
            adj_max_gap=cfg.adj_max_gap,
            use_structural=cfg.use_structural_edges,
        )

        total_nodes = self.K + (1 if cfg.include_query_node else 0)
        self._adj = edge_index_to_adj(
            self._edge_index,
            self._edge_weight,
            num_nodes=total_nodes,
            add_self_loops=cfg.add_self_loops,
            row_normalize=cfg.row_normalize,
        )

    def _build_node_features(self) -> None:
        cfg = self.config
        # Base features: page vectors
        feats = self.page_vectors.clone()   # (K, D)

        # Optionally append (page_vec - query_vec) as diff signal
        if cfg.inject_query_diff:
            diff = feats - self.query_vector.unsqueeze(0)   # (K, D)
            feats = torch.cat([feats, diff], dim=-1)        # (K, 2D)

        # Optionally append stage-1 score as scalar feature
        if cfg.concat_stage1_score:
            s0 = torch.tensor(self.stage1_scores, dtype=torch.float).unsqueeze(-1)  # (K, 1)
            # Normalise to [0, 1]
            s_min, s_max = s0.min(), s0.max()
            s0 = (s0 - s_min) / (s_max - s_min + 1e-8)
            feats = torch.cat([feats, s0], dim=-1)          # (K, 2D+1) or (K, D+1)

        # Query node features: query vector (padded to match page feat dim)
        if cfg.include_query_node:
            q_feat = self.query_vector.clone()               # (D,)
            pad_size = feats.shape[1] - self.D
            if pad_size > 0:
                q_feat = torch.cat([q_feat, torch.zeros(pad_size)], dim=-1)
            feats = torch.cat([feats, q_feat.unsqueeze(0)], dim=0)  # (K+1, feat_dim)

        self._node_feats = feats

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.K + (1 if self.config.include_query_node else 0)

    @property
    def node_features(self) -> torch.Tensor:
        """(num_nodes, feat_dim)"""
        if not self._built:
            self.build()
        return self._node_feats

    @property
    def adj_matrix(self) -> torch.Tensor:
        """(num_nodes, num_nodes)"""
        if not self._built:
            self.build()
        return self._adj

    @property
    def edge_index(self) -> torch.Tensor:
        """(2, E)"""
        if not self._built:
            self.build()
        return self._edge_index

    @property
    def edge_weight(self) -> torch.Tensor:
        """(E,)"""
        if not self._built:
            self.build()
        return self._edge_weight

    @property
    def feat_dim(self) -> int:
        return self.node_features.shape[1]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "EvidenceGraph":
        """Move all tensors to device."""
        if self._node_feats is not None:
            self._node_feats = self._node_feats.to(device)
        if self._adj is not None:
            self._adj = self._adj.to(device)
        if self._edge_index is not None:
            self._edge_index = self._edge_index.to(device)
        if self._edge_weight is not None:
            self._edge_weight = self._edge_weight.to(device)
        return self

    def summary(self) -> Dict:
        if not self._built:
            self.build()
        num_edges = self._edge_index.shape[1] if self._edge_index is not None else 0
        return {
            "num_page_nodes": self.K,
            "num_total_nodes": self.num_nodes,
            "num_edges": num_edges,
            "feat_dim": self.feat_dim,
            "has_query_node": self.config.include_query_node,
            "has_structural_edges": self.config.use_structural_edges,
        }


# ---------------------------------------------------------------------------
# Factory: build a graph from an inferencer's output
# ---------------------------------------------------------------------------

def build_evidence_graph_from_retrieval(
    page_embs: torch.Tensor,       # (K, T, D) multi-vector
    query_embs: torch.Tensor,      # (1, T, D) multi-vector
    page_numbers: List[int],
    stage1_scores: List[float],
    doc_ids: Optional[List[str]] = None,
    layout_info: Optional[List[Dict]] = None,
    config: Optional[EvidenceGraphConfig] = None,
) -> EvidenceGraph:
    """
    Convenience function: pools multi-vector embeddings then builds the graph.

    Args:
        page_embs:      (K, T, D)  multi-vector page embeddings
        query_embs:     (1, T, D)  multi-vector query embeddings
        page_numbers:   absolute page positions
        stage1_scores:  ColPali scores from stage-1 retrieval
        doc_ids:        document id per page (optional)
        layout_info:    layout dicts per page (optional)
        config:         graph config

    Returns:
        Built EvidenceGraph
    """
    # Pool multi-vectors to single vectors
    page_vectors = page_embs.float().mean(dim=1)         # (K, D)
    query_vector = query_embs.float().squeeze(0).mean(dim=0)  # (D,)

    graph = EvidenceGraph(
        page_vectors=page_vectors,
        query_vector=query_vector,
        page_numbers=page_numbers,
        stage1_scores=stage1_scores,
        doc_ids=doc_ids,
        layout_info=layout_info,
        config=config,
    )
    graph.build()
    return graph

"""
PageGATReranker — graph attention network reranker operating at page level.

Architecture:
  1. Linear input projection: feat_dim → hidden_dim
  2. L stacked GATLayer modules (multi-head attention over adjacency)
  3. Query-conditioned scoring head: outputs a delta score per page node
  4. Final score: s(q, p_i) = s0(q, p_i) + lambda * delta_i

This is a pure-PyTorch implementation with no dependency on PyG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GATConfig:
    """Hyperparameters for PageGATReranker."""
    input_dim: int = 128       # ColPali embedding dim (auto-adjusted if diff/score appended)
    hidden_dim: int = 256
    output_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    alpha: float = 0.2         # LeakyReLU negative slope
    residual: bool = True
    lambda_mix: float = 0.5    # weight for graph score in final interpolation
    score_only_pages: bool = True  # ignore query node when outputting scores


# ---------------------------------------------------------------------------
# Single GAT layer (multi-head, dense adjacency)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    One layer of Graph Attention Network (Veličković et al., 2018).

    Operates on a dense adjacency matrix (suitable for small K ≤ ~100 nodes).
    Uses multi-head attention; outputs are concatenated (not averaged) except
    for the last layer where they are averaged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,     # True for intermediate layers
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        # Shared linear transformation per head
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        # Attention vector: [Wh_i || Wh_j] → scalar
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features))

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.drop = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a.unsqueeze(0), gain=1.414)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (N, in_features)
            adj: (N, N) adjacency (row-normalised or raw, already masked to 0 for non-edges)

        Returns:
            out: (N, num_heads * out_features)  if concat
                 (N, out_features)              if not concat
        """
        N = x.shape[0]

        # x: (N, F_in) → Wh: (heads, N, F_out)
        Wh = torch.einsum("ni,hio->hno", x, self.W)   # (H, N, F_out)

        # Attention coefficients
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Wh_i: (H, N, 1, F), Wh_j: (H, 1, N, F) → concat → (H, N, N, 2F)
        Wh_i = Wh.unsqueeze(2)        # (H, N, 1, F)
        Wh_j = Wh.unsqueeze(1)        # (H, 1, N, F)
        cat_ij = torch.cat([Wh_i.expand(-1, N, N, -1), Wh_j.expand(-1, N, N, -1)], dim=-1)  # (H, N, N, 2F)

        # a: (H, 2F) → (H, 1, 1, 2F)
        e = self.leaky_relu((cat_ij * self.a[:, None, None, :]).sum(dim=-1))  # (H, N, N)

        # Mask non-edges with -inf
        mask = (adj == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        e = e.masked_fill(mask, float("-inf"))

        # Normalise
        attn = F.softmax(e, dim=-1)                  # (H, N, N)
        attn = attn.nan_to_num(nan=0.0)              # handle isolated nodes
        attn = self.drop(attn)

        # Aggregate: (H, N, N) x (H, N, F) → (H, N, F)
        out = torch.bmm(attn, Wh)                    # (H, N, F_out)

        if self.concat:
            out = out.permute(1, 0, 2).reshape(N, -1)   # (N, H*F_out)
        else:
            out = out.mean(dim=0)                        # (N, F_out)

        return F.elu(out)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class PageGATReranker(nn.Module):
    """
    Query-conditioned page reranker using stacked GAT layers.

    Forward:
        node_features: (K+1, feat_dim)   — pages + optional query node
        adj_matrix:    (K+1, K+1)
        stage1_scores: (K,) or None       — original ColPali scores
        num_pages:     K  (to separate page nodes from query node)

    Output:
        final_scores: (K,) after combining stage-1 + graph delta
    """

    def __init__(self, config: Optional[GATConfig] = None):
        super().__init__()
        self.config = config or GATConfig()
        cfg = self.config

        # ---- Input projection ----
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # ---- GAT layers ----
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(cfg.num_layers):
            is_last = (layer_idx == cfg.num_layers - 1)
            in_f = cfg.hidden_dim * cfg.num_heads if layer_idx > 0 else cfg.hidden_dim
            out_f = cfg.output_dim if is_last else cfg.hidden_dim
            self.gat_layers.append(
                GATLayer(
                    in_features=in_f,
                    out_features=out_f,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    alpha=cfg.alpha,
                    concat=not is_last,   # last layer averages heads
                )
            )

        # ---- Residual projection ----
        # Maps input_proj output → final GAT output dim for skip connection
        self.residual_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim) if cfg.residual else None

        # ---- Scoring head ----
        # Takes [page_repr || query_repr] and produces a scalar delta
        self.score_head = nn.Sequential(
            nn.Linear(cfg.output_dim * 2, cfg.output_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.output_dim, 1),
        )

        self.lambda_mix = nn.Parameter(torch.tensor(cfg.lambda_mix))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        node_features: torch.Tensor,   # (N, feat_dim)
        adj_matrix: torch.Tensor,      # (N, N)
        stage1_scores: Optional[torch.Tensor] = None,  # (K,) ColPali scores
        num_pages: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns final page scores (K,).
        """
        N = node_features.shape[0]
        K = num_pages if num_pages is not None else N

        # --- Input projection ---
        h = self.input_proj(node_features)     # (N, hidden_dim)
        h_res = h.clone()                       # for residual

        # --- GAT message passing ---
        for gat in self.gat_layers:
            h = self.drop(gat(h, adj_matrix))

        # --- Residual ---
        if self.residual_proj is not None:
            h = h + self.residual_proj(h_res)

        # --- Separate page nodes and query node ---
        page_repr = h[:K]       # (K, output_dim)
        query_repr = h[K] if N > K else h.mean(dim=0)   # (output_dim,)

        # --- Query-conditioned scoring ---
        q_expanded = query_repr.unsqueeze(0).expand(K, -1)  # (K, output_dim)
        pair = torch.cat([page_repr, q_expanded], dim=-1)   # (K, 2*output_dim)
        delta = self.score_head(pair).squeeze(-1)            # (K,)

        # --- Interpolate with stage-1 scores ---
        if stage1_scores is not None:
            # Normalise stage-1 scores to [0, 1]
            s0 = stage1_scores.float()
            s0_min, s0_max = s0.min(), s0.max()
            s0_norm = (s0 - s0_min) / (s0_max - s0_min + 1e-8)
            lam = torch.sigmoid(self.lambda_mix)
            final = (1 - lam) * s0_norm + lam * delta
        else:
            final = delta

        return final   # (K,)

    # ------------------------------------------------------------------
    # Convenience wrapper for a single evidence graph
    # ------------------------------------------------------------------

    def rerank(
        self,
        graph,   # EvidenceGraph
        stage1_scores: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Run reranking on an EvidenceGraph object.

        Args:
            graph:         EvidenceGraph (already built)
            stage1_scores: (K,) tensor of ColPali scores (optional)
            device:        target device

        Returns:
            scores: (K,) reranked page scores
        """
        if device is None:
            device = next(self.parameters()).device

        node_feats = graph.node_features.to(device)
        adj = graph.adj_matrix.to(device)
        if stage1_scores is not None:
            stage1_scores = stage1_scores.to(device)

        return self.forward(node_feats, adj, stage1_scores, num_pages=graph.K)


# ---------------------------------------------------------------------------
# MLP reranker baseline (no graph)
# ---------------------------------------------------------------------------

class MLPReranker(nn.Module):
    """
    Non-graph MLP reranker baseline.
    Input: concatenation of [page_vector, query_vector, stage1_score]
    Output: scalar reranking score per page.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        page_vectors: torch.Tensor,    # (K, D)
        query_vector: torch.Tensor,    # (D,)
        stage1_scores: torch.Tensor,   # (K,)
    ) -> torch.Tensor:
        """Returns (K,) scores."""
        K = page_vectors.shape[0]
        q_exp = query_vector.unsqueeze(0).expand(K, -1)    # (K, D)
        s0 = stage1_scores.unsqueeze(-1)                   # (K, 1)
        x = torch.cat([page_vectors, q_exp, s0], dim=-1)  # (K, 2D+1)
        return self.net(x).squeeze(-1)                     # (K,)

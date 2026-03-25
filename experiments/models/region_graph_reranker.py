"""
RegionGraphReranker — Phase-2 extension of PageGATReranker.

Adds region-level nodes on top of page nodes:
  - Each page is split into a fixed grid of sub-regions (e.g., 2×2 or 3×3).
  - Region embeddings are extracted by spatial pooling of the multi-vector
    ColPali output (which is already spatially aligned to PaliGemma patches).
  - Both page nodes and region nodes participate in message passing.

Node layout:
  0 … K-1          : page nodes
  K … K+K*R-1      : region nodes  (R regions per page, ordered page-major)
  K+K*R             : query node

Edge types added on top of page-level edges:
  - Page → Region:   page node to its own regions
  - Region → Page:   (reciprocal)
  - Region → Region: within same page (spatial neighbours)
  - Cross-page semantic edges between regions (optional, expensive)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .page_gat_reranker import GATLayer, GATConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RegionGATConfig:
    """Hyperparameters for RegionGraphReranker."""
    # Grid
    grid_rows: int = 2
    grid_cols: int = 2

    # Model dimensions
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    alpha: float = 0.2
    residual: bool = True
    lambda_mix: float = 0.5

    # Edge options
    cross_page_region_edges: bool = False  # expensive; disable for Phase 2 baseline
    sem_threshold_region: float = 0.70


# ---------------------------------------------------------------------------
# Spatial region extraction
# ---------------------------------------------------------------------------

def extract_region_vectors(
    page_multivec: torch.Tensor,    # (T, D)  multi-vector for one page
    grid_rows: int = 2,
    grid_cols: int = 2,
    patch_h: int = 28,
    patch_w: int = 28,
) -> torch.Tensor:
    """
    Split a multi-vector page embedding (patch sequence) into grid regions.

    ColPali encodes a 448×448 image into 28×28=784 patches.
    Each region is the mean-pool of the patches belonging to its grid cell.

    Args:
        page_multivec: (T, D) patch embeddings for one page.
        grid_rows/cols: grid split.
        patch_h/w:      number of patches along each axis (28 for ColPali).

    Returns:
        region_vecs: (grid_rows * grid_cols, D)
    """
    T, D = page_multivec.shape
    expected = patch_h * patch_w

    if T != expected:
        # Fallback: uniform split if sequence length doesn't match expected patches
        chunk_size = max(T // (grid_rows * grid_cols), 1)
        regions = []
        for i in range(grid_rows * grid_cols):
            start = i * chunk_size
            end = start + chunk_size
            regions.append(page_multivec[start:end].mean(dim=0))
        return torch.stack(regions, dim=0)

    # Reshape to (patch_h, patch_w, D) then split into grid cells
    grid = page_multivec.view(patch_h, patch_w, D)
    row_step = patch_h // grid_rows
    col_step = patch_w // grid_cols
    regions = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = grid[r * row_step:(r + 1) * row_step, c * col_step:(c + 1) * col_step, :]
            regions.append(cell.reshape(-1, D).mean(dim=0))

    return torch.stack(regions, dim=0)    # (R, D)


# ---------------------------------------------------------------------------
# Build region-aware adjacency
# ---------------------------------------------------------------------------

def build_region_adjacency(
    K: int,
    R: int,
    page_vectors: torch.Tensor,    # (K, D)
    region_vectors: torch.Tensor,  # (K*R, D)
    query_vector: torch.Tensor,    # (D,)
    page_numbers: List[int],
    include_query_node: bool = True,
    cross_page_region_edges: bool = False,
    sem_threshold: float = 0.70,
) -> torch.Tensor:
    """
    Build a dense adjacency matrix for page+region+query nodes.

    Node ordering: [pages (K), regions (K*R), query (1)]

    Returns:
        adj: (K + K*R + Q, K + K*R + Q)  where Q = 1 if include_query_node
    """
    Q = 1 if include_query_node else 0
    N = K + K * R + Q
    adj = torch.zeros(N, N)

    pv = F.normalize(page_vectors.float(), dim=-1)   # (K, D)
    rv = F.normalize(region_vectors.float(), dim=-1)  # (K*R, D)

    # 1. Page ↔ Page semantic edges
    page_sim = pv @ pv.T   # (K, K)
    page_sim.fill_diagonal_(1.0)
    for i in range(K):
        for j in range(K):
            if i != j and page_sim[i, j] >= sem_threshold:
                adj[i, j] = page_sim[i, j].item()
                adj[j, i] = page_sim[j, i].item()
    # adjacency
    for i in range(K):
        for j in range(K):
            if abs(page_numbers[i] - page_numbers[j]) == 1:
                adj[i, j] = max(adj[i, j].item(), 0.8)
                adj[j, i] = max(adj[j, i].item(), 0.8)

    # 2. Page ↔ Region edges (bidirectional, weight=1.0)
    for pi in range(K):
        for ri in range(R):
            region_global = K + pi * R + ri
            adj[pi, region_global] = 1.0
            adj[region_global, pi] = 1.0

    # 3. Region ↔ Region within same page
    for pi in range(K):
        for r1 in range(R):
            for r2 in range(R):
                if r1 != r2:
                    g1 = K + pi * R + r1
                    g2 = K + pi * R + r2
                    adj[g1, g2] = 0.9

    # 4. Cross-page region semantic edges (optional)
    if cross_page_region_edges:
        region_sim = rv @ rv.T   # (K*R, K*R)
        for gi in range(K * R):
            pi = gi // R
            for gj in range(gi + 1, K * R):
                pj = gj // R
                if pi == pj:
                    continue   # same-page already handled
                if region_sim[gi, gj] >= sem_threshold:
                    adj[K + gi, K + gj] = region_sim[gi, gj].item()
                    adj[K + gj, K + gi] = region_sim[gi, gj].item()

    # 5. Query ↔ all page + region nodes
    if include_query_node:
        q_idx = K + K * R
        for i in range(K + K * R):
            adj[q_idx, i] = 1.0
            adj[i, q_idx] = 1.0

    # Self-loops
    adj += torch.eye(N)

    # Row normalise
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj = adj / row_sum

    return adj


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class RegionGraphReranker(nn.Module):
    """
    Phase-2 reranker that uses both page-level and region-level nodes.

    Scores are produced only for page nodes (K outputs).
    Region nodes act as auxiliary nodes that provide fine-grained visual
    features to improve page-level message passing.
    """

    def __init__(self, config: Optional[RegionGATConfig] = None):
        super().__init__()
        self.config = config or RegionGATConfig()
        cfg = self.config
        self.R = cfg.grid_rows * cfg.grid_cols

        # Input projection
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # GAT layers
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
                    concat=not is_last,
                )
            )

        self.residual_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim) if cfg.residual else None

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
        page_vectors: torch.Tensor,      # (K, D)
        region_vectors: torch.Tensor,    # (K*R, D)
        query_vector: torch.Tensor,      # (D,)
        page_numbers: List[int],
        stage1_scores: Optional[torch.Tensor] = None,  # (K,)
    ) -> torch.Tensor:
        """
        Returns (K,) final page scores.
        """
        cfg = self.config
        K = page_vectors.shape[0]
        R = self.R
        Q = 1
        N = K + K * R + Q

        # Build node features: [page_vecs || region_vecs || query_vec]
        q_vec = F.normalize(query_vector.float().unsqueeze(0), dim=-1)  # (1, D)
        node_feats = torch.cat([page_vectors, region_vectors, q_vec], dim=0)  # (N, D)
        node_feats = F.normalize(node_feats, dim=-1)

        # Build adjacency
        adj = build_region_adjacency(
            K=K,
            R=R,
            page_vectors=page_vectors,
            region_vectors=region_vectors,
            query_vector=query_vector,
            page_numbers=page_numbers,
            include_query_node=True,
            cross_page_region_edges=cfg.cross_page_region_edges,
            sem_threshold=cfg.sem_threshold_region,
        ).to(node_feats.device)

        # GAT forward
        h = self.input_proj(node_feats)    # (N, hidden_dim)
        h_res = h.clone()
        for gat in self.gat_layers:
            h = self.drop(gat(h, adj))
        if self.residual_proj is not None:
            h = h + self.residual_proj(h_res)

        # Score only page nodes
        page_repr = h[:K]                             # (K, output_dim)
        query_repr = h[K + K * R]                    # (output_dim,)
        q_exp = query_repr.unsqueeze(0).expand(K, -1)
        pair = torch.cat([page_repr, q_exp], dim=-1)
        delta = self.score_head(pair).squeeze(-1)    # (K,)

        if stage1_scores is not None:
            s0 = stage1_scores.float()
            s0 = (s0 - s0.min()) / (s0.max() - s0.min() + 1e-8)
            lam = torch.sigmoid(self.lambda_mix)
            final = (1 - lam) * s0 + lam * delta
        else:
            final = delta

        return final   # (K,)

    def extract_regions(
        self,
        page_multivecs: torch.Tensor,   # (K, T, D)
    ) -> torch.Tensor:
        """
        Extract region vectors for all K pages.

        Returns: (K*R, D)
        """
        cfg = self.config
        regions = []
        for k in range(page_multivecs.shape[0]):
            r_vecs = extract_region_vectors(
                page_multivecs[k],
                grid_rows=cfg.grid_rows,
                grid_cols=cfg.grid_cols,
            )   # (R, D)
            regions.append(r_vecs)
        return torch.cat(regions, dim=0)   # (K*R, D)

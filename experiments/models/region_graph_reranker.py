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
    include_typed_nodes: bool = True
    typed_node_scale: float = 1.0


def _safe_logit(p: float) -> float:
    """Numerically stable logit for lambda initialisation."""
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return float(torch.log(torch.tensor(p / (1 - p))))


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


def extract_typed_vectors(
    page_multivec: torch.Tensor,    # (T, D)
    patch_h: int = 28,
    patch_w: int = 28,
) -> torch.Tensor:
    """
    Heuristic typed evidence vectors from layout bands.

    Node order:
      0: header
      1: text_block
      2: caption
      3: table_region
    """
    T, D = page_multivec.shape
    expected = patch_h * patch_w
    if T != expected:
        # Fallback: simple chunking if patch topology is unknown.
        chunk = max(T // 4, 1)
        vecs = []
        for i in range(4):
            s = i * chunk
            e = min(T, s + chunk)
            vecs.append(page_multivec[s:e].mean(dim=0))
        return torch.stack(vecs, dim=0)

    grid = page_multivec.view(patch_h, patch_w, D)
    h20 = max(1, int(0.20 * patch_h))
    h75 = max(h20 + 1, int(0.75 * patch_h))
    h80 = max(h75 + 1, int(0.80 * patch_h))

    # Header: upper band
    header = grid[:h20, :, :].reshape(-1, D).mean(dim=0)
    # Text block: main body band
    text_block = grid[h20:h75, :, :].reshape(-1, D).mean(dim=0)
    # Caption: lower band
    caption = grid[h80:, :, :].reshape(-1, D).mean(dim=0)
    # Table region: central area proxy
    r1, r2 = int(0.35 * patch_h), int(0.70 * patch_h)
    c1, c2 = int(0.10 * patch_w), int(0.90 * patch_w)
    table = grid[r1:r2, c1:c2, :].reshape(-1, D).mean(dim=0)
    return torch.stack([header, text_block, caption, table], dim=0)


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


def build_typed_region_adjacency(
    K: int,
    R: int,
    Tn: int,
    page_vectors: torch.Tensor,     # (K, D)
    region_vectors: torch.Tensor,   # (K*R, D)
    typed_vectors: torch.Tensor,    # (K*Tn, D)
    page_numbers: List[int],
    include_query_node: bool = True,
    cross_page_region_edges: bool = False,
    sem_threshold: float = 0.70,
) -> torch.Tensor:
    """
    Node ordering:
      [pages (K), regions (K*R), typed evidence (K*Tn), query (Q=1)]
    """
    Q = 1 if include_query_node else 0
    n_pages = K
    n_regions = K * R
    n_typed = K * Tn
    base_typed = n_pages + n_regions
    q_idx = n_pages + n_regions + n_typed if include_query_node else -1
    N = n_pages + n_regions + n_typed + Q
    adj = torch.zeros(N, N)

    pv = F.normalize(page_vectors.float(), dim=-1)
    rv = F.normalize(region_vectors.float(), dim=-1)
    tv = F.normalize(typed_vectors.float(), dim=-1)

    # Page <-> Page semantic + adjacent
    page_sim = pv @ pv.T
    page_sim.fill_diagonal_(1.0)
    for i in range(K):
        for j in range(K):
            if i != j and page_sim[i, j] >= sem_threshold:
                adj[i, j] = float(page_sim[i, j])
                adj[j, i] = float(page_sim[j, i])
            if abs(page_numbers[i] - page_numbers[j]) == 1:
                adj[i, j] = max(adj[i, j].item(), 0.8)
                adj[j, i] = max(adj[j, i].item(), 0.8)

    # Page <-> Region
    for pi in range(K):
        for ri in range(R):
            rg = K + pi * R + ri
            adj[pi, rg] = 1.0
            adj[rg, pi] = 1.0

    # Region intra-page full + same row/col stronger links
    rows = max(1, int(R ** 0.5))
    cols = max(1, R // rows)
    for pi in range(K):
        for r1 in range(R):
            g1 = K + pi * R + r1
            rr1, cc1 = divmod(r1, cols)
            for r2 in range(R):
                if r1 == r2:
                    continue
                g2 = K + pi * R + r2
                rr2, cc2 = divmod(r2, cols)
                w = 0.9
                if rr1 == rr2 or cc1 == cc2:
                    w = 1.0  # same table row/col proxy
                adj[g1, g2] = max(adj[g1, g2].item(), w)

    # Optional cross-page region semantic edges
    if cross_page_region_edges:
        region_sim = rv @ rv.T
        for gi in range(K * R):
            pi = gi // R
            for gj in range(gi + 1, K * R):
                pj = gj // R
                if pi == pj:
                    continue
                if region_sim[gi, gj] >= sem_threshold:
                    a, b = K + gi, K + gj
                    w = float(region_sim[gi, gj])
                    adj[a, b] = max(adj[a, b].item(), w)
                    adj[b, a] = max(adj[b, a].item(), w)

    # Page <-> Typed evidence nodes
    for pi in range(K):
        for ti in range(Tn):
            tg = base_typed + pi * Tn + ti
            adj[pi, tg] = 1.0
            adj[tg, pi] = 1.0

    # Typed intra-page structural edges:
    # header<->text, caption<->table
    for pi in range(K):
        header = base_typed + pi * Tn + 0
        text = base_typed + pi * Tn + 1
        caption = base_typed + pi * Tn + 2
        table = base_typed + pi * Tn + 3
        adj[header, text] = max(adj[header, text].item(), 1.0)
        adj[text, header] = max(adj[text, header].item(), 1.0)
        adj[caption, table] = max(adj[caption, table].item(), 1.0)
        adj[table, caption] = max(adj[table, caption].item(), 1.0)

    # Same typed node across adjacent pages
    for i in range(K):
        for j in range(K):
            if i == j or abs(page_numbers[i] - page_numbers[j]) != 1:
                continue
            for t in range(Tn):
                a = base_typed + i * Tn + t
                b = base_typed + j * Tn + t
                # semantic-weighted adjacent typed linkage
                sim = float((tv[i * Tn + t] * tv[j * Tn + t]).sum())
                w = max(0.7, sim)
                adj[a, b] = max(adj[a, b].item(), w)
                adj[b, a] = max(adj[b, a].item(), w)

    # Query <-> all evidence nodes
    if include_query_node:
        for i in range(N - 1):
            adj[q_idx, i] = 1.0
            adj[i, q_idx] = 1.0

    adj += torch.eye(N)
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return adj / row_sum


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
        self.Tn = 4  # header, text_block, caption, table_region

        # Input projection
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.type_emb = nn.Embedding(7, cfg.hidden_dim)  # page,region,header,text,caption,table,query

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
        self.evidence_head = nn.Linear(cfg.output_dim, 1)

        self.lambda_mix = nn.Parameter(torch.tensor(_safe_logit(cfg.lambda_mix), dtype=torch.float))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        page_vectors: torch.Tensor,      # (K, D)
        region_vectors: torch.Tensor,    # (K*R, D)
        query_vector: torch.Tensor,      # (D,)
        page_numbers: List[int],
        stage1_scores: Optional[torch.Tensor] = None,  # (K,)
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Returns (K,) final page scores.
        """
        cfg = self.config
        K = page_vectors.shape[0]
        R = self.R
        Tn = self.Tn if cfg.include_typed_nodes else 0
        Q = 1
        N = K + K * R + K * Tn + Q

        # Build node features: [page_vecs || region_vecs || typed_vecs || query_vec]
        q_vec = F.normalize(query_vector.float().unsqueeze(0), dim=-1)  # (1, D)
        typed_vectors = None
        if Tn > 0:
            # typed_vectors is expected to be provided by caller through extractor wrapper
            typed_vectors = region_vectors.new_zeros((K * Tn, region_vectors.shape[-1]))
        node_chunks = [page_vectors, region_vectors]
        if typed_vectors is not None:
            node_chunks.append(typed_vectors)
        node_chunks.append(q_vec)
        node_feats = torch.cat(node_chunks, dim=0)  # (N, D)
        node_feats = F.normalize(node_feats, dim=-1)

        # Build adjacency
        if Tn > 0:
            adj = build_typed_region_adjacency(
                K=K,
                R=R,
                Tn=Tn,
                page_vectors=page_vectors,
                region_vectors=region_vectors,
                typed_vectors=typed_vectors,
                page_numbers=page_numbers,
                include_query_node=True,
                cross_page_region_edges=cfg.cross_page_region_edges,
                sem_threshold=cfg.sem_threshold_region,
            ).to(node_feats.device)
        else:
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
        type_ids = torch.zeros(N, dtype=torch.long, device=h.device)
        type_ids[K:K + K * R] = 1
        if Tn > 0:
            base_t = K + K * R
            for pi in range(K):
                off = base_t + pi * Tn
                type_ids[off + 0] = 2
                type_ids[off + 1] = 3
                type_ids[off + 2] = 4
                type_ids[off + 3] = 5
        type_ids[-1] = 6
        h = h + self.type_emb(type_ids) * cfg.typed_node_scale
        h_res = h.clone()
        for gat in self.gat_layers:
            h = self.drop(gat(h, adj))
        if self.residual_proj is not None:
            h = h + self.residual_proj(h_res)

        # Score only page nodes
        page_repr = h[:K]                             # (K, output_dim)
        q_index = K + K * R + K * Tn
        query_repr = h[q_index]                      # (output_dim,)
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

        if not return_aux:
            return final   # (K,)

        evidence_start = K
        evidence_end = K + K * R + K * Tn
        region_repr = h[evidence_start:evidence_end]   # (K*R + K*Tn, output_dim)
        evidence_scores = self.evidence_head(region_repr).squeeze(-1)
        return final, {
            "evidence_scores": evidence_scores,
            "page_delta": delta,
        }

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

    def extract_typed_nodes(
        self,
        page_multivecs: torch.Tensor,   # (K, T, D)
    ) -> torch.Tensor:
        """Extract heuristic typed evidence nodes per page."""
        typed = []
        for k in range(page_multivecs.shape[0]):
            typed.append(extract_typed_vectors(page_multivecs[k]))  # (4, D)
        return torch.cat(typed, dim=0)   # (K*4, D)

    def rerank_from_multivector(
        self,
        page_embs: torch.Tensor,            # (K, T, D)
        query_embs: torch.Tensor,           # (1, T, D)
        page_numbers: List[int],
        stage1_scores: Optional[torch.Tensor] = None,  # (K,)
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Convenience wrapper for cached ColPali tensors.
        """
        page_vectors = page_embs.float().mean(dim=1)               # (K, D)
        query_vector = query_embs.float().squeeze(0).mean(dim=0)   # (D,)
        region_vectors = self.extract_regions(page_embs.float())    # (K*R, D)
        typed_vectors = self.extract_typed_nodes(page_embs.float()) if self.config.include_typed_nodes else None
        if typed_vectors is not None:
            # Concatenate typed vectors after region vectors by reusing forward's expected slot.
            region_plus = torch.cat([region_vectors, typed_vectors], dim=0)
            # forward will split based on K,R and include_typed_nodes; pass expanded tensor.
            # Keep compatibility by passing region vectors only and overriding inside via closure not possible.
            # So we call direct custom path below.
            cfg = self.config
            K = page_vectors.shape[0]
            R = self.R
            Tn = self.Tn
            q_vec = F.normalize(query_vector.float().unsqueeze(0), dim=-1)
            node_feats = torch.cat([page_vectors, region_vectors, typed_vectors, q_vec], dim=0)
            node_feats = F.normalize(node_feats, dim=-1)
            adj = build_typed_region_adjacency(
                K=K,
                R=R,
                Tn=Tn,
                page_vectors=page_vectors,
                region_vectors=region_vectors,
                typed_vectors=typed_vectors,
                page_numbers=page_numbers,
                include_query_node=True,
                cross_page_region_edges=cfg.cross_page_region_edges,
                sem_threshold=cfg.sem_threshold_region,
            ).to(node_feats.device)
            h = self.input_proj(node_feats)
            type_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            type_ids[K:K + K * R] = 1
            base_t = K + K * R
            for pi in range(K):
                off = base_t + pi * Tn
                type_ids[off + 0] = 2
                type_ids[off + 1] = 3
                type_ids[off + 2] = 4
                type_ids[off + 3] = 5
            type_ids[-1] = 6
            h = h + self.type_emb(type_ids) * cfg.typed_node_scale
            h_res = h.clone()
            for gat in self.gat_layers:
                h = self.drop(gat(h, adj))
            if self.residual_proj is not None:
                h = h + self.residual_proj(h_res)
            page_repr = h[:K]
            query_repr = h[K + K * R + K * Tn]
            q_exp = query_repr.unsqueeze(0).expand(K, -1)
            pair = torch.cat([page_repr, q_exp], dim=-1)
            delta = self.score_head(pair).squeeze(-1)
            if stage1_scores is not None:
                s0 = stage1_scores.float()
                s0 = (s0 - s0.min()) / (s0.max() - s0.min() + 1e-8)
                lam = torch.sigmoid(self.lambda_mix)
                final = (1 - lam) * s0 + lam * delta
            else:
                final = delta
            if not return_aux:
                return final
            evidence_repr = h[K:K + K * R + K * Tn]
            evidence_scores = self.evidence_head(evidence_repr).squeeze(-1)
            return final, {"evidence_scores": evidence_scores, "page_delta": delta}

        return self.forward(
            page_vectors=page_vectors,
            region_vectors=region_vectors,
            query_vector=query_vector,
            page_numbers=page_numbers,
            stage1_scores=stage1_scores,
            return_aux=return_aux,
        )

"""
Edge-construction routines for the query-conditioned evidence graph.

Edge types implemented:
  1. Semantic  — connects page pairs whose pooled embeddings exceed a cosine threshold.
  2. Adjacency — connects consecutive pages within a document (|page_i - page_j| <= gap).
  3. Query     — connects the query node to all page nodes (directed).
  4. Structural — figure-caption / table-paragraph proximity (requires layout info).

All functions accept (N,) or (N, D) tensors and return an edge_index of shape (2, E)
following the PyG convention, plus optional edge_weight (E,).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Semantic edges
# ---------------------------------------------------------------------------

def compute_semantic_edges(
    page_vectors: torch.Tensor,
    threshold: float = 0.65,
    top_k: Optional[int] = None,
    self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build semantic similarity edges between page nodes.

    Args:
        page_vectors: (N, D) L2-normalised pooled page embeddings.
        threshold:    Cosine similarity cutoff to add an edge.
        top_k:        If given, keep only the top-k neighbours per node
                      regardless of threshold (used for very dense graphs).
        self_loops:   Whether to include self-loops.

    Returns:
        edge_index:  (2, E)  LongTensor  [source, target]
        edge_weight: (E,)    FloatTensor cosine similarities
    """
    page_vectors = F.normalize(page_vectors.float(), dim=-1)  # (N, D)
    N = page_vectors.shape[0]

    # Full pairwise cosine similarity matrix (N, N)
    sim = page_vectors @ page_vectors.T  # (N, N)

    if not self_loops:
        sim.fill_diagonal_(float("-inf"))

    if top_k is not None:
        # Keep only the top-k similarities per row
        _, top_idx = sim.topk(min(top_k, N - 1), dim=1)
        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        sim = sim.masked_fill(~mask, float("-inf"))

    # Apply threshold
    mask = sim >= threshold
    src, dst = mask.nonzero(as_tuple=True)
    weights = sim[src, dst]

    edge_index = torch.stack([src, dst], dim=0)  # (2, E)
    return edge_index, weights


# ---------------------------------------------------------------------------
# 2. Page adjacency edges
# ---------------------------------------------------------------------------

def compute_adjacency_edges(
    page_numbers: List[int],
    max_gap: int = 1,
    include_cross_doc: bool = False,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Connect pages that are within `max_gap` positions of each other.

    Args:
        page_numbers:     List of absolute page numbers for each node.
        max_gap:          Maximum distance in page number to add an edge.
        include_cross_doc: If False (default) and doc_ids provided, only
                           connect pages within the same document.
        doc_ids:          Document ID per node (to prevent cross-doc edges).

    Returns:
        edge_index:  (2, E)
        edge_weight: (E,)  weight = 1 / (1 + |page_i - page_j|)
    """
    N = len(page_numbers)
    src_list, dst_list, w_list = [], [], []

    for i in range(N):
        for j in range(i + 1, N):
            if doc_ids is not None and not include_cross_doc and doc_ids[i] != doc_ids[j]:
                continue
            gap = abs(page_numbers[i] - page_numbers[j])
            if gap <= max_gap:
                w = 1.0 / (1.0 + gap)
                src_list += [i, j]
                dst_list += [j, i]
                w_list += [w, w]

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float)
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# 3. Query-to-page edges
# ---------------------------------------------------------------------------

def compute_query_edges(
    num_page_nodes: int,
    query_node_idx: int,
    bidirectional: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Connect query node to all page nodes.

    Args:
        num_page_nodes: number of page nodes (indices 0 … N-1).
        query_node_idx: index assigned to the query node (usually N).
        bidirectional:  also add page → query edges.

    Returns:
        edge_index:  (2, E)
        edge_weight: (E,)  all 1.0
    """
    page_idx = torch.arange(num_page_nodes, dtype=torch.long)
    q_idx = torch.full((num_page_nodes,), query_node_idx, dtype=torch.long)

    src = torch.cat([q_idx, page_idx] if bidirectional else [q_idx])
    dst = torch.cat([page_idx, q_idx] if bidirectional else [page_idx])

    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = torch.ones(edge_index.shape[1])
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# 4. Structural edges (optional — needs layout parser output)
# ---------------------------------------------------------------------------

def compute_structural_edges(
    layout_info: List[Dict],
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Connect nodes based on document structure (figure↔caption, table↔paragraph).

    Args:
        layout_info: list of dicts per node with keys:
                     - "type": "figure" | "table" | "caption" | "paragraph" | "page"
                     - "section": section heading string (optional)
                     - "bbox": [x0, y0, x1, y1] in normalised [0, 1] coords (optional)
        num_nodes:  total number of nodes

    Returns:
        edge_index:  (2, E)
        edge_weight: (E,)
    """
    PAIRED = {
        ("figure", "caption"), ("caption", "figure"),
        ("table", "paragraph"), ("paragraph", "table"),
    }

    src_list, dst_list, w_list = [], [], []

    for i in range(num_nodes):
        ti = layout_info[i].get("type", "page")
        si = layout_info[i].get("section")
        for j in range(i + 1, num_nodes):
            tj = layout_info[j].get("type", "page")
            sj = layout_info[j].get("section")

            # Figure-caption / table-paragraph pairing
            if (ti, tj) in PAIRED:
                src_list += [i, j]
                dst_list += [j, i]
                w_list += [1.0, 1.0]
                continue

            # Same-section link
            if si and sj and si == sj:
                src_list += [i, j]
                dst_list += [j, i]
                w_list += [0.7, 0.7]

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float)
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Graph assembly helper
# ---------------------------------------------------------------------------

def build_full_edge_index(
    page_vectors: torch.Tensor,
    page_numbers: List[int],
    query_node_idx: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
    layout_info: Optional[List[Dict]] = None,
    sem_threshold: float = 0.65,
    sem_top_k: Optional[int] = 5,
    adj_max_gap: int = 1,
    use_structural: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble all edge types into a single edge_index and edge_weight.

    Args:
        page_vectors:    (N, D) pooled page embeddings
        page_numbers:    page position list [0, 1, 2, …]
        query_node_idx:  index of query node (None = no query edges)
        doc_ids:         document IDs per page (prevents cross-doc adj edges)
        layout_info:     per-node layout dicts (None = skip structural edges)
        sem_threshold:   cosine threshold for semantic edges
        sem_top_k:       max neighbours per node for semantic edges
        adj_max_gap:     max page gap for adjacency edges
        use_structural:  whether to add structural edges

    Returns:
        edge_index:  (2, E)  LongTensor
        edge_weight: (E,)    FloatTensor
    """
    all_ei: List[torch.Tensor] = []
    all_ew: List[torch.Tensor] = []

    # Semantic
    ei, ew = compute_semantic_edges(page_vectors, threshold=sem_threshold, top_k=sem_top_k)
    if ei.numel() > 0:
        all_ei.append(ei)
        all_ew.append(ew)

    # Adjacency
    ei, ew = compute_adjacency_edges(page_numbers, max_gap=adj_max_gap, doc_ids=doc_ids)
    if ei.numel() > 0:
        all_ei.append(ei)
        all_ew.append(ew)

    # Query
    if query_node_idx is not None:
        ei, ew = compute_query_edges(len(page_numbers), query_node_idx)
        if ei.numel() > 0:
            all_ei.append(ei)
            all_ew.append(ew)

    # Structural
    if use_structural and layout_info is not None:
        ei, ew = compute_structural_edges(layout_info, len(page_numbers))
        if ei.numel() > 0:
            all_ei.append(ei)
            all_ew.append(ew)

    if not all_ei:
        N = len(page_numbers)
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    edge_index = torch.cat(all_ei, dim=1)
    edge_weight = torch.cat(all_ew, dim=0)

    # De-duplicate edges by taking max weight
    edge_index, edge_weight = _dedup_edges(edge_index, edge_weight)

    return edge_index, edge_weight


def _dedup_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge duplicate (src, dst) edges by max weight."""
    # Build a dictionary keyed by (src, dst)
    edge_dict: Dict[Tuple[int, int], float] = {}
    for (s, d), w in zip(edge_index.T.tolist(), edge_weight.tolist()):
        key = (int(s), int(d))
        edge_dict[key] = max(edge_dict.get(key, float("-inf")), w)

    if not edge_dict:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    pairs = list(edge_dict.keys())
    weights = list(edge_dict.values())
    ei = torch.tensor(pairs, dtype=torch.long).T            # (2, E)
    ew = torch.tensor(weights, dtype=torch.float)           # (E,)
    return ei, ew


# ---------------------------------------------------------------------------
# Adjacency matrix from edge_index
# ---------------------------------------------------------------------------

def edge_index_to_adj(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
    row_normalize: bool = True,
) -> torch.Tensor:
    """
    Convert edge_index / edge_weight to a dense adjacency matrix.

    Args:
        edge_index:     (2, E)
        edge_weight:    (E,)
        num_nodes:      size of the adjacency matrix
        add_self_loops: add I to the adjacency before normalization
        row_normalize:  apply D^-1 A normalisation

    Returns:
        adj: (num_nodes, num_nodes)  FloatTensor
    """
    adj = torch.zeros(num_nodes, num_nodes)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = edge_weight

    if add_self_loops:
        adj = adj + torch.eye(num_nodes)

    if row_normalize:
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adj = adj / row_sum

    return adj

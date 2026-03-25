"""
PageIndex — persistent index of page embeddings for a document corpus.

Supports:
- Indexing a collection of documents (each a list of PIL images)
- Saving/loading to disk
- Efficient top-k retrieval per query
- BM25 baseline retrieval (requires `rank_bm25`)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageRecord:
    """Metadata for a single indexed page."""
    global_idx: int    # position in the flat index
    doc_id: str
    page_idx: int      # local page position within its document
    page_id: str       # unique string identifier


class PageIndex:
    """
    Flat index of page embeddings across a document corpus.

    Usage
    -----
    >>> idx = PageIndex(inferencer)
    >>> idx.add_document("doc_001", images, page_ids=["d001_p0", "d001_p1"])
    >>> idx.build()
    >>> top_idx, top_scores = idx.retrieve("What is the revenue?", k=10)
    >>> records = [idx.records[i] for i in top_idx[0].tolist()]
    """

    def __init__(self, inferencer=None):
        """
        Args:
            inferencer: a ColVisionInferencer instance (can be None if loading from disk)
        """
        self.inferencer = inferencer
        self._page_embs: List[torch.Tensor] = []   # each: (T, D)
        self.records: List[PageRecord] = []
        self._index_embs: Optional[torch.Tensor] = None  # (N, T, D) — built after add_document calls

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        images: List[Image.Image],
        page_ids: Optional[List[str]] = None,
        desc: Optional[str] = None,
    ) -> None:
        """
        Encode and add all pages of a document to the index.

        Args:
            doc_id: unique document identifier
            images: list of page PIL images
            page_ids: optional list of string page identifiers
            desc: description for tqdm progress bar
        """
        if self.inferencer is None:
            raise RuntimeError("Cannot index: no inferencer attached.")

        page_ids = page_ids or [f"{doc_id}_p{i}" for i in range(len(images))]
        embs = self.inferencer.encode_pages(images, desc=desc or f"Indexing {doc_id}")

        start = len(self.records)
        for local_idx, (emb, pid) in enumerate(zip(embs, page_ids)):
            self._page_embs.append(emb)  # (T, D)
            self.records.append(PageRecord(
                global_idx=start + local_idx,
                doc_id=doc_id,
                page_idx=local_idx,
                page_id=pid,
            ))

        self._index_embs = None  # invalidate cache

    def add_documents_batch(
        self,
        doc_ids: List[str],
        images_list: List[List[Image.Image]],
        page_ids_list: Optional[List[List[str]]] = None,
    ) -> None:
        """Add multiple documents to the index."""
        page_ids_list = page_ids_list or [None] * len(doc_ids)
        for doc_id, images, page_ids in zip(doc_ids, images_list, page_ids_list):
            self.add_document(doc_id, images, page_ids)

    def build(self) -> None:
        """Stack all page embeddings into a single tensor for fast batch scoring."""
        if not self._page_embs:
            raise RuntimeError("Index is empty. Call add_document() first.")
        self._index_embs = torch.stack(self._page_embs, dim=0)  # (N, T, D)
        logger.info("Index built: %d pages, embedding shape %s", len(self.records), self._index_embs.shape)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        queries: List[str],
        k: int = 10,
        per_doc: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k pages for a list of queries.

        Args:
            queries: list of query strings
            k: number of results per query
            per_doc: if True, k pages are retrieved per document independently
                     (useful when doc_id is known at query time)

        Returns:
            indices: (Q, k) global page indices
            scores:  (Q, k) MaxSim scores
        """
        if self._index_embs is None:
            self.build()

        query_embs = self.inferencer.encode_queries(queries)
        return self.inferencer.retrieve_top_k(query_embs, self._index_embs, k=k)

    def retrieve_with_records(
        self,
        queries: List[str],
        k: int = 10,
    ) -> List[List[Tuple[PageRecord, float]]]:
        """
        Retrieve top-k pages and return (PageRecord, score) pairs.

        Returns:
            List (one per query) of top-k (PageRecord, score) sorted descending.
        """
        indices, scores = self.retrieve(queries, k=k)
        results = []
        for q_idx in range(len(queries)):
            hits = []
            for rank in range(k):
                global_idx = indices[q_idx, rank].item()
                score = scores[q_idx, rank].item()
                hits.append((self.records[global_idx], score))
            results.append(hits)
        return results

    def get_doc_pages(self, doc_id: str) -> List[PageRecord]:
        """Return all PageRecord entries for a given document, in order."""
        return [r for r in self.records if r.doc_id == doc_id]

    def get_page_embs_for_doc(self, doc_id: str) -> torch.Tensor:
        """
        Return stacked multi-vector embeddings for all pages in a document.
        Shape: (num_pages, T, D)
        """
        if self._index_embs is None:
            self.build()
        indices = [r.global_idx for r in self.get_doc_pages(doc_id)]
        return self._index_embs[indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save index to disk (separate tensor and metadata files)."""
        os.makedirs(directory, exist_ok=True)
        if self._index_embs is None:
            self.build()
        torch.save(self._index_embs, os.path.join(directory, "embs.pt"))
        with open(os.path.join(directory, "records.json"), "w") as f:
            json.dump([vars(r) for r in self.records], f, indent=2)
        logger.info("Saved index (%d pages) to %s", len(self.records), directory)

    @classmethod
    def load(cls, directory: str) -> "PageIndex":
        """Load a previously saved index (no inferencer needed)."""
        obj = cls(inferencer=None)
        emb_path = os.path.join(directory, "embs.pt")
        rec_path = os.path.join(directory, "records.json")
        obj._index_embs = torch.load(emb_path, map_location="cpu")
        obj._page_embs = list(obj._index_embs.unbind(dim=0))
        with open(rec_path) as f:
            obj.records = [PageRecord(**r) for r in json.load(f)]
        logger.info("Loaded index: %d pages from %s", len(obj.records), directory)
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_docs(self) -> int:
        return len({r.doc_id for r in self.records})

    def stats(self) -> Dict:
        if not self.records:
            return {"num_pages": 0, "num_docs": 0}
        return {
            "num_pages": len(self.records),
            "num_docs": self.num_docs,
            "avg_pages_per_doc": round(len(self.records) / self.num_docs, 2),
        }


# ---------------------------------------------------------------------------
# BM25 baseline index (text-based, requires OCR or metadata)
# ---------------------------------------------------------------------------

class BM25PageIndex:
    """
    BM25 retrieval baseline. Requires `rank_bm25` (pip install rank-bm25).
    Uses OCR text or question keywords — for pages without OCR, falls back
    to page ID strings.
    """

    def __init__(self):
        self.records: List[PageRecord] = []
        self._corpus: List[List[str]] = []   # tokenized page texts
        self._bm25 = None

    def add_page(self, record: PageRecord, text: str) -> None:
        self.records.append(record)
        self._corpus.append(text.lower().split())
        self._bm25 = None  # invalidate

    def build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("Install rank_bm25: pip install rank-bm25") from e
        self._bm25 = BM25Okapi(self._corpus)

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[PageRecord, float]]:
        if self._bm25 is None:
            self.build()
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.records[i], float(scores[i])) for i in top_k]

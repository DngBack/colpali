"""
ColVision inferencer — wraps any ColPali/ColQwen-style model from colpali_engine
to encode pages and queries, and compute late-interaction (MaxSim) scores.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from colpali_engine.utils.hf_peft_patches import apply_hf_peft_patches

logger = logging.getLogger(__name__)

# Ensures Transformers 5.5+ PEFT fixes apply even if this module is imported without colpali_engine.models.
apply_hf_peft_patches()

# ---------------------------------------------------------------------------
# Padding utility (sequence length may vary across batches)
# ---------------------------------------------------------------------------


def _pad_to_max_seq_len(embs_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad a list of (B, T, D) tensors to the same T then concatenate on dim=0.

    Some ColVision processors can yield variable sequence lengths depending on
    dynamic padding/resizing. This makes `torch.cat` fail across batches.
    We pad with zeros along the token dimension.
    """
    if not embs_list:
        return torch.empty(0, 0, 0)

    max_t = max(t.shape[1] for t in embs_list)
    d = embs_list[0].shape[2]
    padded: List[torch.Tensor] = []
    for t in embs_list:
        if t.shape[2] != d:
            raise ValueError(f"Embedding dim mismatch: expected {d}, got {t.shape[2]}")
        if t.shape[1] == max_t:
            padded.append(t)
            continue
        if t.shape[1] > max_t:
            padded.append(t[:, :max_t, :])
            continue
        pad_len = max_t - t.shape[1]
        pad = torch.zeros(t.shape[0], pad_len, d, dtype=t.dtype, device=t.device)
        padded.append(torch.cat([t, pad], dim=1))

    return torch.cat(padded, dim=0)

# ---------------------------------------------------------------------------
# Supported model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "colpali": (
        "colpali_engine.models.paligemma.colpali",
        "ColPali",
        "ColPaliProcessor",
    ),
    "colqwen2": (
        "colpali_engine.models.qwen2.colqwen2",
        "ColQwen2",
        "ColQwen2Processor",
    ),
    "colqwen2_5": (
        "colpali_engine.models.qwen2_5.colqwen2_5",
        "ColQwen2_5",
        "ColQwen2_5_Processor",
    ),
    "colqwen3": (
        "colpali_engine.models.qwen3.colqwen3",
        "ColQwen3",
        "ColQwen3Processor",
    ),
    "colidefics3": (
        "colpali_engine.models.idefics3.colidefics3",
        "ColIdefics3",
        "ColIdefics3Processor",
    ),
}


def _import_model_and_processor(model_type: str):
    import importlib
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(_MODEL_REGISTRY)}")
    module_path, model_cls, proc_cls = _MODEL_REGISTRY[model_type]
    mod = importlib.import_module(module_path)
    return getattr(mod, model_cls), getattr(mod, proc_cls)


# ---------------------------------------------------------------------------
# Late-interaction score (MaxSim)
# ---------------------------------------------------------------------------

def maxsim_score(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    batch_size: int = 64,
    doc_chunk_size: int = 16,
) -> torch.Tensor:
    """
    Compute MaxSim late-interaction scores between queries and documents.

    Args:
        query_embs: (Q, Tq, D)  — L2-normalised multi-vector query embeddings
        doc_embs:   (N, Td, D)  — L2-normalised multi-vector doc embeddings
        batch_size: process queries in chunks to avoid OOM
        doc_chunk_size: process this many pages at a time per query batch.
            The full ``(B, N, Tq, Td)`` tensor can explode RAM / CUDA with
            large N and long token sequences (e.g. ColIdefics on 60-page docs).

    Returns:
        scores: (Q, N)  float32 tensor
    """
    Q, Tq, D = query_embs.shape
    N = doc_embs.shape[0]
    scores = torch.zeros(Q, N, device=query_embs.device, dtype=torch.float32)

    if doc_chunk_size < 1:
        doc_chunk_size = N

    for q_start in range(0, Q, batch_size):
        q_chunk = query_embs[q_start : q_start + batch_size]  # (B, Tq, D)
        q_end = q_start + q_chunk.shape[0]
        for n_start in range(0, N, doc_chunk_size):
            n_end = min(n_start + doc_chunk_size, N)
            d_chunk = doc_embs[n_start:n_end]  # (n_sub, Td, D)
            # (B, Tq, D) x (n_sub, Td, D) -> (B, n_sub, Tq, Td)
            raw = torch.einsum("bqd,ntd->bnqt", q_chunk, d_chunk)
            # MaxSim: max over doc tokens, sum over query tokens
            scores[q_start:q_end, n_start:n_end] = raw.amax(dim=-1).sum(dim=-1)

    return scores


def pool_multivector(embs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean-pool multi-vector embeddings to a single vector per item.

    Args:
        embs: (B, T, D) or (T, D)
        mask: (B, T) boolean, True = valid token (optional)

    Returns:
        pooled: (B, D) or (D,)
    """
    if embs.dim() == 2:
        if mask is not None:
            valid = embs[mask]
            return valid.mean(dim=0) if valid.numel() > 0 else embs.mean(dim=0)
        return embs.mean(dim=0)

    if mask is not None:
        # (B, T, D) masked mean
        mask_f = mask.float().unsqueeze(-1)           # (B, T, 1)
        summed = (embs * mask_f).sum(dim=1)           # (B, D)
        counts = mask_f.sum(dim=1).clamp(min=1)       # (B, 1)
        return summed / counts
    return embs.mean(dim=1)


# ---------------------------------------------------------------------------
# Main inferencer
# ---------------------------------------------------------------------------

class ColVisionInferencer:
    """
    Wrapper around any ColVision model (ColPali, ColQwen2, …) for encoding
    pages and queries, and for retrieving top-k candidates via MaxSim.

    Example
    -------
    >>> infer = ColVisionInferencer("vidore/colpali-v1.2", model_type="colpali")
    >>> page_embs = infer.encode_pages(images)         # (N, T, 128)
    >>> query_embs = infer.encode_queries(["..."])     # (1, T, 128)
    >>> scores = infer.score(query_embs, page_embs)    # (1, N)
    >>> top_idx, top_scores = infer.retrieve_top_k(query_embs, page_embs, k=10)
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "colpali",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 4,
        maxsim_doc_chunk: int = 16,
        show_progress: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.maxsim_doc_chunk = max(1, int(maxsim_doc_chunk))
        self.show_progress = show_progress

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_dtype = torch_dtype

        apply_hf_peft_patches()
        ModelCls, ProcessorCls = _import_model_and_processor(model_type)

        logger.info("Loading %s from '%s' …", model_type, model_name_or_path)
        # Transformers 5.x may leave some params on "meta" until a device_map load; calling .to(cuda)
        # then raises "Cannot copy out of meta tensor". Loading with device_map materializes weights
        # on the target device in one step.
        if self.device.type == "cuda":
            device_map_arg: Union[str, dict] = f"cuda:{self.device.index}" if self.device.index is not None else "cuda:0"
        else:
            device_map_arg = "cpu"
        self.model = ModelCls.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map_arg,
        ).eval()

        self.processor = ProcessorCls.from_pretrained(model_name_or_path)
        logger.info("Model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_pages(
        self,
        images: List[Image.Image],
        desc: str = "Encoding pages",
    ) -> torch.Tensor:
        """
        Encode a list of page images.

        Returns:
            embs: (N, T, D) float32 on CPU
        """
        all_embs: List[torch.Tensor] = []
        for i in tqdm(range(0, len(images), self.batch_size), desc=desc, disable=not self.show_progress):
            batch_imgs = images[i : i + self.batch_size]
            batch = self.processor.process_images(batch_imgs).to(self.device)
            embs = self.model(**batch).to(dtype=torch.float32).cpu()  # (B, T, D)
            all_embs.append(embs)

        return _pad_to_max_seq_len(all_embs)  # (N, T, D)

    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        desc: str = "Encoding queries",
    ) -> torch.Tensor:
        """
        Encode a list of text queries.

        Returns:
            embs: (Q, T, D) float32 on CPU
        """
        all_embs: List[torch.Tensor] = []
        for i in tqdm(range(0, len(queries), self.batch_size), desc=desc, disable=not self.show_progress):
            batch_q = queries[i : i + self.batch_size]
            batch = self.processor.process_queries(batch_q).to(self.device)
            embs = self.model(**batch).to(dtype=torch.float32).cpu()  # (B, T, D)
            all_embs.append(embs)

        return _pad_to_max_seq_len(all_embs)  # (Q, T, D)

    # ------------------------------------------------------------------
    # Scoring & retrieval
    # ------------------------------------------------------------------

    def score(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
    ) -> torch.Tensor:
        """MaxSim late-interaction: (Q, N)."""
        return maxsim_score(
            query_embs,
            doc_embs,
            batch_size=self.batch_size * 4,
            doc_chunk_size=self.maxsim_doc_chunk,
        )

    def retrieve_top_k(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
        k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (indices, scores) of top-k pages per query.

        Returns:
            indices: (Q, k)  LongTensor
            scores:  (Q, k)  FloatTensor
        """
        scores = self.score(query_embs, doc_embs)         # (Q, N)
        k = min(k, scores.shape[1])
        top_scores, top_idx = scores.topk(k, dim=1, sorted=True)
        return top_idx, top_scores

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_page_vectors(self, page_embs: torch.Tensor) -> torch.Tensor:
        """
        Pool multi-vector page embeddings to single vectors for graph nodes.

        Returns: (N, D)
        """
        return pool_multivector(page_embs)

    def get_query_vector(self, query_embs: torch.Tensor) -> torch.Tensor:
        """
        Pool multi-vector query embeddings to single vectors.

        Returns: (Q, D) or (D,) for single query
        """
        pooled = pool_multivector(query_embs)
        return pooled.squeeze(0) if query_embs.shape[0] == 1 else pooled

    @property
    def embedding_dim(self) -> int:
        return self.model.dim  # 128 for ColPali

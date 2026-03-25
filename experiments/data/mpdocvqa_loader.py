"""
MP-DocVQA dataset loader for X-PageRerank experiments.

Dataset: https://rrc.cvc.uab.es/?ch=17  (official)
HuggingFace mirror: "HuggingFaceM4/MP-DocVQA"  (or "vidore/mpdocvqa")

Each sample contains a multi-page document and a question whose answer
may require evidence from one or more pages.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MPDocVQASample:
    """One question–document pair from MP-DocVQA."""
    question_id: str
    question: str
    answers: List[str]
    doc_id: str
    page_ids: List[str]
    answer_page_idx: int          # primary answer page (0-based within this document)
    support_page_idxs: List[int]  # all pages that contain supporting evidence
    images: List[Image.Image]     # page images in order

    @property
    def num_pages(self) -> int:
        return len(self.images)

    @property
    def is_cross_page(self) -> bool:
        """True when answer evidence spans more than one page."""
        return len(set(self.support_page_idxs)) > 1


class MPDocVQADataset(Dataset):
    """
    PyTorch Dataset wrapping MP-DocVQA for retrieval experiments.

    Supports two loading modes:
    1. HuggingFace Hub  (default) — requires `datasets` package.
    2. Local JSON  — pass ``local_json_path`` to the constructor.

    Local JSON format expected:
    [
      {
        "questionId": "...", "question": "...", "answers": [...],
        "doc_id": "...", "page_ids": [...], "answer_page_idx": 0,
        "evidence_pages": [0],          # optional
        "image_paths": ["path/p0.png", ...]  # list of page image paths
      }, ...
    ]
    """

    HF_DATASET_ID = "lmms-lab/MP-DocVQA"

    def __init__(
        self,
        split: str = "train",
        hf_dataset_id: Optional[str] = None,
        local_json_path: Optional[str] = None,
        max_pages_per_doc: int = 60,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        streaming: bool = False,
        num_samples: Optional[int] = None,
        sample_offset: int = 0,
    ):
        self.split = split
        self.max_pages_per_doc = max_pages_per_doc

        if local_json_path is not None:
            self.samples = self._load_from_json(local_json_path)
        else:
            self.samples = self._load_from_hf(
                hf_dataset_id or self.HF_DATASET_ID,
                split,
                cache_dir,
                hf_token,
                streaming=streaming,
                num_samples=num_samples,
                sample_offset=sample_offset,
            )

        logger.info(
            "MPDocVQA [%s]: %d samples | %d cross-page",
            split, len(self.samples),
            sum(1 for s in self.samples if s.is_cross_page),
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_from_hf(
        self,
        hf_id: str,
        split: str,
        cache_dir: Optional[str],
        hf_token: Optional[str] = None,
        streaming: bool = False,
        num_samples: Optional[int] = None,
        sample_offset: int = 0,
    ) -> List[MPDocVQASample]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Install `datasets` via `pip install datasets`") from e

        kwargs: Dict = {"cache_dir": cache_dir, "streaming": streaming}
        if hf_token:
            kwargs["token"] = hf_token

        if streaming:
            logger.info(
                "Streaming dataset '%s' [%s]%s",
                hf_id, split,
                f" — first {num_samples} samples" if num_samples else "",
            )

        # Auto-enable offline mode when network is unreachable
        import os as _os
        if not _os.environ.get("HF_DATASETS_OFFLINE"):
            try:
                import socket
                socket.setdefaulttimeout(3)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            except OSError:
                logger.warning("Network unreachable — enabling HF_DATASETS_OFFLINE=1")
                _os.environ["HF_DATASETS_OFFLINE"] = "1"
                kwargs.pop("token", None)  # token not needed offline

        try:
            raw = load_dataset(hf_id, split=split, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "Unauthorized" in err_msg or "cannot be accessed" in err_msg:
                raise PermissionError(
                    f"\n\nDataset '{hf_id}' requires authentication or does not exist.\n"
                    "Options:\n"
                    "  1) Pass --hf_token YOUR_TOKEN  (get at https://huggingface.co/settings/tokens)\n"
                    "  2) Use local parquet:  --parquet_dir ~/.cache/huggingface/hub/datasets--lmms-lab"
                    "--MP-DocVQA/snapshots/<hash>/data/\n"
                ) from e
            # Split not found → try what's available in cache
            for alt in ("test", "val", "validation", "train"):
                if alt == split:
                    continue
                try:
                    logger.warning(
                        "Split '%s' not found locally, trying '%s' as substitute", split, alt
                    )
                    raw = load_dataset(hf_id, split=alt, **kwargs)
                    logger.info("Loaded '%s' split as substitute for '%s'", alt, split)
                    break
                except Exception:
                    continue
            else:
                raise

        # Apply offset + limit (for non-overlapping train/val splits)
        if streaming:
            if sample_offset > 0:
                raw = raw.skip(sample_offset)
            if num_samples is not None:
                raw = raw.take(num_samples)
        else:
            total = len(raw)
            start = sample_offset
            end = min(start + num_samples, total) if num_samples is not None else total
            if start > 0 or end < total:
                raw = raw.select(range(start, end))
                logger.info("Selected rows %d–%d of %d total", start, end - 1, total)

        samples = []
        for row in raw:
            try:
                samples.append(self._parse_hf_row(row))
            except Exception as exc:
                logger.debug("Skipping malformed row: %s", exc)
        return samples

    def _parse_hf_row(self, row: dict) -> MPDocVQASample:
        # --- question / id ---
        question = row.get("question") or row.get("query") or row.get("prompt") or ""
        qid = str(
            row.get("questionId") or row.get("question_id") or row.get("id") or id(row)
        )

        # --- answers ---
        answers = row.get("answers") or row.get("answer") or []
        if isinstance(answers, str):
            try:
                import json
                answers = json.loads(answers)
            except Exception:
                answers = [answers] if answers not in ("[]", "") else []
        if not isinstance(answers, list):
            answers = [str(answers)]

        # --- doc id ---
        doc_id = str(
            row.get("doc_id") or row.get("documentId")
            or row.get("document_id") or row.get("doc_name") or ""
        )

        # --- images ---
        # lmms-lab/MP-DocVQA format: images in columns image_1 … image_20
        images: List[Image.Image] = []

        if "image_1" in row:
            # lmms-lab columnar format
            for i in range(1, 21):
                img = row.get(f"image_{i}")
                if img is None:
                    break
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                elif isinstance(img, (str, os.PathLike)):
                    images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, bytes):
                    import io
                    images.append(Image.open(io.BytesIO(img)).convert("RGB"))
        else:
            # Standard list format: "images" or single "image"
            raw_images = row.get("images") or ([row["image"]] if "image" in row else [])
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            for img in raw_images:
                if img is None:
                    continue
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                elif isinstance(img, (str, os.PathLike)):
                    images.append(Image.open(img).convert("RGB"))

        images = images[: self.max_pages_per_doc]
        n = len(images)

        # --- answer page index ---
        raw_idx = row.get("answer_page_idx") or row.get("page_idx") or row.get("answer_page")
        # Handle '[]' (unlabeled competition test set) → skip row
        if raw_idx in (None, "", "[]", [], "null"):
            raise ValueError("Row has no answer_page_idx — likely unlabeled test split, skipping")
        if isinstance(raw_idx, str):
            try:
                import json
                parsed = json.loads(raw_idx)
                raw_idx = parsed[0] if isinstance(parsed, list) and parsed else 0
            except Exception:
                raw_idx = 0
        answer_page_idx = min(int(raw_idx), max(n - 1, 0))

        # --- support pages ---
        evidence = (
            row.get("evidence_pages")
            or row.get("supporting_pages")
            or row.get("support_pages")
        )
        if evidence is not None:
            if isinstance(evidence, (int, float)):
                support_idxs = [int(evidence)]
            else:
                support_idxs = [int(e) for e in evidence if int(e) < n]
        else:
            support_idxs = [answer_page_idx]
        support_idxs = support_idxs or [answer_page_idx]

        # --- page_ids ---
        raw_pids = row.get("page_ids")
        if isinstance(raw_pids, str):
            try:
                import json
                raw_pids = json.loads(raw_pids)
            except Exception:
                raw_pids = None
        page_ids = list(raw_pids) if raw_pids else [f"{doc_id}_p{i}" for i in range(n)]

        return MPDocVQASample(
            question_id=qid,
            question=question,
            answers=answers,
            doc_id=doc_id,
            page_ids=page_ids,
            answer_page_idx=answer_page_idx,
            support_page_idxs=support_idxs,
            images=images,
        )

    @classmethod
    def from_parquet_dir(
        cls,
        parquet_dir: str,
        split_prefix: str = "val",
        max_pages_per_doc: int = 60,
        num_samples: Optional[int] = None,
        sample_offset: int = 0,
    ) -> "MPDocVQADataset":
        """
        Load directly from local .parquet files (e.g. from HF hub cache).

        Args:
            parquet_dir:    Directory containing *.parquet files.
            split_prefix:   File prefix to match (e.g. "val", "train").
            max_pages_per_doc: Max pages to load per document.
            num_samples:    Maximum number of samples to load (None = all).
            sample_offset:  Skip the first N rows (use with num_samples to create
                            non-overlapping train/val splits from the same file).
                            E.g. train: offset=0, n=800 → val: offset=800, n=200

        Example:
            DATA = "~/.cache/huggingface/hub/datasets--lmms-lab--MP-DocVQA/snapshots/<hash>/data/"
            train_ds = MPDocVQADataset.from_parquet_dir(DATA, num_samples=800)
            val_ds   = MPDocVQADataset.from_parquet_dir(DATA, sample_offset=800, num_samples=200)
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        import glob
        pattern = os.path.join(os.path.expanduser(parquet_dir), f"{split_prefix}-*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No parquet files matching '{pattern}' found.\n"
                f"Directory contents: {os.listdir(os.path.expanduser(parquet_dir))}"
            )
        logger.info("Loading %d parquet files from %s", len(files), parquet_dir)

        raw = load_dataset("parquet", data_files={split_prefix: files}, split=split_prefix)
        total = len(raw)
        start = sample_offset
        end = min(start + num_samples, total) if num_samples is not None else total
        if start > 0 or end < total:
            raw = raw.select(range(start, end))
            logger.info("Selected rows %d–%d of %d total", start, end - 1, total)

        obj = cls.__new__(cls)
        obj.split = split_prefix
        obj.max_pages_per_doc = max_pages_per_doc
        obj.samples = []
        for row in raw:
            try:
                obj.samples.append(obj._parse_hf_row(row))
            except Exception as exc:
                logger.debug("Skipping row: %s", exc)

        logger.info(
            "MPDocVQA [parquet/%s]: %d samples | %d cross-page",
            split_prefix, len(obj.samples),
            sum(1 for s in obj.samples if s.is_cross_page),
        )
        return obj

    def _load_from_json(self, json_path: str) -> List[MPDocVQASample]:
        with open(json_path) as f:
            data = json.load(f)

        samples = []
        for row in data:
            image_paths = row.get("image_paths", [])
            images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
            images = images[: self.max_pages_per_doc]
            n = len(images)
            answer_page_idx = min(int(row.get("answer_page_idx", 0)), max(n - 1, 0))
            evidence = row.get("evidence_pages")
            support_idxs = [int(e) for e in evidence if int(e) < n] if evidence else [answer_page_idx]
            samples.append(
                MPDocVQASample(
                    question_id=str(row["questionId"]),
                    question=row["question"],
                    answers=row.get("answers", [row.get("answer", "")]),
                    doc_id=str(row.get("doc_id", "")),
                    page_ids=row.get("page_ids", [f"p{i}" for i in range(n)]),
                    answer_page_idx=answer_page_idx,
                    support_page_idxs=support_idxs,
                    images=images,
                )
            )
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MPDocVQASample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # Subset helpers
    # ------------------------------------------------------------------

    def _make_subset(self, samples: List[MPDocVQASample]) -> "MPDocVQADataset":
        obj = MPDocVQADataset.__new__(MPDocVQADataset)
        obj.split = self.split
        obj.max_pages_per_doc = self.max_pages_per_doc
        obj.samples = samples
        return obj

    def get_multi_page_subset(self) -> "MPDocVQADataset":
        return self._make_subset([s for s in self.samples if s.num_pages > 1])

    def get_cross_page_subset(self) -> "MPDocVQADataset":
        """Questions that need evidence from ≥ 2 pages."""
        return self._make_subset([s for s in self.samples if s.is_cross_page])

    def get_single_hop_subset(self) -> "MPDocVQADataset":
        return self._make_subset([s for s in self.samples if not s.is_cross_page])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        total = len(self.samples)
        multi_page = sum(1 for s in self.samples if s.num_pages > 1)
        cross_page = sum(1 for s in self.samples if s.is_cross_page)
        avg_pages = sum(s.num_pages for s in self.samples) / max(total, 1)
        avg_support = sum(len(s.support_page_idxs) for s in self.samples) / max(total, 1)
        return {
            "split": self.split,
            "total_questions": total,
            "multi_page_docs": multi_page,
            "cross_page_questions": cross_page,
            "avg_pages_per_doc": round(avg_pages, 2),
            "avg_support_pages": round(avg_support, 2),
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(f"\n=== MP-DocVQA [{s['split']}] ===")
        for k, v in s.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_mpdocvqa(batch: List[MPDocVQASample]) -> Dict:
    """
    Returns a dict with:
    - questions: List[str]
    - question_ids: List[str]
    - doc_images: List[List[PIL.Image]]   (one list per sample)
    - support_page_idxs: List[List[int]]
    - answers: List[List[str]]
    """
    return {
        "questions": [s.question for s in batch],
        "question_ids": [s.question_id for s in batch],
        "doc_images": [s.images for s in batch],
        "support_page_idxs": [s.support_page_idxs for s in batch],
        "answer_page_idxs": [s.answer_page_idx for s in batch],
        "answers": [s.answers for s in batch],
        "doc_ids": [s.doc_id for s in batch],
        "num_pages": [s.num_pages for s in batch],
    }

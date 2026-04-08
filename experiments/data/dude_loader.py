"""
DUDE (Document Understanding Dataset and Evaluation) loader for X-PageRerank.

Dataset: https://rrc.cvc.uab.es/?ch=23  (official)
HuggingFace: "jordyvl/DUDE_loader" or "BartKoelman/DUDE"

DUDE has diverse, long documents with complex questions including
abstractive, extractive, and unanswerable types. It tests robustness
beyond MP-DocVQA's extractive setting.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Question types in DUDE
QUESTION_TYPES = ("extractive", "abstractive", "not-answerable", "list")


@dataclass
class DUDESample:
    """One question–document pair from DUDE."""
    question_id: str
    question: str
    answers: List[str]
    answer_type: str
    doc_id: str
    doc_name: str
    page_ids: List[str]
    support_page_idxs: List[int]   # pages containing supporting evidence
    images: List[Image.Image]

    @property
    def num_pages(self) -> int:
        return len(self.images)

    @property
    def is_answerable(self) -> bool:
        return self.answer_type != "not-answerable"

    @property
    def is_cross_page(self) -> bool:
        return len(set(self.support_page_idxs)) > 1


class DUDEDataset(Dataset):
    """
    PyTorch Dataset wrapping DUDE for retrieval experiments.

    Loading modes:
    1. HuggingFace Hub — pass ``hf_dataset_id`` (default "jordyvl/DUDE_loader").
    2. Local JSON — pass ``local_json_path``.

    Local JSON format expected:
    [
      {
        "questionId": "...", "question": "...", "answers": [...],
        "answer_type": "extractive",
        "doc_id": "...", "doc_name": "...",
        "evidence_pages": [2, 5],
        "image_paths": ["path/p0.png", ...]
      }, ...
    ]
    """

    HF_DATASET_ID = "jordyvl/DUDE_loader"

    def __init__(
        self,
        split: str = "train",
        hf_dataset_id: Optional[str] = None,
        local_json_path: Optional[str] = None,
        max_pages_per_doc: int = 100,
        answerable_only: bool = False,
        cache_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        sample_offset: int = 0,
    ):
        # jordyvl/DUDE_loader exposes splits: train, val, test (not "validation")
        if split == "validation":
            split = "val"
        self.split = split
        self.max_pages_per_doc = max_pages_per_doc
        self.answerable_only = answerable_only

        if local_json_path is not None:
            self.samples = self._load_from_json(local_json_path)
        else:
            self.samples = self._load_from_hf(hf_dataset_id or self.HF_DATASET_ID, self.split, cache_dir)

        if answerable_only:
            self.samples = [s for s in self.samples if s.is_answerable]

        if sample_offset > 0 or num_samples is not None:
            end = sample_offset + num_samples if num_samples is not None else None
            self.samples = self.samples[sample_offset:end]

        logger.info(
            "DUDE [%s]: %d samples | %d answerable | %d cross-page",
            split, len(self.samples),
            sum(1 for s in self.samples if s.is_answerable),
            sum(1 for s in self.samples if s.is_cross_page),
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_from_hf(self, hf_id: str, split: str, cache_dir: Optional[str]) -> List[DUDESample]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Install `datasets` via `pip install datasets`") from e

        # jordyvl/DUDE_loader uses a Hub dataset *script* (no Parquet in-repo).
        # - datasets 2.x: must pass trust_remote_code=True (non-interactive; avoids [y/N] prompt).
        # - datasets 3.x: script loaders were removed — pin datasets<3 or use local_json_path.
        import datasets as datasets_pkg

        major = int(datasets_pkg.__version__.split(".", 1)[0])
        load_kw: Dict[str, object] = {"cache_dir": cache_dir}
        if major < 3:
            load_kw["trust_remote_code"] = True

        try:
            raw = load_dataset(hf_id, split=split, **load_kw)
        except ModuleNotFoundError as e:
            # jordyvl/DUDE_loader imports pdf2image at module load time.
            missing = getattr(e, "name", None) or ""
            if "pdf2image" in str(e) or missing == "pdf2image":
                raise RuntimeError(
                    "The Hub dataset script for DUDE imports `pdf2image`. Install it:\n"
                    "  pip install pdf2image\n"
                    "You also need Poppler on the system (pdf2image calls `pdftoppm`), e.g. on Ubuntu/Debian:\n"
                    "  sudo apt install poppler-utils\n"
                    "Or use --local_json_path with pre-rendered page images (see docstring)."
                ) from e
            raise
        except RuntimeError as e:
            msg = str(e)
            if "Dataset scripts are no longer supported" in msg or "loading script" in msg.lower():
                raise RuntimeError(
                    "This DUDE Hub repo uses a dataset loading script, which requires Hugging Face "
                    "`datasets` 2.x (v3 dropped script loaders). Fix: pip install 'datasets>=2.19.1,<3' "
                    "then retry. Alternative: export data to JSON and pass local_json_path (see docstring)."
                ) from e
            raise
        return [self._parse_hf_row(row) for row in raw]

    def _parse_hf_row(self, row: dict) -> DUDESample:
        qid = str(row.get("questionId") or row.get("question_id") or id(row))
        question = row.get("question", "")
        answers = row.get("answers", [row.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]

        answer_type = row.get("answer_type", "extractive")
        doc_id = str(row.get("docId") or row.get("doc_id", ""))
        doc_name = str(row.get("docName") or row.get("doc_name", ""))

        # Images
        images: List[Image.Image] = []
        raw_images = row.get("images") or ([row["image"]] if "image" in row else [])
        for img in raw_images:
            if img is not None:
                images.append(img if isinstance(img, Image.Image) else Image.open(img).convert("RGB"))
        images = images[: self.max_pages_per_doc]
        n = len(images)

        # Evidence/support pages
        evidence = row.get("evidence_pages") or row.get("supporting_pages") or row.get("page_idx")
        if evidence is None:
            support_idxs = [0]
        elif isinstance(evidence, (int, float)):
            support_idxs = [int(evidence)]
        else:
            support_idxs = [int(e) for e in evidence if int(e) < n]
        support_idxs = support_idxs or [0]

        page_ids = list(row.get("page_ids", [f"{doc_id}_p{i}" for i in range(n)]))

        return DUDESample(
            question_id=qid,
            question=question,
            answers=answers,
            answer_type=answer_type,
            doc_id=doc_id,
            doc_name=doc_name,
            page_ids=page_ids,
            support_page_idxs=support_idxs,
            images=images,
        )

    def _load_from_json(self, json_path: str) -> List[DUDESample]:
        with open(json_path) as f:
            data = json.load(f)

        samples = []
        for row in data:
            image_paths = row.get("image_paths", [])
            images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
            images = images[: self.max_pages_per_doc]
            n = len(images)
            evidence = row.get("evidence_pages", [0])
            support_idxs = [int(e) for e in evidence if int(e) < n] or [0]
            samples.append(
                DUDESample(
                    question_id=str(row["questionId"]),
                    question=row["question"],
                    answers=row.get("answers", [row.get("answer", "")]),
                    answer_type=row.get("answer_type", "extractive"),
                    doc_id=str(row.get("doc_id", "")),
                    doc_name=str(row.get("doc_name", "")),
                    page_ids=row.get("page_ids", [f"p{i}" for i in range(n)]),
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

    def __getitem__(self, idx: int) -> DUDESample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # Subset helpers
    # ------------------------------------------------------------------

    def _make_subset(self, samples: List[DUDESample]) -> "DUDEDataset":
        obj = DUDEDataset.__new__(DUDEDataset)
        obj.split = self.split
        obj.max_pages_per_doc = self.max_pages_per_doc
        obj.answerable_only = self.answerable_only
        obj.samples = samples
        return obj

    def filter_by_type(self, answer_type: str) -> "DUDEDataset":
        assert answer_type in QUESTION_TYPES, f"Unknown type: {answer_type}"
        return self._make_subset([s for s in self.samples if s.answer_type == answer_type])

    def get_cross_page_subset(self) -> "DUDEDataset":
        return self._make_subset([s for s in self.samples if s.is_cross_page])

    def get_long_doc_subset(self, min_pages: int = 10) -> "DUDEDataset":
        return self._make_subset([s for s in self.samples if s.num_pages >= min_pages])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        total = len(self.samples)
        if total == 0:
            return {"total": 0}

        type_counts = {t: sum(1 for s in self.samples if s.answer_type == t) for t in QUESTION_TYPES}
        cross_page = sum(1 for s in self.samples if s.is_cross_page)
        avg_pages = sum(s.num_pages for s in self.samples) / total
        avg_support = sum(len(s.support_page_idxs) for s in self.samples) / total

        return {
            "split": self.split,
            "total_questions": total,
            "answerable": sum(1 for s in self.samples if s.is_answerable),
            "cross_page_questions": cross_page,
            "avg_pages_per_doc": round(avg_pages, 2),
            "avg_support_pages": round(avg_support, 2),
            **{f"type_{k}": v for k, v in type_counts.items()},
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(f"\n=== DUDE [{s['split']}] ===")
        for k, v in s.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Shared collate
# ---------------------------------------------------------------------------

def collate_dude(batch: List[DUDESample]) -> Dict:
    return {
        "questions": [s.question for s in batch],
        "question_ids": [s.question_id for s in batch],
        "doc_images": [s.images for s in batch],
        "support_page_idxs": [s.support_page_idxs for s in batch],
        "answers": [s.answers for s in batch],
        "answer_types": [s.answer_type for s in batch],
        "doc_ids": [s.doc_id for s in batch],
        "num_pages": [s.num_pages for s in batch],
    }

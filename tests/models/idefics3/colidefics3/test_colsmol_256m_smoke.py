"""
Smoke tests for vidore/colSmol-256M (ColIdefics3 / SmolVLM backbone).

These tests download model weights from Hugging Face and are marked ``slow``;
CI runs ``pytest -m "not slow"`` and skips them.
"""

from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

MODEL_NAME = "vidore/colSmol-256M"


@pytest.fixture(scope="module")
def device() -> str:
    return get_torch_device("auto")


@pytest.fixture(scope="module")
def colsmol_model(device: str) -> Generator[ColIdefics3, None, None]:
    yield cast(
        ColIdefics3,
        ColIdefics3.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def colsmol_processor() -> ColIdefics3Processor:
    return cast(ColIdefics3Processor, ColIdefics3Processor.from_pretrained(MODEL_NAME))


@pytest.mark.slow
def test_colsmol_256m_image_and_query_embeddings(
    colsmol_model: ColIdefics3,
    colsmol_processor: ColIdefics3Processor,
):
    images = [
        Image.new("RGB", (64, 48), color="white"),
        Image.new("RGB", (48, 32), color="black"),
    ]
    queries = [
        "What is on this page?",
        "Summarize the figure.",
    ]

    batch_images = colsmol_processor.process_images(images).to(colsmol_model.device)
    batch_queries = colsmol_processor.process_queries(queries).to(colsmol_model.device)

    with torch.inference_mode():
        image_embeddings = colsmol_model(**batch_images)
        query_embeddings = colsmol_model(**batch_queries)

    assert image_embeddings.ndim == 3
    assert query_embeddings.ndim == 3
    assert image_embeddings.shape[0] == len(images)
    assert query_embeddings.shape[0] == len(queries)
    assert image_embeddings.shape[-1] == colsmol_model.dim
    assert query_embeddings.shape[-1] == colsmol_model.dim
    assert torch.isfinite(image_embeddings).all()
    assert torch.isfinite(query_embeddings).all()


@pytest.mark.slow
def test_colsmol_256m_score_multi_vector_shape(
    colsmol_model: ColIdefics3,
    colsmol_processor: ColIdefics3Processor,
):
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (24, 24), color="blue"),
    ]
    queries = ["First query", "Second query"]

    batch_images = colsmol_processor.process_images(images).to(colsmol_model.device)
    batch_queries = colsmol_processor.process_queries(queries).to(colsmol_model.device)

    with torch.inference_mode():
        image_embeddings = colsmol_model(**batch_images)
        query_embeddings = colsmol_model(**batch_queries)

    scores = colsmol_processor.score_multi_vector(query_embeddings, image_embeddings)

    assert scores.shape == (len(queries), len(images))
    assert torch.isfinite(scores).all()

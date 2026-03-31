# ColPali: Efficient Document Retrieval with Vision Language Models 👀

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/ViDoRe_Benchmark-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/vidore)
[![GitHub](https://img.shields.io/badge/Cookbooks-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tonywu71/colpali-cookbooks)

[![Test](https://github.com/illuin-tech/colpali/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/illuin-tech/colpali/actions/workflows/test.yml)
[![Version](https://img.shields.io/pypi/v/colpali-engine?color=%2334D058&label=pypi%20package)](https://pypi.org/project/colpali-engine/)
[![Downloads](https://static.pepy.tech/badge/colpali-engine)](https://pepy.tech/project/colpali-engine)

---

[[Model card]](https://huggingface.co/vidore/colpali)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
[[Demo]](https://huggingface.co/spaces/manu/ColPali-demo)
[[Blog Post]](https://huggingface.co/blog/manu/colpali)

## Associated Paper

This repository contains the code used for training the vision retrievers in the [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://arxiv.org/abs/2407.01449) paper. In particular, it contains the code for training the ColPali model, which is a vision retriever based on the ColBERT architecture and the PaliGemma model.

## Introduction

With our new model *ColPali*, we propose to leverage VLMs to construct efficient multi-vector embeddings in the visual space for document retrieval. By feeding the ViT output patches from PaliGemma-3B to a linear projection, we create a multi-vector representation of documents. We train the model to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

![ColPali Architecture](assets/colpali_architecture.webp)

## List of ColVision models

| Model                                                               | Score on [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) 🏆 | License    | Comments                                                                                                                                                       | Currently supported |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| [vidore/colpali](https://huggingface.co/vidore/colpali)             | 81.3                                                                          | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Checkpoint used in the ColPali paper.                                                                         | ❌                   |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1)   | 81.5                                                                          | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Fix right padding for queries.                                                                                | ✅                   |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)   | 83.9                                                                          | Gemma      | • Similar to `vidore/colpali-v1.1`.                                                                                                                            | ✅                   |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3)   | 84.8                                                                          | Gemma      | • Similar to `vidore/colpali-v1.2`.<br />• Trained with a larger effective batch size of 256 batch size for 3 epochs.                                          | ✅                   |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 87.3                                                                          | Apache 2.0 | • Based on `Qwen/Qwen2-VL-2B-Instruct`.<br />• Supports dynamic resolution.<br />• Trained using 768 image patches per page and an effective batch size of 32. | ✅                   |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 89.3                                                                          | Apache 2.0 | • Similar to `vidore/colqwen2-v0.1`, but trained with more powerful GPUs and with a larger effective batch size (256).                                         | ✅                   |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 88.8                                                                          | Apache 2.0 | • Based on `Qwen/Qwen2 5-VL-3B-Instruct`<br />• Supports dynamic resolution.<br />• Trained using 768 image patches per page and an effective batch size of 32.                                         | ✅                   |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 89.4                                                                          | Apache 2.0 | • Similar to `vidore/colqwen2.5-v0.1`, but trained with slightly different hyper parameters                                        | ✅                   |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 90.6                                                                           | Apache 2.0 | • Based on the Qwen3-VL backbone.<br />• 320-dim ColBERT-style embeddings with dynamic resolution.<br />• Trained for multi-vector document retrieval.          | ✅                   |
| [athrael-soju/colqwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) | 90.9                                                                           | Apache 2.0 | • Based on `Qwen/Qwen3.5-4B` (hybrid GatedDeltaNet + full-attention).<br />• 320-dim ColBERT-style embeddings.<br />• 4.5B params, LoRA-trained.          | ✅                   |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M)   | 80.1                                                                          | Apache 2.0 | • Based on `HuggingFaceTB/SmolVLM-256M-Instruct`.                                                                                                              | ✅                   |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M)   | 82.3                                                                          | Apache 2.0 | • Based on `HuggingFaceTB/SmolVLM-500M-Instruct`.                                                                                                              | ✅                   |
| [Cognitive-Lab/ColNetraEmbed](https://huggingface.co/Cognitive-Lab/ColNetraEmbed) | 86.4                                                                          | Gemma      | • Based on `google/gemma-3-4b-it`.<br />• Multi-vector late interaction retrieval model.<br />• Multilingual support across 22 languages.      | ✅                   |
| [Cognitive-Lab/NetraEmbed](https://huggingface.co/Cognitive-Lab/NetraEmbed)       | 81.0                                                                          | Gemma      | • Based on `google/gemma-3-4b-it`.<br />• Bi-encoder retrieval model.<br />• Supports Matryoshka embeddings (768, 1536, 2560).<br />• Multilingual support across 22 languages. | ✅                   |

## Setup

We used Python 3.11.6 and PyTorch 2.4 to train and test our models, but the codebase is compatible with Python >=3.10 and recent PyTorch versions. To install the package, run:

```bash
pip install colpali-engine # from PyPi
pip install git+https://github.com/illuin-tech/colpali # from source
```

Mac users using MPS with the ColQwen models have reported errors with torch 2.6.0. These errors are fixed by downgrading to torch 2.5.1.

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.

## Usage

### Quick start

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

We now support `fast-plaid` experimentally to make matching quicker for larger corpus sizes:

```python
# !pip install --no-deps fast-plaid fastkmeans

# Process the inputs by batches of 4
dataloader = DataLoader(
    dataset=images,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: processor.process_images(x),
)

ds  = []
for batch_doc in tqdm(dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

plaid_index = processor.create_plaid_index(ds)

scores = processor.get_topk_plaid(query_embeddings, plaid_index, k=10)
```

### Benchmarking

To benchmark ColPali on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

### Interpretability with similarity maps

By superimposing the late interaction similarity maps on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

To use the `interpretability` module, you need to install the `colpali-engine[interpretability]` package:

```bash
pip install colpali-engine[interpretability]
```

Then, after generating your embeddings with ColPali, use the following code to plot the similarity maps for each query token:

<details>
<summary><strong>🔽 Click to expand code snippet</strong></summary>

```python
import torch
from PIL import Image

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

model_name = "vidore/colpali-v1.3"
device = get_torch_device("auto")

# Load the model
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

# Load the processor
processor = ColPaliProcessor.from_pretrained(model_name)

# Load the image and query
image = Image.open("shift_kazakhstan.jpg")
query = "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"

# Preprocess inputs
batch_images = processor.process_images([image]).to(device)
batch_queries = processor.process_queries([query]).to(device)

# Forward passes
with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)

# Get the number of image patches
n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size)

# Get the tensor mask to filter out the embeddings that are not related to the image
image_mask = processor.get_image_mask(batch_images)

# Generate the similarity maps
batched_similarity_maps = get_similarity_maps_from_embeddings(
    image_embeddings=image_embeddings,
    query_embeddings=query_embeddings,
    n_patches=n_patches,
    image_mask=image_mask,
)

# Get the similarity map for our (only) input image
similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

# Tokenize the query
query_tokens = processor.tokenizer.tokenize(query)

# Plot and save the similarity maps for each query token
plots = plot_all_similarity_maps(
    image=image,
    query_tokens=query_tokens,
    similarity_maps=similarity_maps,
)
for idx, (fig, ax) in enumerate(plots):
    fig.savefig(f"similarity_map_{idx}.png")
```

</details>

For a more detailed example, you can refer to the interpretability notebooks from the [ColPali Cookbooks 👨🏻‍🍳](https://github.com/tonywu71/colpali-cookbooks) repository.

### Token pooling

[Token pooling](https://doi.org/10.48550/arXiv.2409.14683) is a CRUDE-compliant method (document addition/deletion-friendly) that aims at reducing the sequence length of multi-vector embeddings. For ColPali, many image patches share redundant information, e.g. white background patches. By pooling these patches together, we can reduce the amount of embeddings while retaining most of the page's signal. Retrieval performance with hierarchical mean token pooling on image embeddings can be found in the [ColPali paper](https://doi.org/10.48550/arXiv.2407.01449). In our experiments, we found that a pool factor of 3 offered the optimal trade-off: the total number of vectors is reduced by $66.7\%$ while $97.8\%$ of the original performance is maintained.

To use token pooling, you can use the `HierarchicalEmbeddingPooler` class from the `colpali-engine` package:

<details>
<summary><strong>🔽 Click to expand code snippet</strong></summary>

```python
import torch

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

# Dummy multivector embeddings
list_embeddings = [
    torch.rand(10, 768),
    torch.rand(20, 768),
]

# Define the pooler with the desired level of compression
pooler = HierarchicalTokenPooler()

# Pool the embeddings
outputs = pooler.pool_embeddings(list_embeddings, pool_factor=2)
```

If your inputs are padded 3D tensor embeddings instead of lists of 2D tensors, use `padding=True` and specify the padding used by your tokenizer to make sure the `HierarchicalTokenPooler` correctly removes the padding values before pooling:

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from colpali_engine.models import ColQwen2, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)

token_pooler = HierarchicalTokenPooler()

# Your page images
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (32, 32), color="black"),
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)

# Apply token pooling (reduces the sequence length of the multi-vector embeddings)
image_embeddings = token_pooler.pool_embeddings(
    image_embeddings,
    pool_factor=2,
    padding=True,
    padding_side=processor.tokenizer.padding_side,
)
```

</details>

### Training

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script for ColPali. You can do this using the following command:

```bash
pip install "colpali-engine[train]"
```

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

<details>
<summary><strong>🔽 Example 1: Local training</strong></summary>


```bash
accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py
```

</details>

<details>
<summary><strong>🔽 Example 2: Training on a SLURM cluster</strong></summary>

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py"
```

</details>

## Contributing

We welcome contributions to ColPali! 🤗

To contribute to ColPali, first install the development dependencies for proper testing/linting:

```bash
pip install "colpali-engine[dev]"
```

To run all the tests, you will have to install all optional dependencies (or you'll get an error in test discovery):

```bash
pip install "colpali-engine[all]"
```

When your PR is ready, ping one of the repository maintainers. We will do our best to review it as soon as possible!

## Community Projects

Several community projects and ressources have been developed around ColPali to facilitate its usage. Feel free to reach out if you want to add your project to this list!

<details>
<summary><strong>🔽 Libraries 📚</strong></summary>

| Library Name  | Description                                                                                                                                                                                                                                          |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
| Byaldi        | [`Byaldi`](https://github.com/AnswerDotAI/byaldi) is [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)'s equivalent for ColPali, leveraging the `colpali-engine` package to facilitate indexing and storing embeddings.                      |
| PyVespa       | [`PyVespa`](https://pyvespa.readthedocs.io/en/latest/examples/colpali-document-retrieval-vision-language-models-cloud.html) allows interaction with [Vespa](https://vespa.ai/), a production-grade vector database, with detailed ColPali support.   |
| Qdrant | Tutorial about using ColQwen2 with the [Qdrant](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/) vector database. |
| Elastic Search     | Tutorial about using ColPali with the [Elastic Search](https://www.elastic.co/search-labs/blog/elastiacsearch-colpali-document-search) vector database. |
| Weaviate | Tutorial about using multi-vector embeddings with the [Weaviate](https://weaviate.io/developers/weaviate/tutorials/multi-vector-embeddings) vector database. |
| Candle        | [Candle](https://github.com/huggingface/candle/tree/main/candle-examples/examples/colpali) enables ColPali inference with an efficient ML framework for Rust.                                                                                        |
| EmbedAnything | [`EmbedAnything`](https://github.com/StarlightSearch/EmbedAnything) Allows end-to-end ColPali inference with both Candle and ONNX backend.                                                                                                           |
| DocAI         | [DocAI](https://github.com/PragmaticMachineLearning/docai) uses ColPali with GPT-4o and Langchain to extract structured information from documents.                                                                                                  |
| VARAG         | [VARAG](https://github.com/adithya-s-k/VARAG) uses ColPali in a vision-only and a hybrid RAG pipeline.                                                                                                                                               |
| ColBERT Live! | [`ColBERT Live!`](https://github.com/jbellis/colbert-live/) enables ColPali usage with vector databases supporting large datasets, compression, and non-vector predicates.                                                                           |
| ColiVara      | [`ColiVara`](https://github.com/tjmlabs/ColiVara/) is retrieval API that allows you to store, search, and retrieve documents based on their visual embedding. It is a web-first implementation of the ColPali paper using ColQwen2 as the LLM model. |
| BentoML       | Deploy ColPali easily with BentoML using [this example repository](https://github.com/bentoml/BentoColPali). BentoML features adaptive batching and zero-copy I/O to minimize overhead.                                                              |
| NoOCR       | NoOCR is end-to-end, [open source](https://github.com/kyryl-opens-ml/no-ocr) solution for complex PDFs, powered by ColPali embeddings. |
| Astra Multi-vector     | [`Astra-multivector`](https://github.com/brian-ogrady/astradb-multivector) provides enterprise-grade integration with AstraDB for late-interaction models like ColPali, ColQwen2, and ColBERT. It implements efficient token pooling and embedding caching strategies to dramatically reduce latency and index size while maintaining retrieval quality. The library leverages Cassandra's distributed architecture for high-throughput vector search at scale. |
| Mixpeek       | [Mixpeek](https://docs.mixpeek.com/processing/feature-extractors) is a production platform for multimodal late-interaction retrieval. It supports models like ColBERT, ColPaLI, and ColQwen2 with built-in indexing, versioning, A/B testing, and explainability across image, text, video, and PDF pipelines. |


</details>

<details>
<summary><strong>🔽 Notebooks 📙</strong></summary>

| Notebook Title                                               | Author & Link                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ColPali Cookbooks                                            | [Tony's Cookbooks (ILLUIN)](https://github.com/tonywu71/colpali-cookbooks) 🙋🏻 |
| Vision RAG Tutorial                                          | [Manu's Vision Rag Tutorial (ILLUIN)](https://github.com/ManuelFay/Tutorials/blob/main/Tuesday_Practical_2_Vision_RAG.ipynb) 🙋🏻 |
| ColPali (Byaldi) + Qwen2-VL for RAG                          | [Merve's Notebook (HuggingFace 🤗)](https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb) |
| Indexing ColPali with Qdrant                                 | [Daniel's Notebook (HuggingFace 🤗)](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html) |
| Weaviate Tutorial                                            | [Connor's ColPali POC (Weaviate)](https://github.com/weaviate/recipes/blob/main/weaviate-features/named-vectors/NamedVectors-ColPali-POC.ipynb) |
| Use ColPali for Multi-Modal Retrieval with Milvus            | [Milvus Documentation](https://milvus.io/docs/use_ColPali_with_milvus.md) |
| Data Generation                                              | [Daniel's Notebook (HuggingFace 🤗)](https://danielvanstrien.xyz/posts/post-with-code/colpali/2024-09-23-generate_colpali_dataset.html) |
| Finance Report Analysis with ColPali and Gemini              | [Jaykumaran (LearnOpenCV)](https://github.com/spmallick/learnopencv/tree/master/Multimodal-RAG-with-ColPali-Gemini) |
| Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs) | [Sergio Paniego](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms) |
| Document Similarity Search with ColPali                      | [Frank Sommers](https://colab.research.google.com/github/fsommers/documentai/blob/main/Document_Similarity_with_ColPali_0_2_2_version.ipynb) |
| End-to-end ColPali inference with EmbedAnything              | [Akshay Ballal (EmbedAnything)](https://colab.research.google.com/drive/1-Eiaw8wMm8I1n69N1uKOHkmpw3yV22w8?usp=sharing) |
| ColiVara: A ColPali Retrieval API                            | [A simple RAG Example](https://github.com/tjmlabs/ColiVara-docs/blob/main/cookbook/RAG.ipynb) |
| Multimodal RAG with Document Retrieval (ColPali), Vision Language Model (ColQwen2) and Amazon Nova | [Suman's Notebook (AWS)](https://github.com/debnsuma/fcc-ai-engineering-aws/blob/main/05-multimodal-rag-with-colpali/01-multimodal-retrival-with-colpali-retreve-gen.ipynb) |
| Multi-vector RAG: Using Weaviate to search a collection of PDF documents | [Weaviate's Notebook](https://github.com/weaviate/recipes/blob/main/weaviate-features/multi-vector/multi-vector-colipali-rag.ipynb) |

</details>

<details>
<summary><strong>🔽 Other resources</strong></summary>

- 📝 = blog post
- 📋 = PDF / slides
- 📹 = video

| Title                                                                                    | Author & Link                                                                                                                                                 |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| State of AI report 2024                                                                  | [Nathan's report](https://www.stateof.ai/) 📋                                                                                                                 |
| Technology Radar Volume 31 (October 2024)                                                | [thoughtworks's report](https://www.thoughtworks.com/radar) 📋                                                                                                |
| LlamaIndex Webinar: ColPali - Efficient Document Retrieval with Vision Language Models   | [LlamaIndex's Youtube video](https://youtu.be/nzcBvba7mzI?si=WL9MsyiAFJMyEolz) 📹                                                                             |
| PDF Retrieval with Vision Language Models                                                | [Jo's blog post #1 (Vespa)](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/) 📝                                                          |
| Scaling ColPali to billions of PDFs with Vespa                                           | [Jo's blog post #2 (Vespa)](https://blog.vespa.ai/scaling-colpali-to-billions/) 📝                                                                            |
| Neural Search Talks: ColPali (with Manuel Faysse)                                        | [Zeta Alpha's Podcast](https://open.spotify.com/episode/2s6ljhd6VQTL2mIU9cFzCb) 📹                                                                            |
| Multimodal Document RAG with Llama 3.2 Vision and ColQwen2                               | [Zain's blog post (Together AI)](https://www.together.ai/blog/multimodal-document-rag-with-llama-3-2-vision-and-colqwen2) 📝                                  |
| ColPali: Document Retrieval with Vision Language Models                                  | [Antaripa Saha](https://antaripasaha.notion.site/ColPali-Efficient-Document-Retrieval-with-Vision-Language-Models-10f5314a5639803d94d0d7ac191bb5b1) 📝        |
| Minimalist diagrams explaining ColPali                                                   | [Leonie's ColPali diagrams on X](https://twitter.com/helloiamleonie/status/1839321865195851859)📝                                                            |
| Multimodal RAG with ColPali and Gemini : Financial Report Analysis Application           | [Jaykumaran's blog post (LearnOpenCV)](https://learnopencv.com/multimodal-rag-with-colpali/) 📝                                                               |
| Implement Multimodal RAG with ColPali and Vision Language Model Groq(Llava) and Qwen2-VL | [Plaban's blog post](https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd) 📝 |
| multimodal AI. open-source. in a nutshell.                                               | [Merve's Youtube video](https://youtu.be/IoGaGfU1CIg?si=yEhxMqJYxvMzGyUm) 📹                                                                                  |
| Remove Complexity from Your RAG Applications                                             | [Kyryl's blog post (KOML)](https://kyrylai.com/2024/09/09/remove-complexity-from-your-rag-applications/) 📝                                                   |
| Late interaction & efficient Multi-modal retrievers need more than a vector index        | [Ayush Chaurasia (LanceDB)](https://blog.lancedb.com/late-interaction-efficient-multi-modal-retrievers-need-more-than-just-a-vector-index/) 📝                |
| Optimizing Document Retrieval with ColPali and Qdrant's Binary Quantization              | [Sabrina Aquino (Qdrant)]( https://youtu.be/_A90A-grwIc?si=MS5RV17D6sgirCRm)  📹                                                                              |
| Hands-On Multimodal Retrieval and Interpretability (ColQwen + Vespa)                     | [Antaripa Saha](https://www.analyticsvidhya.com/blog/2024/10/multimodal-retrieval-with-colqwen-vespa/) 📝                                                     |

</details>

## Paper result reproduction

To reproduce the results from the paper, you should checkout to the `v0.1.1` tag or install the corresponding `colpali-engine` package release using:

```bash
pip install colpali-engine==0.1.1
```

## Phase 1: X-PageRerank & EvidenceGraph-RAG (data → cache → train → eval)

This repo contains an *experiment pipeline* for Phase-1 **post-retrieval reranking** on multi-page documents: **X-PageRerank** (page-level GAT graph) and **EvidenceGraph-RAG** (page + spatial regions + typed layout proxies; `--reranker_type region`). Both consume the same cached `RerankDataset` and improve ordering of top-`K` pages from a visual late-interaction retriever (e.g. ColPali).

**End-to-end flow**

| Step | What | Command / module |
| --- | --- | --- |
| 0 | **MP-DocVQA** available locally or via Hub | See *0) Dataset* below |
| 1 | **Cache** embeddings (slow; run once per cache folder) | `python -m experiments.run_phase1 cache ...` |
| 2 | **Splits** (optional doc-disjoint train/val/test) | `python -m experiments.train.make_disjoint_splits` or `run_phase1_multi_seed` |
| 3 | **Train** GAT / region reranker | `python -m experiments.run_phase1 train ...` |
| 4 | **Eval** + metrics | `python -m experiments.run_phase1 eval ...` |
| 5 | **Multi-seed** aggregation | `python -m experiments.run_phase1_multi_seed` → `summarize_phase1_multi_seed` |

### Experiment documentation (`docs/`)

| Doc | Contents |
| --- | --- |
| [docs/evidencegraph_rag_region_experiments.md](docs/evidencegraph_rag_region_experiments.md) | EvidenceGraph-RAG (`region`): hyperparameters, cache/train commands, sweep & multi-seed **result tables** |
| [docs/xpagererank_detailed_paper_multi_seed.md](docs/xpagererank_detailed_paper_multi_seed.md) | X-PageRerank (page GAT): methodology, multi-seed protocol, detailed metrics |
| [docs/experiment_note_2026-03-30_region_multiseed.md](docs/experiment_note_2026-03-30_region_multiseed.md) | Short note: best region config vs GAT baseline (aggregated artifacts) |
| [docs/phase1_experiments_report.md](docs/phase1_experiments_report.md) | Earlier Phase 1 reporting |
| [docs/idea.md](docs/idea.md) | Design notes for the reranking line |

Typical ColPali cache + multi-seed setting: **`vidore/colpali-v1.2`**, **`--top_k 20`**, doc-disjoint splits with seeds **`42–46`**. The **primary** multi-seed command in this repo uses **EvidenceGraph-RAG** (`--reranker_type region`) with **`--graph_preset default`** (see §4A). Some baselines and older notes use **`--graph_preset sparse-graph`** for a sparser page graph; that is an optional variant, not a requirement for the region model.

### 0) Dataset: MP-DocVQA (download / local mirror)

The loader defaults to Hugging Face **`lmms-lab/MP-DocVQA`** (`experiments/data/mpdocvqa_loader.py`). You need network access the first time, or a full local snapshot for **`--offline`** caching.

**Option A — let `datasets` pull the split (online)**

Requires a token if the dataset is gated: set `HUGGING_FACE_HUB_TOKEN` in the environment or in a `.env` file at the repo root (the cache command loads it).

**Option B — download the dataset repo once, then point `--parquet_dir`**

```bash
# Example: full dataset mirror (adjust cache directory as needed)
huggingface-cli download lmms-lab/MP-DocVQA --repo-type dataset

# Path to parquet shards (split name on disk is val-*.parquet for the validation split)
export PARQUET_DIR="$HOME/.cache/huggingface/hub/datasets--lmms-lab--MP-DocVQA/snapshots/<hash>/data/"
```

Use **`--split validation`** (or `val`) with **`MPDocVQADataset.from_parquet_dir`**: the code normalises `validation` → `val` for parquet file names.

### 1) Cache algorithm (what gets written to `cache/...`) — Step 1 of the pipeline

The caching command is implemented in `experiments/run_phase1.py` under the `cache` sub-command. Conceptually, for each dataset sample (question + document pages):

1. Encode the question into multi-vector query embeddings `q_embs`.
2. Encode the pages into multi-vector page embeddings `p_embs`.
3. Compute **stage-1 scores** between `q_embs` and each page in the document (`stage1_scores`).
4. Sort pages by stage-1 scores and keep only top `K` pages (`top_k`).
5. Build a binary **support mask** over the kept pages:
   - `support_mask[i] = 1` if the i-th kept page is a gold support page,
   - `support_mask[i] = 0` otherwise.
6. Save everything to disk via `RerankDataset.save()`:
   - `meta.json`: list of `{question_id, doc_id, page_numbers}` for each cached sample
   - `*_q.pt`: query embeddings
   - `*_p.pt`: page embeddings for the top-K candidate pages
   - `*_s0.pt`: stage-1 scores for the top-K pages
   - `*_mask.pt`: support mask over the top-K pages

> Important: `train` later relies on `meta.json`. If your cache was interrupted, you may end up with missing/empty `meta.json`.

### 2) Train algorithm (GAT reranker and losses)

Training is implemented in `experiments/run_phase1.py` under the `train` sub-command, and the actual model/loss logic is in `experiments/train/train_reranker.py`.

Per epoch:

1. Load cached samples with `RerankDataset.load()`.
2. For each query sample:
   - Build an **evidence graph** on the query-time top-K candidate pages: page-only path uses `build_evidence_graph_from_retrieval` + `PageGATReranker`; with `--reranker_type region`, the model is `RegionGraphReranker` (extra region + typed layout nodes; logits still align to the **K pages**).
   - Run the reranker to predict reranking scores.
3. Optimize a combination of losses:
   - **Listwise loss** (softmax over candidate scores; positives are the support pages indicated by `support_mask`)
   - **Pairwise margin loss** (hinge-style ranking loss between positive support pages and negative pages, using a margin)

The early stopping criterion is **validation `Recall@5`** (patience on `Recall@5`).

### 3) Evaluation metrics (exact definitions used in this code)

Evaluation uses `experiments/eval/eval_retrieval.py` and compares:

- `predictions`: ranked list of *page numbers* per query (top-k from stage-1 or reranker scores)
- `ground_truth`: list of gold *support page numbers* per query, derived as:
  - `gold = [sample.page_numbers[i] for i, v in enumerate(sample.support_mask.tolist()) if v > 0]`

Metrics:

1. **Recall@k** (`recall_at_k` in `experiments/eval/eval_retrieval.py`)
   - For each query, it is a hit if **any** gold support page is present in `pred[:k]`.
   - `Recall@k = (#queries with at least one hit) / (#queries)`

2. **MRR@10** (`mrr_at_k`)
   - For each query, find the rank of the **first** relevant (gold support) page within top-10.
   - Reciprocal rank is `1 / rank_first_hit` (or 0 if none in top-10).
   - `MRR@10 = average(rec_rank across queries)`

3. **nDCG@10** (`ndcg_at_k`)
   - Binary relevance: `rel = 1` if a predicted page is in gold support set, else 0.
   - `DCG@k = sum_{i=1..k} rel_i / log2(i+1)`
   - `IDCG@k` uses an ideal ranking with `min(|gold|, k)` ones.
   - `nDCG@k = DCG / IDCG`

### 4) Reproducible commands

#### A) **Step 1 — Cache (paper-style experiments): ColPali + `top_k=20`**

This matches the setup documented in [docs/evidencegraph_rag_region_experiments.md](docs/evidencegraph_rag_region_experiments.md). Training expects a **single** `RerankDataset` directory (e.g. all labeled `val` rows); doc-disjoint **train/val/test** slices are created later inside each `seed_*` by `run_phase1_multi_seed`.

**Online (Hugging Face loads `lmms-lab/MP-DocVQA`):**

```bash
python -m experiments.run_phase1 cache \
  --split validation \
  --model_name vidore/colpali-v1.2 \
  --model_type colpali \
  --output_dir cache/mpdoc_val_full \
  --top_k 20 \
  --batch_size 4
```

**Offline (local parquet mirror — set `<hash>` after `huggingface-cli download`):**

```bash
export PARQUET_DIR="${HOME}/.cache/huggingface/hub/datasets--lmms-lab--MP-DocVQA/snapshots/<hash>/data/"
python -m experiments.run_phase1 cache \
  --split validation \
  --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colpali-v1.2 \
  --model_type colpali \
  --output_dir cache/mpdoc_val_full \
  --top_k 20 \
  --batch_size 4
```

For **low-RAM** incremental writes, use `--cache_chunk_rows` with `--parquet_dir` and an optional row window (`--num_samples`, `--sample_offset`); see `python -m experiments.run_phase1 cache --help`.

After `cache/mpdoc_val_full` exists and contains `meta.json`, run **multi-seed** train+eval (creates `seed_*/splits`, checkpoints, `seed_*/eval/phase1_results.json`).

**Main command — EvidenceGraph-RAG (`region`)** (page + region + typed nodes; this is the new method, not page-only GAT):

```bash
python3 -m experiments.run_phase1_multi_seed \
  --candidate_cache cache/mpdoc_val_full \
  --root multi_seed_runs/mpdoc_val_disjoint_region_best \
  --seeds 42,43,44,45,46 \
  --graph_preset default \
  --reranker_type region \
  --num_epochs 20 \
  --train_batch_size 16 \
  --lr 5e-5 \
  --top_k 20 \
  --eval_k 10 \
  --grid_rows 2 \
  --grid_cols 2 \
  --sem_threshold_region 0.70 \
  --lambda_mix_start 0.15 \
  --lambda_mix_end 0.55 \
  --lambda_mix_warmup_steps 1200 \
  --eval_dir_name eval
```

Add `--cross_page_region_edges` if you want cross-page region–region edges (slower). For **X-PageRerank (page GAT only)** on the same cache and seeds:

```bash
python3 -m experiments.run_phase1_multi_seed \
  --candidate_cache cache/mpdoc_val_full \
  --root multi_seed_runs/mpdoc_val_disjoint \
  --graph_preset sparse-graph \
  --reranker_type page \
  --seeds 42,43,44,45,46
```

(You can also put `--graph_preset sparse-graph` on the **region** run as an ablation; **`default`** is what the command above uses.)

**Aggregate metrics** (adjust `--glob` / paths to match `--root` and `--eval_dir_name`):

```bash
python3 -m experiments.summarize_phase1_multi_seed \
  --glob 'multi_seed_runs/mpdoc_val_disjoint_region_best/seed_*/eval/phase1_results.json' \
  --out_json multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_results.json \
  --out_md multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_report.md
```

#### B) **Lightweight offline slices: ColSmol (smaller GPU / quick tests)**

Set `PARQUET_DIR` to the directory that contains local parquet shards (Hub cache or any local export). Because offline mode can miss a labeled `train` split, this path usually builds both train and val from the labeled **`val`** parquet split.

```bash
export PARQUET_DIR="${HOME}/.cache/huggingface/hub/datasets--lmms-lab--MP-DocVQA/snapshots/<hash>/data/"
```

- **Row-only split** (`sample_offset` / `num_samples`): train and val do not share the same *rows*, but they can still share the same **`doc_id`**.
- **Document-disjoint val (recommended):** pass `--exclude_train_doc_ids_cache` when building val. The script **drops candidate rows before encoding** if `doc_id` appears in the train cache.

**Cache train slice (example: rows 0..199)**

```bash
rm -rf cache/real_train_split
python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/real_train_split \
  --top_k 10 --batch_size 4 \
  --num_samples 200 --sample_offset 0
```

**Cache val (document-disjoint from train — widen the window so enough rows survive filtering)**

```bash
rm -rf cache/real_val_doc_disjoint
python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/real_val_doc_disjoint \
  --top_k 10 --batch_size 4 \
  --num_samples 600 --sample_offset 200 \
  --exclude_train_doc_ids_cache cache/real_train_split
```

Check the log line `Doc-disjoint filter: kept X/Y samples`; increase `--num_samples` if `X` is too small.

**Train page GAT reranker**

```bash
python -m experiments.run_phase1 train \
  --train_cache cache/real_train_split \
  --val_cache   cache/real_val_doc_disjoint \
  --output_dir  checkpoints/gat_real_split \
  --num_epochs 10 --batch_size 16 --lr 5e-4 \
  --graph_preset sparse-graph \
  --reranker_type page
```

**Train EvidenceGraph-RAG (`region`)** — same caches, add region flags (see [docs/evidencegraph_rag_region_experiments.md](docs/evidencegraph_rag_region_experiments.md)):

```bash
python -m experiments.run_phase1 train \
  --train_cache cache/real_train_split \
  --val_cache   cache/real_val_doc_disjoint \
  --output_dir  checkpoints/region_real_split \
  --num_epochs 10 --batch_size 32 --lr 1e-4 \
  --graph_preset sparse-graph \
  --reranker_type region \
  --grid_rows 2 --grid_cols 2 \
  --sem_threshold_region 0.70
```

**Eval on val cache**

```bash
python -m experiments.run_phase1 eval \
  --eval_cache cache/real_val_doc_disjoint \
  --checkpoint checkpoints/gat_real_split/best.pt \
  --output_dir results/phase1_doc_disjoint_val \
  --graph_preset sparse-graph \
  --reranker_type page
```

For a region checkpoint, pass `--reranker_type region` and the same grid / `--sem_threshold_region` / optional `--cross_page_region_edges` as in training.

### 5) Strict “no document overlap” test (recommended)

If you want **test** documents that do not appear in train **or** val, filter a candidate test pool by `doc_id`.

1. Build a candidate test pool (example: more rows than needed):
```bash
rm -rf cache/real_test_pool
python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/real_test_pool \
  --top_k 10 --batch_size 4 \
  --num_samples 300 --sample_offset 300
```

2. Filter out any `doc_id` present in train/val without re-encoding embeddings:
```bash
rm -rf cache/real_test_split_no_doc_overlap
python -m experiments.train.filter_rerank_cache \
  --candidate_cache cache/real_test_pool \
  --exclude_cache cache/real_train_split \
  --exclude_cache cache/real_val_doc_disjoint \
  --output_dir cache/real_test_split_no_doc_overlap
```

3. Eval on the filtered cache:
```bash
python -m experiments.run_phase1 eval \
  --eval_cache cache/real_test_split_no_doc_overlap \
  --checkpoint checkpoints/gat_real_split/best.pt \
  --output_dir results/test_real_no_doc_overlap
```

### 6) Example results from this workspace (Phase 1, row-split val — not document-disjoint)

For **doc-disjoint multi-seed** ColPali runs (X-PageRerank vs EvidenceGraph-RAG), see the tables in [docs/evidencegraph_rag_region_experiments.md](docs/evidencegraph_rag_region_experiments.md) and [docs/xpagererank_detailed_paper_multi_seed.md](docs/xpagererank_detailed_paper_multi_seed.md).

The numbers below used `cache/real_val_split` built with **row offset only** (train and val could share `doc_id`).

On:
- `cache/real_train_split` = 190 samples
- `cache/real_val_split` = 78 samples

Training:
- early stopping at epoch 8
- best checkpoint with `Best Recall@5 = 0.9359`

Evaluation (`Recall@1`, `Recall@5`, `MRR@10`, `nDCG@10`):

```text
ColPali (stage-1):             Recall@1=0.3590, Recall@5=0.8718, MRR@10=0.5840, nDCG@10=0.6852
ColPali + MLP reranker:       Recall@1=0.4615, Recall@5=0.9231, MRR@10=0.6454, nDCG@10=0.7323
X-PageRerank (ablation no-graph): Recall@1=0.5769, Recall@5=0.9359, MRR@10=0.7322, nDCG@10=0.7980
X-PageRerank (GAT):          Recall@1=0.5897, Recall@5=0.9359, MRR@10=0.7377, nDCG@10=0.8017
```

Interpretation:
- Recall@5 saturates, but reranking improves ranking quality in deeper ranks (`MRR@10`, `nDCG@10`).

Re-run train/eval with `cache/real_val_doc_disjoint` for **document-level** generalization.

## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  

Authors: **Manuel Faysse**\*, **Hugues Sibille**\*, **Tony Wu**\*, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (\* denotes equal contribution)

```latex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}

@misc{macé2025vidorebenchmarkv2raising,
      title={ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval}, 
      author={Quentin Macé and António Loison and Manuel Faysse},
      year={2025},
      eprint={2505.17166},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.17166}, 
}
```

python3 -m experiments.run_phase1_multi_seed \
  --candidate_cache cache/mpdoc_val_full \
  --root multi_seed_runs/mpdoc_val_disjoint_region_best \
  --seeds 42,43,44,45,46 \
  --graph_preset default \
  --reranker_type region \
  --num_epochs 20 \
  --train_batch_size 16 \
  --lr 5e-5 \
  --top_k 20 \
  --eval_k 10 \
  --grid_rows 2 \
  --grid_cols 2 \
  --sem_threshold_region 0.70 \
  --lambda_mix_start 0.15 \
  --lambda_mix_end 0.55 \
  --lambda_mix_warmup_steps 1200 \
  --eval_dir_name eval
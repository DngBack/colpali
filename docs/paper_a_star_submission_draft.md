# EvidenceGraph-RAG: Query-Conditioned Region Graph Reranking for Multi-Page Document QA

## Abstract

Multi-page document question answering requires retrieving support pages that may contain evidence spread across regions, figures, tables, captions, and neighboring pages. In practical retrieval pipelines, a strong stage-1 retriever still ranks pages mostly independently, which makes it difficult to exploit evidence relations inside a document. We present **EvidenceGraph-RAG**, a two-stage reranking framework that keeps stage-1 retrieval cached and trains a lightweight graph reranker over the top-k candidates. The reranker builds a query-conditioned evidence graph with page nodes, region nodes, heuristic typed evidence nodes, and a query node, then performs multi-head graph attention to refine page ordering.

On MP-DocVQA-derived cached candidates with document-disjoint multi-seed evaluation (`seed = 42..46`), the current best region setting improves over `ColPali (stage-1)` by **+6.23 Recall@1**, **+7.26 Recall@5**, **+7.01 MRR@10**, and **+5.53 nDCG@10**. The results show that structured region-aware reranking can substantially improve early-rank retrieval quality while preserving the deployment-friendly top-k cached retrieval pipeline.

This draft is intentionally written as a paper-ready technical report: the method is specified at implementation level, the experiment protocol is reproducible from the repository command line, and remaining gaps for an A* submission are called out explicitly.

## 1. Introduction

Retrieval is the first bottleneck in document QA systems. In multi-page settings, the answer may be supported by one page, but the evidence often depends on local layout cues, nearby pages, or a combination of text blocks, captions, tables, and figures. Late-interaction retrievers such as ColPali are strong at matching query-page similarity, but they still treat retrieved pages largely independently after the first ranking step.

EvidenceGraph-RAG addresses this gap with a practical reranking layer:

1. Stage-1 retrieval is computed once and cached.
2. A query-conditioned graph is constructed over the top-k pages.
3. Graph attention propagates information between pages, subpage regions, and typed evidence proxies.
4. The final score interpolates the original retrieval score with a learned graph correction.

The goal is not to replace stage-1 retrieval, but to improve the order of the top candidates that matter most for downstream QA. This makes the method easy to deploy in a real system where corpus-wide indexing and repeated encoder passes are expensive.

### Contributions

1. A query-conditioned region graph reranker that extends page-level reranking with region and typed evidence nodes.
2. A reproducible training protocol for cached multi-vector retrieval embeddings with document-disjoint multi-seed evaluation.
3. A detailed experimental report showing consistent gains over stage-1 and MLP reranking on early-rank retrieval metrics.
4. A submission-oriented checklist that identifies the remaining experiments needed for a stronger paper claim.

## 2. Related Work

This draft should position the method relative to four threads of work:

1. **Late-interaction document retrieval**: multi-vector retrievers that score query-page pairs using patch/token-level similarity.
2. **Graph neural reranking**: rerankers that use graph structure to propagate context among candidate items.
3. **Evidence-centric document QA**: methods that explicitly model support pages, evidence spans, or evidence graphs.
4. **Layout-aware multimodal retrieval**: approaches that exploit page layout, region structure, or spatial substructure.

For the final paper, this section should include formal citations and a careful comparison table. In this draft, the main claim is narrower: a graph built over candidate pages and page regions improves early ranking in a deployable post-retrieval setting.

## 3. Method

### 3.1 Problem formulation

Given a query `q` and a document with candidate pages `{p_1, ..., p_K}` retrieved by a stage-1 retriever, the reranker learns a score `s_i` for each page such that support pages are ranked above non-support pages.

The input to stage-2 is not raw pages, but cached multi-vector embeddings:

- `query_embs`: multi-vector embedding of the query
- `page_embs`: multi-vector embeddings of the top-k candidate pages
- `stage1_scores`: original retrieval scores from ColPali
- `support_mask`: binary labels for support pages

The output is a reordered list of the same top-k pages.

### 3.2 Stage-1 cache

The repository caches stage-1 retrieval outputs in `RerankDataset` samples. Each sample contains:

- `*_q.pt`: query embeddings
- `*_p.pt`: page embeddings
- `*_s0.pt`: stage-1 scores
- `*_mask.pt`: support labels
- `meta.json`: question and document metadata

This design is important for reproducibility and efficiency because the expensive encoder pass is not repeated during reranker training.

### 3.3 Evidence graph construction

EvidenceGraph-RAG builds a graph from the top-k candidates. In the current best region setting, the graph contains:

- page nodes
- region nodes
- typed evidence nodes
- a query node

#### Page nodes

Each page embedding is obtained by mean pooling the multi-vector page representation:

`page_vector = mean(page_embs, dim=1)`

This gives a single vector per candidate page for page-level graph reasoning.

#### Region nodes

Each page is split into a fixed `grid_rows × grid_cols` grid over the patch sequence. With the current best setting, `grid_rows = 2` and `grid_cols = 2`, so each page yields 4 region nodes.

Region vectors are computed by mean pooling the patch embeddings that fall inside each grid cell. This is a lightweight proxy for region-level evidence aggregation that preserves spatial locality without requiring a full detector or OCR pipeline.

#### Typed evidence nodes

The implementation also introduces heuristic typed evidence nodes:

- header
- text block
- caption
- table region

These are extracted from fixed bands of the page patch grid. They are not semantic layout annotations from a parser; they are a cheap structural proxy that helps the reranker learn layout-sensitive message passing.

#### Query node

The query node conditions the graph on the user question and provides a global anchor for message passing.

#### Edges

The graph combines several edge types:

1. **Page semantic edges**: connect pages with cosine similarity above a threshold.
2. **Page adjacency edges**: connect pages with page-distance `<= 1`.
3. **Page-region edges**: connect each page to its own regions.
4. **Region-region edges**: connect regions within the same page, with stronger same-row / same-column links.
5. **Typed-evidence edges**: connect page to typed nodes and connect typed nodes with strong intra-page structure links.
6. **Cross-page typed edges**: connect same typed evidence across neighboring pages.
7. **Query edges**: connect the query node to all evidence nodes.

In code, the graph is normalized row-wise after self-loops are added, so each node aggregates a distribution over its neighbors.

### 3.4 Graph encoder

The graph encoder is a compact multi-head GAT:

1. Project input vectors to a hidden space.
2. Add a learned type embedding for page, region, typed evidence, and query nodes.
3. Apply stacked dense GAT layers.
4. Use a residual projection to preserve the original representation.

The implementation is intentionally simple and self-contained, which keeps it easy to inspect and debug.

### 3.5 Page scoring head

After message passing, the model keeps only page-node outputs for the final ranking. The query node representation is paired with each page representation, and a small MLP produces a page-level delta score:

`delta_i = MLP([h_page_i || h_query])`

The final ranking score interpolates the normalized stage-1 score and the learned delta:

`final_i = (1 - lambda) * normalize(s0_i) + lambda * delta_i`

where `lambda = sigmoid(lambda_mix)`.

This interpolation is important because it stabilizes training and prevents the graph model from completely discarding the strong stage-1 prior.

### 3.6 Lambda warmup

`lambda_mix` is not fixed. It is scheduled from `lambda_mix_start` to `lambda_mix_end` over `lambda_mix_warmup_steps`. In the current best run:

- start: `0.15`
- end: `0.55`
- warmup: `1200` steps

This schedule lets the model start close to stage-1 behavior and gradually trust the graph correction more as training progresses.

### 3.7 Training objective

The reranker is trained with a mixture of ranking losses:

1. **Listwise softmax loss** over the top-k pages, treating support pages as positives.
2. **Pairwise margin loss** between positive and negative pages.

The default combination in the trainer is:

`L = 0.7 * L_listwise + 0.3 * L_pairwise`

Optimization uses AdamW, cosine annealing, gradient clipping, and early stopping on validation `Recall@5`.

## 4. Experimental Setup

### 4.1 Data and split protocol

The current paper draft uses the cached MP-DocVQA candidate set:

- `--candidate_cache cache/mpdoc_val_full`
- multi-seed root: `multi_seed_runs/mpdoc_val_disjoint_region_best`
- seeds: `42, 43, 44, 45, 46`

The split protocol is document-disjoint. The default split ratios in the multi-seed runner are:

- train: `12%`
- val: `8%`
- test: `80%`

This is a hard split at the document level, so candidate pages from the same document do not cross split boundaries.

### 4.2 Hyperparameters for the current best region run

The current best region configuration used:

- `reranker_type=region`
- `grid_rows=2`
- `grid_cols=2`
- `sem_threshold_region=0.70`
- `cross_page_region_edges=off`
- `num_epochs=20`
- `train_batch_size=16`
- `lr=5e-5`
- `top_k=20`
- `eval_k=10`
- `lambda_mix_start=0.15`
- `lambda_mix_end=0.55`
- `lambda_mix_warmup_steps=1200`

The exact command is:

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

### 4.3 Metrics

The main retrieval metrics are:

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR@10`
- `nDCG@10`

The support-page metrics are:

- `Coverage@5`
- `Coverage@10`
- `AllSupportHit@10`
- `SupportF1@10`

### 4.4 Baselines

The current on-disk multi-seed summary reports:

- `ColPali (stage-1)`
- `ColPali + MLP reranker`
- `EvidenceGraph-RAG (page+region)`

For the final paper, a page-level GAT baseline (`X-PageRerank`) should also be included under the exact same split family to make the comparison stronger.

## 5. Results

Source: `multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_results_latest.json`

### 5.1 Retrieval metrics

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5415 ± 0.0088 | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.6899 ± 0.0056 | 0.7644 ± 0.0042 |
| ColPali + MLP reranker | 0.5371 ± 0.0119 | 0.8843 ± 0.0109 | 1.0000 ± 0.0000 | 0.6871 ± 0.0095 | 0.7624 ± 0.0074 |
| **EvidenceGraph-RAG (page+region)** | **0.6038 ± 0.0382** | **0.9534 ± 0.0324** | **1.0000 ± 0.0000** | **0.7600 ± 0.0403** | **0.8198 ± 0.0315** |

### 5.2 Support-page metrics

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| ColPali + MLP reranker | 0.8843 ± 0.0109 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| **EvidenceGraph-RAG (page+region)** | **0.9534 ± 0.0324** | **1.0000 ± 0.0000** | **0.0000 ± 0.0000** | **0.6102 ± 0.0058** |

### 5.3 Main deltas against stage-1

- `Recall@1`: `+0.0623`
- `Recall@5`: `+0.0726`
- `MRR@10`: `+0.0701`
- `nDCG@10`: `+0.0553`

### 5.4 Interpretation

The strongest and most consistent gains are at the top of the ranking. This is the most meaningful region for a downstream QA pipeline, because only the first few pages are typically passed to a reader or answer extractor.

The support-page metrics show a similar trend: EvidenceGraph-RAG increases the chance that at least one support page appears early in the ranking, but the strict multi-support metric remains limited in this evaluation regime.

## 6. Discussion

### 6.1 Why the method helps

EvidenceGraph-RAG improves retrieval because it can propagate evidence across multiple granularities:

1. **Page level** captures coarse document relevance.
2. **Region level** preserves local visual structure inside a page.
3. **Typed evidence** injects a simple structural prior for headers, captions, text blocks, and table-like zones.
4. **Query node** keeps the reranking query-conditioned.

This combination lets the model correct cases where a support page is semantically close to the query but needs region-level context to be ranked correctly.

### 6.2 Why stage-1 interpolation matters

The learned graph score is not used in isolation. The model blends graph correction with the original stage-1 score. This reduces the risk that the reranker destroys a good retrieval prior, which is especially useful when the top-k candidate list already contains many strong pages.

### 6.3 What the current metrics do and do not show

The current results support a claim about **early-rank retrieval quality**. They do not yet support a broader claim about end-to-end answering accuracy, because the repository currently reports retrieval and support-page metrics rather than final answer EM/F1.

That distinction should stay explicit in the paper.

### 6.4 Metric caveat

`AllSupportHit@10` remains `0.0` in the current summary. That means the benchmark composition is not yet ideal for strict multi-support claims, so the paper should avoid overemphasizing this metric. The more reliable story is the improvement in `Recall@1`, `Recall@5`, `MRR@10`, and `nDCG@10`.

## 7. Limitations

1. **No end-to-end QA metric yet**: the current repo evidence is retrieval-centric, not answer-centric.
2. **No final page-GAT comparison in the latest summary**: the paper should include a same-protocol page-level GAT baseline before submission.
3. **Typed evidence is heuristic**: the typed nodes are simple layout proxies, not parsed semantics.
4. **The graph is candidate-local**: the method reasons over top-k candidates only, so it cannot recover evidence outside the retrieved set.
5. **Single-support bias**: strict all-support metrics are not informative under the current data composition.

## 8. What is still needed for a strong submission

### 8.1 Must-have

1. Re-run the page-level GAT baseline under the same split family and report it alongside EvidenceGraph-RAG.
2. Add statistical significance tests for `Recall@1`, `Recall@5`, `MRR@10`, and `nDCG@10`.
3. Report the exact environment, GPU, seed, and commit hash used for the final run.
4. Add a short ablation table for:
   - region nodes off
   - typed nodes off
   - cross-page region edges on/off
   - lambda schedule fixed vs warmup

### 8.2 Strongly recommended

1. Run a second split regime to test robustness beyond `12/8/80`.
2. Compare `2x2` and `3x3` region grids.
3. Measure latency and memory overhead of the reranker.
4. Add qualitative case studies showing pages before and after reranking.

### 8.3 Nice-to-have

1. End-to-end QA metrics.
2. A second benchmark dataset.
3. Failure taxonomy and calibration analysis.

## 9. Suggested Main-Paper Structure

1. Introduction
2. Related Work
3. Method
4. Experimental Setup
5. Results
6. Ablation and Analysis
7. Limitations
8. Conclusion

## 10. Conclusion

EvidenceGraph-RAG shows that a lightweight region-aware evidence graph can improve early-rank retrieval over a strong cached retriever baseline. The method is practical because it operates only on top-k candidates, preserves the original stage-1 score through interpolation, and adds structure where it matters most: within pages, across neighboring pages, and around typed evidence proxies.

In its current state, the method is already strong enough for a detailed technical paper draft. To make the submission truly convincing, the next priority is not a new architecture, but stronger experimental support: same-protocol GAT comparison, significance testing, ablations, and a downstream QA metric.


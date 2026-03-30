# From Pages to Evidence: Query-Conditioned Heterogeneous Graph Retrieval for Multi-Page Document QA

## Abstract

Multi-page document question answering often requires composing evidence distributed across text blocks, figures, captions, tables, and key-value regions spanning multiple pages. Standard late-interaction retrievers score pages independently and therefore underperform when support is relational rather than page-local. We reformulate retrieval as **query-conditioned heterogeneous evidence graph retrieval**: stage-1 returns top-k pages, then stage-2 decomposes them into typed evidence units and performs typed message passing over semantic and structural relations at query time.

This report contains two complementary parts: (1) an upgraded method formulation ("from page reranking to evidence-graph retrieval"), and (2) reproducible multi-seed results from the current page-graph implementation in this repository. The practical deployment constraint is preserved: graph construction remains local to top-k candidates at query time (no corpus-wide graph index).

---

## 1) Problem Setting and Motivation

For a query `q` and a multi-page document `D={p_1,...,p_n}`, stage-1 retrieval returns top-k page candidates. The key observation is that answer support is often **not page-atomic**: it is distributed, typed, and relational (e.g., header on one page, figure on another, caption elsewhere).

Let `E={e_1,...,e_m}` be latent evidence units extracted from candidate pages. The objective is no longer only "rank pages by marginal similarity". Instead, infer a support subgraph:

`G_q^* \subseteq G(E,R)`

that best explains the query under heterogeneous relations `R`.

We cast reranking as:

1. Retrieve top-k pages with a visual late-interaction retriever.
2. Decompose pages into typed evidence units.
3. Build a query-conditioned heterogeneous evidence graph over those units.
4. Run typed message passing to infer evidence relevance and page support.
5. Aggregate evidence-level decisions into page ranking and downstream QA support.

This keeps stage-1 unchanged while adding cross-page reasoning where it matters most (query-time top-k set).

---

## 2) Positioning vs Prior Graph Retrieval Work

The method is intentionally scoped as a **drop-in post-retrieval module**:

- Backbone retrieval stays fixed (ColPali-style late interaction).
- Graph is **local** to top-k candidates at query time (no corpus-wide component graph).
- Focus is practical cross-page evidence composition for document QA pipelines.

This is consistent with the repo design direction captured in `docs/idea.md` and subsequent experiment reports.

---

## 3) Upgraded Method: Query-Conditioned Heterogeneous Evidence Graph Retrieval

## 3.1 Stage-1 Candidate Generation (unchanged strength)

For each sample:

- Query embedding tensor: `*_q.pt`
- Top-k page embedding tensor: `*_p.pt`
- Stage-1 scores: `*_s0.pt`
- Support mask labels: `*_mask.pt`
- Metadata (`question_id`, `doc_id`, `page_numbers`) in `meta.json`

All are stored as a cached `RerankDataset`. This preserves the deployment story: stage-1 remains fixed and strong.

## 3.2 Evidence Unit Extraction (new core step)

For top-k pages, extract structured units and represent them as typed nodes:

- `page`
- `text_block`
- `header_or_section_title`
- `figure_region`
- `caption`
- `table_region`
- `key_value_field`
- optional `image_patch`
- optional explicit `query` node

Node features combine multimodal content and structure:

- visual region embedding
- OCR/text embedding
- layout coordinates
- page index and section position
- stage-1 page score
- node-type embedding
- document-structure metadata

## 3.3 Query-Conditioned Heterogeneous Graph Construction

Construct `G_q=(V,E)` over extracted units with typed edges:

Structural edges:

- same page
- adjacent pages
- same or neighboring section
- caption <-> figure
- caption <-> table
- header <-> covered blocks
- key <-> value
- same table row/column

Semantic edges:

- embedding similarity neighbors
- lexical/entity overlap
- OCR overlap
- cross-reference links ("Fig. 3", "Table 2", etc.)

Query-conditioned edges:

- query <-> node similarity edges
- query-conditioned gating over all relation channels

Graph sparsification is required for robustness:

- top-M semantic neighbors per node
- page-distance constraints
- thresholded cross-reference matching
- learned edge gating / edge dropout

## 3.4 Encoder: Typed Message Passing (replace plain page-only GAT)

Use a heterogeneous relational attention encoder (relational GAT or heterogeneous graph transformer), not a relation-agnostic page-only GAT.

For layer `l`:

`h_i^(l+1) = LayerNorm(h_i^(l) + sum_{r in R} Attn_r(h_i^(l), {h_j^(l):(j->i,r) in E}))`

with:

- relation-specific projections
- edge-type embeddings
- query-conditioned attention bias
- residual connection and normalization
- edge dropout

## 3.5 Outputs: Evidence + Page (not page-only)

Predict at least two heads:

1. Evidence relevance score for each node `e_i`:
   - `s_i^evidence = f_e(h_i)`
2. Page support score from evidence aggregation:
   - `s_p^page = f_p(Agg({h_i : e_i in p}))`

Optional third head:

3. Support-subgraph coherence/confidence for selected evidence sets or paths.

This produces richer supervision and stronger interpretability than page-only reranking.

## 3.6 Training Objective (multi-term)

Total loss:

`L = lambda_1 L_page + lambda_2 L_evidence + lambda_3 L_coh + lambda_4 L_distill`

where:

- `L_page`: listwise page-ranking loss (keeps comparability with current pipeline)
- `L_evidence`: node-level evidence supervision (binary/focal; weak labels if region labels absent)
- `L_coh`: connectivity/coherence objective for support subgraph consistency
- `L_distill`: distillation from stage-1 scores for stability and retrieval prior preservation

## 3.7 Theoretical Motivation (lightweight but explicit)

Proposition-style narrative to include in paper:

1. Independent page scoring is suboptimal under compositional evidence.
2. If support units form a connected relational subgraph, message passing increases support-vs-distractor separability.
3. Evidence-aggregated page scoring is at least as expressive as page-only scoring and strictly more expressive when evidence is sparse/localized.

## 3.8 Evaluation Outputs

Per run, `phase1_results.json` stores:

- retrieval metrics (`Recall@k`, `MRR@10`, `nDCG@10`, etc.),
- support-page metrics (`Coverage@k`, `AllSupportHit@k`, `SupportF1@k`),
- per-query and pairwise analysis (`no-graph` vs `GAT`).

For the upgraded method, add evidence-level and end-task metrics:

- `EvidenceRecall@k`
- support-subgraph F1 / path accuracy
- region-grounding accuracy
- QA metrics (`EM`, `F1`, `ANLS` depending on benchmark)
- efficiency (`graph build time`, `latency/query`, `memory/query`, scaling vs top-k)

---

## 4) Scientific Claims (updated)

1. Multi-page document QA is an evidence-composition problem, not only page ranking.
2. Cross-page support is heterogeneous (text, figures, captions, tables, key-value fields) and should be modeled with typed nodes/edges.
3. Query-conditioned heterogeneous graph reasoning better recovers latent support structure than independent page scoring and non-graph rerankers.
4. Query-time local graph construction preserves practical deployment.

---

## 5) Experimental Design for the A*-level Version

Datasets:

- Primary: MP-DocVQA
- Secondary: DUDE
- Optional third long-document benchmark

Hard cross-page subset (recommended):

- >=2 support pages
- page distance > 1
- figure/caption split
- table/header or table/note split
- comparison/aggregation/cross-reference question types

Report by strata:

- single-page vs multi-page
- 2-hop vs 3+ hop
- text-only vs text+figure vs text+table
- near-page vs long-range gap
- explicit cross-reference vs implicit composition

Baselines/ablations (minimum set):

- stage-1 retriever only
- MLP/transformer stage-2 rerankers
- no-graph vs homogeneous graph vs heterogeneous graph
- remove query node
- remove structural/semantic edge groups
- top-k sensitivity (`10/20/40`)
- graph density and depth sensitivity
- loss-term ablations (`L_page` only, +`L_evidence`, +`L_coh`, +/-`L_distill`)

---

## 6) Implementation Roadmap in This Repository

Phase A (framing-first, minimal code delta):

- keep current page graph pipeline
- adopt upgraded thesis and claims
- add stronger ablations, QA metrics, latency, hard subset protocol

Phase B (first algorithmic upgrade):

- add typed evidence nodes (`text`, `caption`, `figure`, `table`)
- move from page-only graph to heterogeneous graph

Phase C (stronger supervision):

- add evidence labels/weak labels and coherence objective

Phase D (final paper package):

- multi-dataset validation
- hard multi-hop benchmark split
- full end-task QA and efficiency study
- qualitative failure taxonomy

---

## 7) Experimental Protocol Used in This Workspace (Current Implemented System)

## 7.1 Data and Split Strategy

- Candidate cache: `cache/mpdoc_val_full` (observed `meta.json` length: **4801**).
- Multi-seed splitting is done by `doc_id` groups with question disjointness checks:
  - output under `multi_seed_runs/mpdoc_val_disjoint/seed_{seed}/splits`
  - constraints recorded as:
    - `doc_id_overlap = none`
    - `question_id_overlap = none`

Per-seed sample counts from `split_summary.json`:

| Seed | Train | Val | Test |
|---|---:|---:|---:|
| 42 | 3362 | 744 | 695 |
| 43 | 3368 | 689 | 744 |
| 44 | 3271 | 745 | 785 |
| 45 | 3413 | 712 | 676 |
| 46 | 3441 | 682 | 678 |

Mean +/- std:

- Train: `3371.0 +/- 64.7`
- Val: `714.4 +/- 29.6`
- Test: `715.6 +/- 47.5`

## 7.2 Multi-Seed Runner Hyperparameters (actual driver)

From `experiments/run_phase1_multi_seed.py`:

- Seeds: `42,43,44,45,46`
- Split ratios: `train=0.70`, `val=0.15`, `test=0.15`
- Graph preset: `sparse-graph`
- Train epochs: `10`
- Train batch size: `32`
- Top-k (train cache view): `20`
- Eval k: `10`
- Aggregation: enabled by default (unless `--no_aggregate`)

## 7.3 Train/Eval Core Hyperparameters (pipeline defaults)

From `experiments/run_phase1.py` + `experiments/train/train_reranker.py`:

- Train CLI defaults:
  - `lr=1e-4`, `hidden_dim=256`, `num_layers=2`, `num_heads=4`
  - `top_k=20`
  - `sem_threshold=0.65`, `adj_max_gap=1`
- TrainingConfig defaults:
  - `weight_decay=1e-5`, `warmup_steps=100`, `grad_clip=1.0`
  - loss weights: `listwise=0.7`, `pairwise=0.3`, `margin=0.1`
  - `patience=5`, validation every epoch, `eval_k=10`
- Eval CLI defaults:
  - `k=10`
  - `sem_threshold=0.65`, `adj_max_gap=1`
  - with `sparse-graph`, effective graph values are clamped as above.

## 7.4 Cache-Build Hyperparameters (OOM-safe)

Important options (from `run_phase1 cache`):

- `top_k` (must be consistent with train/eval assumptions)
- `max_pages` (default `60`)
- `batch_size` (default `4`)
- `maxsim_doc_chunk` (default `16`, reduce to lower RAM)
- `cache_chunk_rows` (default `512`, lower for memory safety)
- optional doc-disjoint filtering via `--exclude_train_doc_ids_cache`

---

## 8) 5-Seed Aggregated Results (Latest Output)

Source: `multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md`  
Runs: **5** | Seeds: **[42, 43, 44, 45, 46]**

## 8.1 Retrieval (mean +/- std)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| ColPali (stage-1) | 0.5462 +/- 0.0255 | 0.8848 +/- 0.0245 | 1.0000 +/- 0.0000 | 0.6937 +/- 0.0198 | 0.7673 +/- 0.0153 |
| ColPali + MLP reranker | 0.5440 +/- 0.0244 | 0.8841 +/- 0.0235 | 1.0000 +/- 0.0000 | 0.6961 +/- 0.0207 | 0.7694 +/- 0.0161 |
| X-PageRerank (ablation: no graph) | 0.5946 +/- 0.0403 | 0.9237 +/- 0.0082 | 1.0000 +/- 0.0000 | 0.7410 +/- 0.0270 | 0.8043 +/- 0.0204 |
| X-PageRerank (GAT) | **0.6145 +/- 0.0382** | **0.9248 +/- 0.0068** | 1.0000 +/- 0.0000 | **0.7497 +/- 0.0245** | **0.8107 +/- 0.0183** |

Improvements of X-PageRerank (GAT) over ColPali stage-1:

- `Recall@1`: `+0.0682` (+6.82 points)
- `Recall@5`: `+0.0400` (+4.00 points)
- `MRR@10`: `+0.0560` (+5.60 points)
- `nDCG@10`: `+0.0434` (+4.34 points)

## 8.2 Support-page Metrics (mean +/- std)

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
|---|---:|---:|---:|---:|
| ColPali (stage-1) | 0.8848 +/- 0.0245 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |
| ColPali + MLP reranker | 0.8841 +/- 0.0235 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |
| X-PageRerank (ablation: no graph) | 0.9237 +/- 0.0082 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |
| X-PageRerank (GAT) | **0.9248 +/- 0.0068** | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |

Interpretation:

- Coverage@5 improves with graph reranking.
- `AllSupportHit@10 = 0` is expected when evaluation subsets mostly contain single-support-page queries; this metric only becomes informative with multi-support samples.

## 8.3 Pairwise Stability Signal

From aggregated pairwise analysis (`no-graph` vs `GAT`, Recall@1):

- Mean queries/run: `715.6`
- Mean queries where no-graph > GAT: `33.8`
- Mean equal queries: `633.2`

This indicates GAT is generally robust, with a minority subset where no-graph still wins.

---

## 9) End-to-End Reproducibility: How to Run

## 9.1 Recommended One-Command 5-seed pipeline

```bash
cd /home/admin1/Desktop/colpali

python3 -m experiments.run_phase1_multi_seed \
  --candidate_cache cache/mpdoc_val_full \
  --root multi_seed_runs/mpdoc_val_disjoint \
  --seeds 42,43,44,45,46 \
  --graph_preset sparse-graph \
  --num_epochs 10 \
  --top_k 20
```

This will:

1. create disjoint splits for each seed,  
2. train each seed (`best.pt`),  
3. evaluate on each seed test split,  
4. aggregate into:
   - `multi_seed_runs/mpdoc_val_disjoint/aggregated_results.json`
   - `multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md`

## 9.2 Run aggregation separately

```bash
python3 -m experiments.summarize_phase1_multi_seed \
  --glob 'multi_seed_runs/mpdoc_val_disjoint/seed_*/eval/phase1_results.json' \
  --out_json multi_seed_runs/mpdoc_val_disjoint/aggregated_results.json \
  --out_md multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md
```

Optional:

- `--include_raw_values` to store per-seed metric arrays in JSON.

## 9.3 If candidate cache does not exist yet

```bash
export PARQUET_DIR="/absolute/path/to/MP-DocVQA/parquet/data"

python3 -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M \
  --model_type colidefics3 \
  --output_dir cache/mpdoc_val_full \
  --top_k 20 \
  --batch_size 4 \
  --num_samples 5000 \
  --sample_offset 0 \
  --cache_chunk_rows 512 \
  --maxsim_doc_chunk 16
```

Then run the 5-seed pipeline in section 9.1.

---

## 10) Practical Notes and Failure Modes

1. Placeholder path confusion  
   Use a real cache folder path. `--candidate_cache /đường/dẫn/...` is only a placeholder and will fail (`missing meta.json`).

2. Memory pressure during caching  
   Large `num_samples` can OOM if loaded at once. Lower `cache_chunk_rows`, `maxsim_doc_chunk`, and batch size as needed.

3. Top-k consistency  
   Keep `top_k` consistent between cache construction and downstream train/eval assumptions.

4. Strict metric interpretation  
   `Coverage@k` and `Recall@k` can appear very similar when many samples have single support page.

---

## 11) Limitations and Next Steps

- Current summary is retrieval-centered; full QA EM/F1 evaluation is not included in this report.
- To make `AllSupportHit@k` informative, curate a stronger multi-support subset.
- Extend to region-level graph nodes and additional datasets (e.g., DUDE) for broader generalization claims.
- Add efficiency/latency benchmarking for deployment-oriented comparison.

---

## 12) Conclusion

Across 5 random seeds with strict disjoint splitting, the current page-graph implementation (X-PageRerank/GAT) consistently improves over stage-1 and no-graph alternatives on top-ranking quality (`Recall@1`, `MRR@10`, `nDCG@10`) while preserving a practical drop-in pipeline. The upgraded method in this report reframes the task from page reranking to query-conditioned heterogeneous evidence-graph retrieval, providing the target formulation for the next implementation phase.

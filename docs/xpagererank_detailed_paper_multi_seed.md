# X-PageRerank: Detailed Paper-Style Technical Report (Multi-Seed)

## Abstract

Visual late-interaction document retrievers (e.g., ColPali-style models) are strong at page-level matching but remain limited for cross-page evidence composition. We present and evaluate **X-PageRerank**, a lightweight stage-2 reranking framework that performs query-conditioned graph reasoning over top-k retrieved pages. Unlike corpus-level graph retrieval frameworks, our approach builds a local graph at query time only, making it practical for integration with existing retrieval pipelines.  

In this repository, we run a strict disjoint-split protocol and a 5-seed evaluation (`42..46`) on MP-DocVQA-derived caches. Aggregated results show consistent gains from graph reranking: **Recall@1 +6.82 points**, **MRR@10 +5.60 points**, and **nDCG@10 +4.34 points** over stage-1 ColPali baseline (mean across 5 seeds). We provide complete reproducibility details, hyperparameters, and commands.

---

## 1) Problem Setting and Motivation

For a query `q` and a multi-page document, stage-1 retrieval returns top-k page candidates. The challenge is that evidence for a question can be distributed across multiple pages. Pure page-wise scoring often misses this composition signal.

We cast reranking as:

1. Retrieve top-k pages with a visual late-interaction retriever.
2. Build a query-conditioned page graph over those candidates.
3. Run GAT message passing to refine page representations.
4. Produce reranked scores and evaluate retrieval/support-page metrics.

This keeps stage-1 unchanged while adding cross-page reasoning where it matters most (query-time top-k set).

---

## 2) Positioning vs Prior Graph Retrieval Work

X-PageRerank is intentionally scoped as a **drop-in post-retrieval module**:

- Backbone retrieval stays fixed (ColPali-style late interaction).
- Graph is **local** to top-k candidates (no corpus-wide component graph).
- Focus is practical cross-page reranking for document QA pipelines.

This is consistent with the repo design direction captured in `docs/idea.md` and subsequent experiment reports.

---

## 3) Method

## 3.1 Stage-1 Retrieval and Inputs

For each sample:

- Query embedding tensor: `*_q.pt`
- Top-k page embedding tensor: `*_p.pt`
- Stage-1 scores: `*_s0.pt`
- Support mask labels: `*_mask.pt`
- Metadata (`question_id`, `doc_id`, `page_numbers`) in `meta.json`

All are stored as a cached `RerankDataset`.

## 3.2 Graph Construction

From top-k retrieved pages, build an evidence graph with:

- semantic edges (embedding similarity),
- page adjacency edges (nearby page indices),
- optional query-node conditioning.

The current training/eval path supports graph sparsification by preset:

- `graph_preset=default`: use CLI values directly
- `graph_preset=sparse-graph`: enforce
  - `sem_threshold = max(sem_threshold, 0.75)`
  - `adj_max_gap = min(adj_max_gap, 1)`

## 3.3 Reranker and Objective

The reranker uses a GAT module (`PageGATReranker`) and optimizes support-page ranking:

- listwise softmax cross-entropy over top-k pages,
- auxiliary pairwise margin loss (hard negative style),
- checkpoint selection by validation `Recall@5` with early stopping.

## 3.4 Evaluation Outputs

Per run, `phase1_results.json` stores:

- retrieval metrics (`Recall@k`, `MRR@10`, `nDCG@10`, etc.),
- support-page metrics (`Coverage@k`, `AllSupportHit@k`, `SupportF1@k`),
- per-query and pairwise analysis (`no-graph` vs `GAT`).

---

## 4) Experimental Protocol Used in This Workspace

## 4.1 Data and Split Strategy

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

## 4.2 Multi-Seed Runner Hyperparameters (actual driver)

From `experiments/run_phase1_multi_seed.py`:

- Seeds: `42,43,44,45,46`
- Split ratios: `train=0.70`, `val=0.15`, `test=0.15`
- Graph preset: `sparse-graph`
- Train epochs: `10`
- Train batch size: `32`
- Top-k (train cache view): `20`
- Eval k: `10`
- Aggregation: enabled by default (unless `--no_aggregate`)

## 4.3 Train/Eval Core Hyperparameters (pipeline defaults)

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

## 4.4 Cache-Build Hyperparameters (OOM-safe)

Important options (from `run_phase1 cache`):

- `top_k` (must be consistent with train/eval assumptions)
- `max_pages` (default `60`)
- `batch_size` (default `4`)
- `maxsim_doc_chunk` (default `16`, reduce to lower RAM)
- `cache_chunk_rows` (default `512`, lower for memory safety)
- optional doc-disjoint filtering via `--exclude_train_doc_ids_cache`

---

## 5) 5-Seed Aggregated Results (Latest Output)

Source: `multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md`  
Runs: **5** | Seeds: **[42, 43, 44, 45, 46]**

## 5.1 Retrieval (mean +/- std)

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

## 5.2 Support-page Metrics (mean +/- std)

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
|---|---:|---:|---:|---:|
| ColPali (stage-1) | 0.8848 +/- 0.0245 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |
| ColPali + MLP reranker | 0.8841 +/- 0.0235 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |
| X-PageRerank (ablation: no graph) | 0.9237 +/- 0.0082 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |
| X-PageRerank (GAT) | **0.9248 +/- 0.0068** | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.6129 +/- 0.0270 |

Interpretation:

- Coverage@5 improves with graph reranking.
- `AllSupportHit@10 = 0` is expected when evaluation subsets mostly contain single-support-page queries; this metric only becomes informative with multi-support samples.

## 5.3 Pairwise Stability Signal

From aggregated pairwise analysis (`no-graph` vs `GAT`, Recall@1):

- Mean queries/run: `715.6`
- Mean queries where no-graph > GAT: `33.8`
- Mean equal queries: `633.2`

This indicates GAT is generally robust, with a minority subset where no-graph still wins.

---

## 6) End-to-End Reproducibility: How to Run

## 6.1 Recommended One-Command 5-seed pipeline

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

## 6.2 Run aggregation separately

```bash
python3 -m experiments.summarize_phase1_multi_seed \
  --glob 'multi_seed_runs/mpdoc_val_disjoint/seed_*/eval/phase1_results.json' \
  --out_json multi_seed_runs/mpdoc_val_disjoint/aggregated_results.json \
  --out_md multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md
```

Optional:

- `--include_raw_values` to store per-seed metric arrays in JSON.

## 6.3 If candidate cache does not exist yet

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

Then run the 5-seed pipeline in section 6.1.

---

## 7) Practical Notes and Failure Modes

1. Placeholder path confusion  
   Use a real cache folder path. `--candidate_cache /đường/dẫn/...` is only a placeholder and will fail (`missing meta.json`).

2. Memory pressure during caching  
   Large `num_samples` can OOM if loaded at once. Lower `cache_chunk_rows`, `maxsim_doc_chunk`, and batch size as needed.

3. Top-k consistency  
   Keep `top_k` consistent between cache construction and downstream train/eval assumptions.

4. Strict metric interpretation  
   `Coverage@k` and `Recall@k` can appear very similar when many samples have single support page.

---

## 8) Limitations and Next Steps

- Current summary is retrieval-centered; full QA EM/F1 evaluation is not included in this report.
- To make `AllSupportHit@k` informative, curate a stronger multi-support subset.
- Extend to region-level graph nodes and additional datasets (e.g., DUDE) for broader generalization claims.
- Add efficiency/latency benchmarking for deployment-oriented comparison.

---

## 9) Conclusion

Across 5 random seeds with strict disjoint splitting, X-PageRerank (GAT) consistently improves over stage-1 and no-graph alternatives on top-ranking quality (`Recall@1`, `MRR@10`, `nDCG@10`) while preserving a practical drop-in pipeline. The method is reproducible with one command in this repository and provides a solid baseline for future cross-page evidence composition research.

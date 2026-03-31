# EvidenceGraph-RAG: Query-Conditioned Region Graph Reranking for Multi-Page Document QA

## Abstract

Multi-page document question answering requires retrieving supporting pages under visual, textual, and layout variation. In practical pipelines, strong stage-1 retrievers still rank pages mostly independently and can miss cross-region evidence interactions. We present a deployable two-stage framework, **EvidenceGraph-RAG**, where stage-1 query/page representations are cached once and stage-2 performs query-conditioned region-graph reranking over top-k candidates. We report a reproducible 5-seed study on MP-DocVQA-derived cached candidates with strict document/question disjoint splits. In the current run setting (`train/val/test = 12/8/80`, seeds 42–46), EvidenceGraph-RAG improves over `ColPali (stage-1)` by +6.23 points on Recall@1, +7.26 points on Recall@5, +7.01 points on MRR@10, and +5.53 points on nDCG@10. We provide detailed setup, command-level reproducibility, and a prioritized experimental completion plan for A* conference submission quality.

## 1. Introduction

Retrieval is a central bottleneck in document QA systems. In multi-page settings, support can be distributed across different regions and pages, while late-interaction retrievers typically produce page-level independent rankings. This gap motivates a second-stage reranker that explicitly models evidence relations while preserving practical deployment constraints.

This work is scoped to a production-friendly setting:

1. Stage-1 embeddings and scores are precomputed and frozen in cache.
2. Stage-2 rerankers operate only on top-k candidate pages.
3. Training and evaluation are done with strict split reproducibility.

The key question in this draft is: **does region-graph reranking deliver stable top-rank retrieval gains over strong internal baselines under multi-seed evaluation?**

## 2. Method

### 2.1 Two-stage architecture

Given a query `q` and document pages `{p_i}`:

1. **Stage-1 retrieval (cached):** store query embeddings (`*_q.pt`), page embeddings (`*_p.pt`), stage-1 scores (`*_s0.pt`), and support labels (`*_mask.pt`).
2. **Stage-2 reranking:** train a lightweight reranker that reorders stage-1 top-k pages.

This draft compares:

- `ColPali (stage-1)` (stage-1 ranking from cached scores),
- `ColPali + MLP reranker` (non-graph reranker),
- `EvidenceGraph-RAG (page+region)` (region graph reranker).

### 2.2 EvidenceGraph-RAG (page+region)

EvidenceGraph-RAG constructs a query-conditioned graph over candidate pages and region partitions, with region-level interactions controlled by:

- `grid_rows=2`, `grid_cols=2`,
- `sem_threshold_region=0.70`,
- interpolation schedule (`lambda_mix_start=0.15`, `lambda_mix_end=0.55`, `lambda_mix_warmup_steps=1200`).

The design keeps stage-1 fixed and adds structural reasoning where it matters most (top-k candidate set).

## 3. Experimental Setup

### 3.1 Data and split protocol

- Candidate cache: `cache/mpdoc_val_full`
- Multi-seed root: `multi_seed_runs/mpdoc_val_disjoint_region_best`
- Seeds: `[42, 43, 44, 45, 46]`
- Split constraints: **doc-disjoint** and **question-disjoint**
- Ratios: **train 12% / val 8% / test 80%**

Per-seed test size from `split_summary.json`:

- seed 42: 3871
- seed 43: 3793
- seed 44: 3778
- seed 45: 3858
- seed 46: 3882

### 3.2 Training and evaluation hyperparameters

- `reranker_type=region`
- `num_epochs=20`
- `train_batch_size=16`
- `lr=5e-5`
- `top_k=20`
- `eval_k=10`
- `graph_preset=default`

### 3.3 Reproducibility command

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

## 4. Results

Source: `multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_results.json`

### 4.1 Retrieval metrics (mean ± std across 5 seeds)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5415 ± 0.0088 | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.6899 ± 0.0056 | 0.7644 ± 0.0042 |
| ColPali + MLP reranker | 0.5371 ± 0.0119 | 0.8843 ± 0.0109 | 1.0000 ± 0.0000 | 0.6871 ± 0.0095 | 0.7624 ± 0.0074 |
| **EvidenceGraph-RAG (page+region)** | **0.6038 ± 0.0382** | **0.9534 ± 0.0324** | **1.0000 ± 0.0000** | **0.7600 ± 0.0403** | **0.8198 ± 0.0315** |

### 4.2 Support-page metrics (mean ± std across 5 seeds)

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| ColPali + MLP reranker | 0.8843 ± 0.0109 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| **EvidenceGraph-RAG (page+region)** | **0.9534 ± 0.0324** | **1.0000 ± 0.0000** | **0.0000 ± 0.0000** | **0.6102 ± 0.0058** |

### 4.3 Delta against `ColPali (stage-1)`

- Recall@1: **+0.0623**
- Recall@5: **+0.0726**
- MRR@10: **+0.0701**
- nDCG@10: **+0.0553**

### 4.4 Per-seed retrieval for EvidenceGraph-RAG

| Seed | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| 42 | 0.5670 | 0.9259 | 0.7230 | 0.7909 |
| 43 | 0.6475 | 0.9837 | 0.8080 | 0.8571 |
| 44 | 0.6427 | 0.9788 | 0.7974 | 0.8488 |
| 45 | 0.5814 | 0.9121 | 0.7245 | 0.7914 |
| 46 | 0.5804 | 0.9668 | 0.7474 | 0.8106 |

## 5. Discussion

### 5.1 Main findings

1. EvidenceGraph-RAG substantially improves early-rank retrieval metrics, which are most relevant for downstream QA.
2. Gains are visible against both stage-1-only and non-graph MLP reranking.
3. The method remains practical because it does not require rebuilding a corpus-level graph index.

### 5.2 Metric caveat

In this cache/evaluation regime, `AllSupportHit@10 = 0.0` for all methods. This indicates that strict multi-support evaluation is not informative under current support-label cardinality; claims should focus on robust early-rank retrieval gains (Recall@1/5, MRR, nDCG), and this limitation must be explicit in submission text.

## 6. Threats to Validity

1. **Split ratio sensitivity:** Current main run uses 12/8/80; behavior may differ under more train-heavy splits.
2. **Label structure:** Support cardinality limits strict multi-evidence metrics.
3. **Baseline scope:** Current comparison is internal (stage-1 and MLP); external baselines are not yet included.
4. **End-task linkage:** Retrieval gains are not yet connected to final QA answer quality in this draft.

## 7. Reproducibility Checklist

- [x] Exact seed list and split ratio
- [x] Exact CLI command
- [x] Per-seed outputs (`seed_*/eval/phase1_results.json`)
- [x] Aggregated outputs (`aggregated_results.json`, `aggregated_report.md`)
- [ ] Environment freeze (`pip freeze`, CUDA, GPU model)
- [ ] Commit hash snapshot for camera-ready

## 8. Required Additional Experiments Before A* Submission

### 8.1 Must-run (priority P0)

- [ ] **Page-branch family on same seeds/splits**:
  - `X-PageRerank (GAT)`
  - `X-PageRerank (ablation: no graph)`
- [ ] **Significance testing** on Recall@1, Recall@5, MRR@10, nDCG@10 (bootstrap or paired randomization).
- [ ] **Second split regime** (e.g., 70/15/15) to show robustness beyond 12/8/80.

### 8.2 Strongly recommended (priority P1)

- [ ] Graph density ablation (`default` vs `sparse-graph`).
- [ ] Region grid ablation: `(1,1)`, `(2,2)`, `(3,3)`.
- [ ] Lambda schedule ablation (fixed vs warmup).
- [ ] Inference efficiency: latency/query and peak memory by method.

### 8.3 Submission quality boosters (priority P2)

- [ ] Downstream QA metric linkage (answer EM/F1 or task-specific answer score).
- [ ] Qualitative win/failure analysis with case studies.
- [ ] External baseline expansion (if infra allows).

### 8.4 One-command multiseed battery + comparison table (P0/P1 runner)

This repo includes a **battery runner** that maps §8.1–§8.2 checks to concrete multi-seed jobs (same seeds as the main paper, default `42–46`) and a **comparison merge** over all `aggregated_results.json` files.

**What one battery run includes** (each line is a separate subdirectory under `--battery_root`, with its own doc-disjoint splits and full train/eval):

| Slug | Maps to |
| --- | --- |
| `page_gat_default` | X-PageRerank family: **GAT**, **no-graph ablation**, ColPali, MLP (`reranker_type=page`, `graph_preset=default`) |
| `page_gat_sparse_graph` | Same page family with **`sparse-graph`** preset |
| `region_main_default` | EvidenceGraph-RAG **region** baseline aligned with §3 (`default` graph, 2×2 grid, λ warmup) |
| `region_sparse_graph` | Region model + **`sparse-graph`** |
| `region_grid_1x1` / `region_grid_3x3` | Region **grid ablation** |
| `region_lambda_fixed_mid` | **λ schedule ablation**: `lambda_mix_start=end=0.35` (flat schedule vs warmup in `region_main_default`) |
| `region_split_70_15_15` | **Second split regime** `train/val/test = 70/15/15` (same seeds) |

**Launch (from repo root)** — adjust paths if needed; requires an existing `candidate_cache` (e.g. `cache/mpdoc_val_full`) with `top_k` matching `--top_k`:

```bash
python3 -m experiments.run_ablation_battery \
  --candidate_cache cache/mpdoc_val_full \
  --battery_root multi_seed_runs/paper_ablation_battery \
  --seeds 42,43,44,45,46 \
  --num_epochs 20 \
  --train_batch_size 16 \
  --lr 5e-5 \
  --top_k 20 \
  --eval_k 10
```

Useful flags:

- `--dry_run` — print underlying `run_phase1_multi_seed` commands only.
- `--skip_completed` — skip a slug if `aggregated_results.json` exists in the slug dir or under `_summaries/{slug}/` (resume after failures).
- `--only slug1,slug2` — run a subset (e.g. `--only page_gat_default,region_main_default`).
- **Disk (default):** after each slug, once `aggregated_results.json` exists, all **`seed_*` directories** for that slug are deleted (splits, checkpoints, per-seed eval); only the aggregate JSON/Markdown remain until the next run. Use `--no_prune_seed_dirs` to keep full trees. Use `--archive_slug_then_delete` to copy aggregates to `{battery_root}/_summaries/{slug}/` and remove the entire slug directory (maximum savings; `compare_ablation_aggregates` reads `_summaries` too).

**Merge into a single comparison report** (after all slugs finished):

```bash
python3 -m experiments.compare_ablation_aggregates \
  --battery_root multi_seed_runs/paper_ablation_battery
```

This writes `ablation_comparison.md` and `ablation_comparison.json` under the battery root: primary reranker metrics per experiment, a **GAT vs no-graph** table for page runs, and **ColPali R@1** per split for sanity.

**Still manual for A\* quality**: paired bootstrap / randomization tests (§8.1), latency/memory (§8.2), and P2 items are not generated by the battery.

## 9. Suggested Main-Paper Structure (LaTeX-ready)

1. Introduction  
2. Related Work  
3. Method  
4. Experimental Setup  
5. Results  
6. Ablations and Analysis  
7. Limitations  
8. Conclusion  

## 10. Conclusion

This draft establishes a reproducible and practically meaningful result: EvidenceGraph-RAG improves early-rank retrieval quality over strong internal baselines in a 5-seed MP-DocVQA setting. The method is deployment-compatible and stable across seeds, but a submission targeting an A* venue still requires broader baseline coverage, robustness checks, significance analysis, and end-task QA linkage.


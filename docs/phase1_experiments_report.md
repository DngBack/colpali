# Phase 1 Experiments Report (Cache → Train → Eval)

## Goal
Evaluate a Phase-1 post-retrieval reranker (`PageGATReranker`, GAT-based) for visual late-interaction retrieval on MP-DocVQA, with the key metric being how well the reranker retrieves *support pages* (gold evidence pages) in the top-k.

## Pipeline
1. **Cache (top-K embeddings)**
   - Encode query + pages using a ColPali/ColQwen-style encoder.
   - Build a `RerankDataset` containing:
     - `*_q.pt`: query embeddings
     - `*_p.pt`: page embeddings
     - `*_s0.pt`: stage-1 scores
     - `*_mask.pt`: support mask (gold evidence pages)
   - Write `meta.json` to store per-sample metadata (`question_id`, `doc_id`, `page_numbers`, …).

2. **Train (GAT reranker)**
   - Load `RerankDataset` from cached files via `RerankDataset.load()`, which expects `meta.json`.
   - Build evidence graphs on the fly from retrieved top-k pages.
   - Train with the implemented losses and early stopping.

3. **Eval**
   - Produce ranked page predictions per query.
   - Compare predictions against gold support pages using IR-style metrics:
     - `Recall@k` (any-hit: at least one gold support page appears in top-k)
     - `MRR@10` (reciprocal rank of the first relevant support page in top-10)
     - `nDCG@10` (binary relevance with discounted gain)
   - Also compute cross-page evidence metrics (coverage, AllSupportHit, etc.).

## Caching Setup
### Train cache split
During early runs, offline mode could not find the `train` labeled split locally and fell back to `test`, which led to empty caches (`meta.json` as `[]`) and `RerankDataset: 0 samples`.

The working approach was to build both train/val caches from the labeled parquet split (`--split val`) and use non-overlapping row windows via:
- `--num_samples`
- `--sample_offset`

Used commands (working configuration):
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

### Val cache split (row offset only — can still share `doc_id` with train)

```bash
rm -rf cache/real_val_split

python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/real_val_split \
  --top_k 10 --batch_size 4 \
  --num_samples 100 --sample_offset 200
```

### Val cache split (**document-disjoint** from train — recommended for GAT validation)

Rows whose `doc_id` appears in `cache/real_train_split/meta.json` are **removed before encoding** (saves GPU). Use a **wider** `--num_samples` window so enough questions remain after filtering.

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

## Training
Command:
```bash
python -m experiments.run_phase1 train \
  --train_cache cache/real_train_split \
  --val_cache   cache/real_val_doc_disjoint \
  --output_dir  checkpoints/gat_real_split \
  --num_epochs 10 --batch_size 16 --lr 5e-4
```

Observed dataset sizes (from logs — **row-split** val, not document-disjoint):
- `cache/real_train_split`: **190 samples**
- `cache/real_val_split`: **78 samples**

When using `cache/real_val_doc_disjoint`, check the log `Doc-disjoint filter: kept X/Y samples` for the final val size.

Training outcome:
- Early stopping at **epoch 8**
- Best checkpoint: `checkpoints/gat_real_split/best.pt`
- Best validation metric used by trainer: `Recall@5`
  - `Best Recall@5 = 0.9359`

## Evaluation Results

Row-split val (historical run):
```bash
python -m experiments.run_phase1 eval \
  --eval_cache cache/real_val_split \
  --checkpoint checkpoints/gat_real_split/best.pt \
  --output_dir results/phase1_real_split
```

Document-disjoint val:
```bash
python -m experiments.run_phase1 eval \
  --eval_cache cache/real_val_doc_disjoint \
  --checkpoint checkpoints/gat_real_split/best.pt \
  --output_dir results/phase1_doc_disjoint_val
```

### Main Retrieval Metrics (MP-DocVQA)
```text
ColPali (stage-1)
  Recall@1=0.3590, Recall@5=0.8718, Recall@10=1.0000, MRR@10=0.5840, nDCG@10=0.6852

ColPali + MLP reranker
  Recall@1=0.4615, Recall@5=0.9231, Recall@10=1.0000, MRR@10=0.6454, nDCG@10=0.7323

X-PageRerank (ablation: no graph)
  Recall@1=0.5769, Recall@5=0.9359, Recall@10=1.0000, MRR@10=0.7322, nDCG@10=0.7980

X-PageRerank (GAT)
  Recall@1=0.5897, Recall@5=0.9359, Recall@10=1.0000, MRR@10=0.7377, nDCG@10=0.8017
```

Key takeaways:
- Moving from stage-1 ColPali to reranking increases deeper-ranking quality:
  - `MRR@10`: `0.5840 → 0.7377` (GAT)
  - `nDCG@10`: `0.6852 → 0.8017` (GAT)
- The strict cutoff metric `Recall@5` is the same for `ablation:no-graph` and `GAT` (**0.9359**). The graph improves ranking depth more than “any-hit within top-5”.

## Overlap Check (Train vs Val)
To avoid evaluation leakage, we checked overlap using `question_id` from both caches:
- `cache/real_train_split/meta.json` (190 samples)
- `cache/real_val_split/meta.json` (78 samples)

Results:
- `question_id overlap count = 0` (no question is shared between train and val)
- `doc_id overlap count = 3` (same documents may appear across splits, but with different questions)

This suggests no direct question leakage, but **documents can still overlap** across row splits. Use `--exclude_train_doc_ids_cache` (val) or `filter_rerank_cache.py` (test) for document-level splits.

## Building a Strict Test Set (No Document Overlap)
### Approach
1. Build a larger **candidate test pool** from the labeled parquet split (again typically `--split val` in offline mode).
2. Filter the candidate pool using `doc_id` so that the final test cache contains documents not present in:
   - `cache/real_train_split`
   - `cache/real_val_doc_disjoint` (or your val cache directory)
3. Run `eval` with the filtered test cache.

### New helper script
To support “no re-encoding” filtering, we added:
- `experiments/train/filter_rerank_cache.py`

It:
- reads `meta.json` from the candidate cache
- excludes any sample whose `doc_id` appears in the excluded caches
- copies only the needed `*_q.pt`, `*_p.pt`, `*_s0.pt`, `*_mask.pt`
- writes a new `meta.json` for the filtered cache

### Command sketch
```bash
rm -rf cache/real_test_pool cache/real_test_split_no_doc_overlap

# 1) Build candidate pool (choose offsets so you have enough to filter)
python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/real_test_pool \
  --top_k 10 --batch_size 4 \
  --num_samples 300 --sample_offset 300

# 2) Filter out any doc_id that appears in train/val
python -m experiments.train.filter_rerank_cache \
  --candidate_cache cache/real_test_pool \
  --exclude_cache cache/real_train_split \
  --exclude_cache cache/real_val_doc_disjoint \
  --output_dir cache/real_test_split_no_doc_overlap

# 3) Evaluate on the strict test cache
python -m experiments.run_phase1 eval \
  --eval_cache cache/real_test_split_no_doc_overlap \
  --checkpoint checkpoints/gat_real_split/best.pt \
  --output_dir results/test_real_no_doc_overlap
```

## Future Directions
1. **Report both any-hit and strict all-support metrics**
   - `Recall_all@k` / `AllSupportHit@k` is stricter and more aligned with evidence composition.
2. **Test sensitivity**
   - Repeat with different offsets to ensure results are not artifacts of small splits.
3. **Ablate graph variants**
   - Since `Recall@5` matches between `GAT` and `no-graph`, focus analysis on how the graph changes the *rank position* of gold supports (MRR/nDCG per-query).


# Experiment Note — 2026-03-30

## Context

This note records today's detailed outputs for the new idea (EvidenceGraph-RAG) and compares against the old baseline (X-PageRerank GAT).

Primary result files:

- `multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_results_with_percentiles.json`
- `multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_report_with_percentiles.md`

Baseline comparison files:

- `multi_seed_runs/mpdoc_val_disjoint/aggregated_results_with_percentiles.json`
- `multi_seed_runs/mpdoc_val_disjoint/aggregated_report_with_percentiles.md`

Seeds in both runs: `[42, 43, 44, 45, 46]`.

## Best config selected from seed-42 sweep

Chosen from `results/region_s42_*` logs:

- `reranker_type=region`
- `grid_rows=2`, `grid_cols=2`
- `sem_threshold_region=0.70`
- `cross_page_region_edges=off`
- `lr=5e-5`
- `num_epochs=20` (early stopping observed before max)
- `lambda_mix_start=0.15`
- `lambda_mix_end=0.55`
- `lambda_mix_warmup_steps=1200`

Reason: best seed-42 `MRR@10`/`Recall@1` among completed sweep runs.

## Multi-seed retrieval summary (new vs old)

### New method: EvidenceGraph-RAG (page+region)

- `Recall@1`: mean `0.6925`, std `0.0427`, p25/p50/p75 = `0.6599/0.6820/0.7030`
- `Recall@5`: mean `0.9852`, std `0.0081`, p25/p50/p75 = `0.9793/0.9813/0.9897`
- `Recall@10`: mean `1.0000`, std `0.0000`, p25/p50/p75 = `1.0000/1.0000/1.0000`
- `MRR@10`: mean `0.8312`, std `0.0227`, p25/p50/p75 = `0.8169/0.8208/0.8442`
- `nDCG@10`: mean `0.8744`, std `0.0168`, p25/p50/p75 = `0.8641/0.8662/0.8845`

### Old method: X-PageRerank (GAT)

- `Recall@1`: mean `0.6145`, std `0.0382`, p25/p50/p75 = `0.5873/0.5924/0.6489`
- `Recall@5`: mean `0.9248`, std `0.0068`, p25/p50/p75 = `0.9197/0.9263/0.9275`
- `Recall@10`: mean `1.0000`, std `0.0000`, p25/p50/p75 = `1.0000/1.0000/1.0000`
- `MRR@10`: mean `0.7497`, std `0.0245`, p25/p50/p75 = `0.7328/0.7350/0.7696`
- `nDCG@10`: mean `0.8107`, std `0.0183`, p25/p50/p75 = `0.7983/0.7996/0.8255`

### Delta (new - old)

- `Recall@1`: `+0.0780`
- `Recall@5`: `+0.0604`
- `Recall@10`: `+0.0000`
- `MRR@10`: `+0.0815`
- `nDCG@10`: `+0.0637`

## Support-page summary (new vs old)

### New method (EvidenceGraph-RAG)

- `Coverage@5`: mean `0.9852`, std `0.0081`, p25/p50/p75 = `0.9793/0.9813/0.9897`
- `Coverage@10`: mean `1.0000`, std `0.0000`
- `AllSupportHit@10`: mean `0.0000`, std `0.0000`
- `SupportF1@10`: mean `0.6129`, std `0.0270`, p25/p50/p75 = `0.6085/0.6189/0.6266`

### Old method (X-PageRerank GAT)

- `Coverage@5`: mean `0.9248`, std `0.0068`, p25/p50/p75 = `0.9197/0.9263/0.9275`
- `Coverage@10`: mean `1.0000`, std `0.0000`
- `AllSupportHit@10`: mean `0.0000`, std `0.0000`
- `SupportF1@10`: mean `0.6129`, std `0.0270`, p25/p50/p75 = `0.6085/0.6189/0.6266`

### Delta (new - old)

- `Coverage@5`: `+0.0604`
- `Coverage@10`: `+0.0000`
- `AllSupportHit@10`: `+0.0000`
- `SupportF1@10`: `+0.0000`

## Notes from today's run logs

- Region sweep was executed on seed 42 for:
  - `grid=2x2`
  - `sem_threshold_region={0.65, 0.70, 0.75}`
  - `cross_page_region_edges={off,on}`
- Top seed-42 run observed:
  - `results/region_s42_g2_thr0.70_crossoff/phase1_results.json`
  - Retrieval: `Recall@1=0.7568`, `Recall@5=0.9813`, `MRR@10=0.8630`, `nDCG@10=0.8977`
- A later long sweep segment was interrupted manually (`KeyboardInterrupt`) while starting another config; completed runs before interruption remain valid.

## Practical conclusion for paper update

The new EvidenceGraph-RAG setting (with typed region/evidence structure and lambda schedule) now shows strong multi-seed gains over the old X-PageRerank baseline on retrieval quality (`Recall@1`, `MRR@10`, `nDCG@10`) and support coverage (`Coverage@5`), while preserving `Recall@10=1.0`.

# Multi-Seed Experiment Report: X-PageRerank vs EvidenceGraph-RAG

## Scope

This note compares:

- Baseline report: `multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md`
- New method report: `multi_seed_runs/mpdoc_val_disjoint/aggregated_report_region.md`

Both reports summarize 5-seed runs on the same split family (`42..46`).

## Main Comparison (Retrieval)

From the two aggregated reports:

- X-PageRerank (GAT): `Recall@1=0.6145`, `Recall@5=0.9248`, `MRR@10=0.7497`, `nDCG@10=0.8107`
- EvidenceGraph-RAG (page+region): `Recall@1=0.5953`, `Recall@5=0.9322`, `MRR@10=0.7416`, `nDCG@10=0.8049`

Delta (new - old):

- `Recall@1`: **-0.0192**
- `Recall@5`: **+0.0074**
- `MRR@10`: **-0.0081**
- `nDCG@10`: **-0.0058**

Interpretation:

- The new method improves **coverage-style retrieval at top-5**.
- The old page-GAT remains stronger on **top-rank precision** (`Recall@1`, `MRR@10`, `nDCG@10`).
- This is consistent with an early evidence-graph variant that expands evidence candidates, but has not yet optimized first-hit ordering.

## Support-Page Comparison

From aggregated reports:

- X-PageRerank (GAT): `Coverage@5=0.9248`
- EvidenceGraph-RAG (page+region): `Coverage@5=0.9322`

Delta (new - old):

- `Coverage@5`: **+0.0074**
- `Coverage@10`, `AllSupportHit@10`, `SupportF1@10`: unchanged in this setup

Interpretation:

- Region evidence nodes are helping retrieve at least one support page in top-5 more often.
- Current benchmark composition still limits gains in `AllSupportHit@10` and `SupportF1@10` (many samples are effectively single-support).

## Statistical Read

- New method has slightly higher seed variance on `Recall@1` (`0.0420` vs `0.0382` std), indicating stability is not yet better than page-GAT.
- Gains appear concentrated in broader top-k support capture, not in first-rank sharpness.

## Important Note About Seed Display

`aggregated_report_region.md` shows `Seeds: [-1, -1, -1, -1, -1]` because files were aggregated from `seed_*/eval_region/phase1_results.json` and the summarizer only auto-parses seed ids when parent directory is exactly `eval`.

This does **not** invalidate metrics. It is only a metadata parsing issue in the report header.

## Conclusion

At current settings, EvidenceGraph-RAG (page+region) is:

- **better** for top-5 support coverage,
- **not yet better** than X-PageRerank on top-rank quality (`Recall@1`, `MRR@10`, `nDCG@10`).

So the method is working and promising, but still needs optimization to beat page-GAT on primary ranking metrics.

## Recommended Next Experiments

Priority order:

1. Train longer for region model (`num_epochs=15~20`) with early stopping.
2. Tune interpolation strength (`lambda_mix`) and lower LR (`5e-5`) for better rank stability.
3. Add stronger region graph structure:
   - enable `--cross_page_region_edges` for selected runs,
   - sweep `--sem_threshold_region` (`0.65/0.70/0.75`).
4. Increase region granularity sweep:
   - `2x2` vs `3x3` grid.
5. Run a strict ablation table:
   - page-only graph,
   - page+region without cross-page region edges,
   - full region graph with cross-page region edges.

If the goal is to claim superiority over old X-PageRerank on the same benchmark, the immediate target is to close:

- ~`1.9` points on `Recall@1`,
- ~`0.8` points on `MRR@10`.

## Reproducibility Artifacts

- Old aggregate: `multi_seed_runs/mpdoc_val_disjoint/aggregated_report.md`
- New aggregate: `multi_seed_runs/mpdoc_val_disjoint/aggregated_report_region.md`
- This analysis doc: `docs/region_multiseed_experiments_report.md`

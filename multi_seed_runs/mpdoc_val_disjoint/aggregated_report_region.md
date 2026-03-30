# Multi-seed Phase 1 summary

Runs: **5**  |  Seeds: `[-1, -1, -1, -1, -1]`

### retrieval

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5462 ± 0.0255 | 0.8848 ± 0.0245 | 1.0000 ± 0.0000 | 0.6937 ± 0.0198 | 0.7673 ± 0.0153 |
| ColPali + MLP reranker | 0.5383 ± 0.0314 | 0.8976 ± 0.0310 | 1.0000 ± 0.0000 | 0.6959 ± 0.0278 | 0.7696 ± 0.0217 |
| EvidenceGraph-RAG (page+region) | 0.5953 ± 0.0420 | 0.9322 ± 0.0092 | 1.0000 ± 0.0000 | 0.7416 ± 0.0267 | 0.8049 ± 0.0201 |

### support_pages

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8848 ± 0.0245 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |
| ColPali + MLP reranker | 0.8976 ± 0.0310 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |
| EvidenceGraph-RAG (page+region) | 0.9322 ± 0.0092 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |

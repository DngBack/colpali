# Multi-seed Phase 1 summary

Runs: **5**  |  Seeds: `[42, 43, 44, 45, 46]`

### retrieval

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5415 ± 0.0088 | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.6899 ± 0.0056 | 0.7644 ± 0.0042 |
| ColPali + MLP reranker | 0.5400 ± 0.0117 | 0.8928 ± 0.0162 | 1.0000 ± 0.0000 | 0.6926 ± 0.0124 | 0.7668 ± 0.0097 |
| EvidenceGraph-RAG (page+region) | 0.5977 ± 0.0359 | 0.9507 ± 0.0324 | 1.0000 ± 0.0000 | 0.7540 ± 0.0390 | 0.8151 ± 0.0305 |

### support_pages

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| ColPali + MLP reranker | 0.8928 ± 0.0162 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| EvidenceGraph-RAG (page+region) | 0.9507 ± 0.0324 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |

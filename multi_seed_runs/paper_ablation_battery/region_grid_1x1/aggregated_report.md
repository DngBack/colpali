# Multi-seed Phase 1 summary

Runs: **5**  |  Seeds: `[42, 43, 44, 45, 46]`

### retrieval

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5415 ± 0.0088 | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.6899 ± 0.0056 | 0.7644 ± 0.0042 |
| ColPali + MLP reranker | 0.5334 ± 0.0100 | 0.8852 ± 0.0056 | 1.0000 ± 0.0000 | 0.6865 ± 0.0067 | 0.7621 ± 0.0051 |
| EvidenceGraph-RAG (page+region) | 0.6356 ± 0.0180 | 0.9776 ± 0.0076 | 1.0000 ± 0.0000 | 0.7934 ± 0.0179 | 0.8458 ± 0.0138 |

### support_pages

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| ColPali + MLP reranker | 0.8852 ± 0.0056 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| EvidenceGraph-RAG (page+region) | 0.9776 ± 0.0076 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |

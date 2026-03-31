# Multi-seed Phase 1 summary

Runs: **5**  |  Seeds: `[42, 43, 44, 45, 46]`

### retrieval

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5415 ± 0.0088 | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.6899 ± 0.0056 | 0.7644 ± 0.0042 |
| ColPali + MLP reranker | 0.5468 ± 0.0082 | 0.8894 ± 0.0078 | 1.0000 ± 0.0000 | 0.6964 ± 0.0067 | 0.7696 ± 0.0052 |
| EvidenceGraph-RAG (page+region) | 0.6153 ± 0.0322 | 0.9584 ± 0.0325 | 1.0000 ± 0.0000 | 0.7715 ± 0.0384 | 0.8286 ± 0.0302 |

### support_pages

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| ColPali + MLP reranker | 0.8894 ± 0.0078 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| EvidenceGraph-RAG (page+region) | 0.9584 ± 0.0325 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |

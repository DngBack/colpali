# Multi-seed Phase 1 summary

Runs: **5**  |  Seeds: `[42, 43, 44, 45, 46]`

### retrieval

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5415 ± 0.0088 | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.6899 ± 0.0056 | 0.7644 ± 0.0042 |
| ColPali + MLP reranker | 0.5443 ± 0.0054 | 0.8875 ± 0.0041 | 1.0000 ± 0.0000 | 0.6938 ± 0.0027 | 0.7676 ± 0.0020 |
| EvidenceGraph-RAG (page+region) | 0.6038 ± 0.0461 | 0.9480 ± 0.0441 | 1.0000 ± 0.0000 | 0.7583 ± 0.0483 | 0.8182 ± 0.0379 |

### support_pages

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8808 ± 0.0030 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| ColPali + MLP reranker | 0.8875 ± 0.0041 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |
| EvidenceGraph-RAG (page+region) | 0.9480 ± 0.0441 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6102 ± 0.0058 |

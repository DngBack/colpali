# Multi-seed Phase 1 summary

Runs: **5**  |  Seeds: `[42, 43, 44, 45, 46]`

### retrieval

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5462 ± 0.0255 | 0.8848 ± 0.0245 | 1.0000 ± 0.0000 | 0.6937 ± 0.0198 | 0.7673 ± 0.0153 |
| ColPali + MLP reranker | 0.5399 ± 0.0329 | 0.8906 ± 0.0146 | 1.0000 ± 0.0000 | 0.6931 ± 0.0187 | 0.7672 ± 0.0140 |
| EvidenceGraph-RAG (page+region) | 0.6925 ± 0.0427 | 0.9852 ± 0.0081 | 1.0000 ± 0.0000 | 0.8312 ± 0.0227 | 0.8744 ± 0.0168 |

### support_pages

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8848 ± 0.0245 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |
| ColPali + MLP reranker | 0.8906 ± 0.0146 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |
| EvidenceGraph-RAG (page+region) | 0.9852 ± 0.0081 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |

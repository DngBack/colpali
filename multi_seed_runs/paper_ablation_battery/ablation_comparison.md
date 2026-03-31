# Ablation comparison (multi-seed aggregates)

Battery root: `/home/admin1/Desktop/colpali/multi_seed_runs/paper_ablation_battery`

## Primary reranker per experiment

| Experiment | Method | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| page_gat_default | X-PageRerank (GAT) | 0.5808 ± 0.0212 | 0.9204 ± 0.0077 | 0.7272 ± 0.0132 | 0.7937 ± 0.0100 |
| page_gat_sparse_graph | X-PageRerank (GAT) | 0.5862 ± 0.0288 | 0.9146 ± 0.0126 | 0.7285 ± 0.0194 | 0.7945 ± 0.0147 |
| region_grid_1x1 | EvidenceGraph-RAG (page+region) | 0.6356 ± 0.0180 | 0.9776 ± 0.0076 | 0.7934 ± 0.0179 | 0.8458 ± 0.0138 |
| region_grid_3x3 | EvidenceGraph-RAG (page+region) | 0.6038 ± 0.0461 | 0.9480 ± 0.0441 | 0.7583 ± 0.0483 | 0.8182 ± 0.0379 |
| region_lambda_fixed_mid | EvidenceGraph-RAG (page+region) | 0.5977 ± 0.0359 | 0.9507 ± 0.0324 | 0.7540 ± 0.0390 | 0.8151 ± 0.0305 |
| region_main_default | EvidenceGraph-RAG (page+region) | 0.6153 ± 0.0322 | 0.9584 ± 0.0325 | 0.7715 ± 0.0384 | 0.8286 ± 0.0302 |
| region_sparse_graph | EvidenceGraph-RAG (page+region) | 0.6233 ± 0.0277 | 0.9684 ± 0.0251 | 0.7804 ± 0.0296 | 0.8356 ± 0.0232 |
| region_split_70_15_15 | EvidenceGraph-RAG (page+region) | 0.6764 ± 0.0436 | 0.9861 ± 0.0090 | 0.8236 ± 0.0200 | 0.8687 ± 0.0146 |

## Page-branch: GAT vs no-graph (Recall@1)

| Experiment | X-PageRerank (GAT) | no-graph |
| --- | --- | --- |
| page_gat_default | 0.5808 ± 0.0212 | 0.5816 ± 0.0168 |
| page_gat_sparse_graph | 0.5862 ± 0.0288 | 0.5784 ± 0.0219 |

## ColPali stage-1 Recall@1 (sanity / split effects)

| Experiment | ColPali R@1 |
| --- | --- |
| page_gat_default | 0.5415 ± 0.0088 |
| page_gat_sparse_graph | 0.5415 ± 0.0088 |
| region_grid_1x1 | 0.5415 ± 0.0088 |
| region_grid_3x3 | 0.5415 ± 0.0088 |
| region_lambda_fixed_mid | 0.5415 ± 0.0088 |
| region_main_default | 0.5415 ± 0.0088 |
| region_sparse_graph | 0.5415 ± 0.0088 |
| region_split_70_15_15 | 0.5462 ± 0.0255 |

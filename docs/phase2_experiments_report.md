# Phase 2 Experiments Report (Metric Fix + Sparse-Graph Preset)

## Thong tin chung
- Ngay thuc hien: 2026-03-26
- Nguoi thuc hien: AI coding assistant + nguoi dung
- Muc tieu: sua pipeline danh gia de on dinh hon va bo sung preset graph de giam nhieu canh trong GAT reranker.

## Boi canh
Trong ket qua truoc do (`results/phase1_real_val_doc_disjoint/phase1_results.json`), mo hinh `X-PageRerank (GAT)` chua on dinh so voi `ablation: no graph`:
- Recall@1: GAT = 0.3871, no-graph = 0.5161
- MRR@10: GAT = 0.6371, no-graph = 0.7124
- nDCG@10: GAT = 0.7288, no-graph = 0.7849

Dong thoi metric `AllSupportHit@k` bi `NaN` tren split hien tai, gay kho cho viec so sanh bao cao.

## Noi dung trien khai trong Experiment 2

### 1) Fix metric `AllSupportHit@k` khong con NaN
- File sua: `experiments/eval/eval_support_pages.py`
- Ham sua: `both_support_hit_at_k(...)`
- Thay doi:
  - Truoc: neu khong co query du dieu kien (`eligible == 0`) thi tra `NaN`
  - Sau: tra `0.0`

Y nghia:
- JSON ket qua va bang tong hop khong con gia tri NaN.
- Duy tri tinh so sanh duoc giua cac lan chay.

### 2) Bo sung preset `sparse-graph`
- File sua: `experiments/run_phase1.py`
- Bo sung:
  - Ham `_resolve_graph_hparams(args)`
  - Tham so CLI moi: `--graph_preset {default,sparse-graph}` cho ca `train` va `eval`
- Quy tac preset:
  - `default`: giu nguyen `--sem_threshold`, `--adj_max_gap`
  - `sparse-graph`:
    - `sem_threshold = max(current, 0.75)`
    - `adj_max_gap = min(current, 1)`

Y nghia:
- Tang do chat luong canh semantic (chi giu canh co tuong dong cao hon).
- Han che canh adjacency qua rong.
- Muc tieu giam message passing nhieu va giam nguy co over-smoothing/noise trong GAT.

## Ket qua thuc nghiem ghi nhan trong ngay 2026-03-26

### A. Ket qua tu bo ket qua nen (truoc cai tien)
Nguon: `results/phase1_real_val_doc_disjoint/phase1_results.json`
- ColPali (stage-1):
  - Recall@1 = 0.3871, Recall@5 = 0.9032, MRR@10 = 0.5921, nDCG@10 = 0.6909
- ColPali + MLP reranker:
  - Recall@1 = 0.4516, Recall@5 = 0.9677, MRR@10 = 0.6637, nDCG@10 = 0.7475
- X-PageRerank (ablation: no graph):
  - Recall@1 = 0.5161, Recall@5 = 1.0000, MRR@10 = 0.7124, nDCG@10 = 0.7849
- X-PageRerank (GAT):
  - Recall@1 = 0.3871, Recall@5 = 1.0000, MRR@10 = 0.6371, nDCG@10 = 0.7288

Nhan xet:
- GAT chua vuot no-graph tren bo val doc-disjoint nho (31 query).
- Co dau hieu graph hien tai dang dua nhieu canh chua thuc su huu ich.

### B. Ket qua ky thuat sau khi implement Experiment 2
- Da sua code thanh cong.
- Da kiem tra lint cho file sua: khong co loi.
- Tu lan chay tiep theo, `AllSupportHit@k` se khong con `NaN`.
- Co san preset `sparse-graph` de train/eval nhanh ma khong phai sua tay tham so graph.

## Lenh chay de tai lap Experiment 2

### Train voi sparse-graph
```bash
python -m experiments.run_phase1 train \
  --train_cache cache/real_train_split \
  --val_cache cache/real_val_doc_disjoint \
  --output_dir checkpoints/gat_sparse_graph \
  --graph_preset sparse-graph
```

### Eval voi sparse-graph
```bash
python -m experiments.run_phase1 eval \
  --eval_cache cache/real_val_doc_disjoint \
  --checkpoint checkpoints/gat_sparse_graph/best.pt \
  --output_dir results/phase2_sparse_graph \
  --graph_preset sparse-graph
```

## Tieu chi danh gia trong lan chay tiep theo
- Muc tieu uu tien:
  1) Tang `Recall@1` cua GAT
  2) Khong giam `Recall@5` (dang la 1.0)
  3) Cai thien `MRR@10` va `nDCG@10` so voi no-graph
- Bao cao them:
  - So luong query hop le cho metric all-support
  - Pairwise phan tich query nao GAT thang/thua no-graph

## Ket luan ngan
Experiment 2 tap trung vao hai viec practical:
- Lam sach metric output (loai bo NaN de de bao cao, so sanh, va tracking).
- Dua vao mot graph preset mang tinh bao thu (sparse) de giam nhieu canh.

Hai thay doi nay tao nen mot nen danh gia va tuning on dinh hon cho vong train/eval tiep theo cua X-PageRerank.

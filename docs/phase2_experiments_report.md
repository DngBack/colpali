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

---

## Ke hoach "ket qua hoan chinh" (Train/Test nhieu hon, tach tu val, khong trung)

Yeu cau:
- Tat ca du lieu train/val/test deu cat ra tu nguon `val` (candidate pool lon).
- Khong trung nhau giua cac split theo:
  - `doc_id`
  - `question_id`

Da bo sung script:
- `experiments/train/make_disjoint_splits.py`

Script nay:
- Nhan 1 cache nguon (da encode tu `--split val` voi so mau lon).
- Chia theo **nhom doc_id** (de dam bao doc disjoint).
- Kiem tra them **question_id overlap = none**.
- Xuat:
  - `output_root/train`
  - `output_root/val`
  - `output_root/test`
  - `output_root/split_summary.json`

### Quy trinh de xay bo ket qua day du

1) Tao candidate cache lon tu val:
```bash
python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/val_candidate_large \
  --top_k 10 --batch_size 4 \
  --num_samples 5000 --sample_offset 0 \
  --cache_chunk_rows 512
```

### Giai thich: vi sao `num_samples` lon (vd >2000) de bi "crash"

**Khong phai do het dong val:** split `val` cua `lmms-lab/MP-DocVQA` (parquet) co khoang **5187** hang, nen `--num_samples 3000` van hop le ve mat du lieu.

**Nguyen nhan chinh: RAM dinh (OOM) khi load dataset.**

Trong `MPDocVQADataset.from_parquet_dir`, moi hang duoc parse thanh mot `MPDocVQASample` voi **tat ca anh trang** (PIL, toi `max_pages`, mac dinh 60) giu trong RAM. Neu ban chon `--num_samples` rat lon **mot lan**, ban se giu dong thoi:

`num_samples x (so trang/doc) x kich thuoc anh decoded`

Ngay khi vuot nguong RAM may / cgroup / `ulimit`, process co the:
- bi **SIGKILL** (Linux OOM killer) — khong co Python stack trace
- hoac `MemoryError`

**Cach xu ly da tich hop trong code:** lenh `cache` co tham so `--cache_chunk_rows` (mac dinh **512**). No xu ly parquet (hoac HF + `--num_samples`) theo tung doan: moi doan chi decode + encode mot phan hang, sau do `del` dataset va `gc.collect()`, nen **dinh RAM thap hon** so voi load het mot lan.

Neu muon tat chunking (hanh vi cu, de OOM khi `--num_samples` rat lon): dung `--cache_chunk_rows 0`.

2) Tach train/val/test disjoint nghiem ngat:
```bash
python -m experiments.train.make_disjoint_splits \
  --candidate_cache cache/val_candidate_large \
  --output_root cache/val_based_splits_strict \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
  --seed 42
```

3) Train tren split moi:
```bash
python -m experiments.run_phase1 train \
  --train_cache cache/val_based_splits_strict/train \
  --val_cache   cache/val_based_splits_strict/val \
  --output_dir  checkpoints/gat_val_based_strict \
  --graph_preset sparse-graph
```

4) Eval tren test split (khong trung doc/question):
```bash
python -m experiments.run_phase1 eval \
  --eval_cache  cache/val_based_splits_strict/test \
  --checkpoint  checkpoints/gat_val_based_strict/best.pt \
  --output_dir  results/phase2_val_based_strict_test \
  --graph_preset sparse-graph
```

### Checklist xac nhan split hop le
- Kiem tra file `cache/val_based_splits_strict/split_summary.json`:
  - `constraints.doc_id_overlap = "none"`
  - `constraints.question_id_overlap = "none"`
- Dam bao train/val/test deu co du so sample (khong qua nho).

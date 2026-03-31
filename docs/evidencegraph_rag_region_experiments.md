# Thực nghiệm EvidenceGraph-RAG (`reranker_type=region`)

Tài liệu này mô tả **thiết lập, hyperparameter và kết quả** cho nhánh reranker **page + region + typed evidence** (EvidenceGraph-RAG), dựa trên các lệnh và artifact có trong repo (cache `mpdoc`, sweep seed 42 trên lưới 2×2, multi-seed 42–46 với cấu hình “best”, so sánh baseline X-PageRerank GAT).

---

## 1. Phương pháp (rút gọn)

- **Đầu vào**: multi-vector ColPali-style mỗi trang (chuỗi patch ~28×28, `D=128` mặc định theo cache).
- **Region nodes**: chia lưới `grid_rows × grid_cols` trên lưới patch; mỗi ô = mean-pool các patch trong ô (`extract_region_vectors` trong `experiments/models/region_graph_reranker.py`).
- **Typed proxy nodes** (bật mặc định `include_typed_nodes=True`): bốn vector heuristic theo dải ảnh — **header**, **text_block**, **caption**, **table** (`extract_typed_vectors`).
- **Đồ thị**: page↔page (semantic theo `sem_threshold` + liền trang), page↔region, region↔region trong trang (và cùng hàng/cột lưới mạnh hơn), liên kết kiểu **header–text**, **caption–table**, typed liền trang; tùy chọn **cross-page region** (`--cross_page_region_edges`) với ngưỡng cosine `sem_threshold_region`.
- **Type embedding**: `nn.Embedding(7, hidden_dim)` — các type: page, region, header, text, caption, table, query; cộng vào đặc trưng node trước GAT (`type_emb` trong `RegionGraphReranker`).
- **Đầu ra**: chỉ **điểm rerank K trang** (logits ở các node page); region/typed chỉ là trung gian message passing.

---

## 2. Môi trường & dữ liệu

| Hạng mục | Giá trị / ghi chú |
| --- | --- |
| Backbone truy vấn / trang | ColPali (vidore/colpali-v1.2 trong docstring `run_phase1`; cache thực tế theo `meta.json` của `--candidate_cache`) |
| Bộ dữ liệu gợi ý | MP-DocVQA; candidate cache đã mã hóa top-K (ví dụ `cache/mpdoc_val_full`) |
| Train / val / test | **Doc-disjoint** qua `experiments.train.make_disjoint_splits`: mặc định `train_ratio=0.70`, `val_ratio=0.15`, `test_ratio=0.15`, seed riêng mỗi lần chạy multi-seed |
| `top_k` ứng viên | **20** (train/eval multi-seed; phải khớp lúc build cache) |
| `eval_k` (metric @k) | **10** |
| Thiết bị | CUDA khi huấn luyện/đánh giá (theo code trainer/inferencer) |

**Preset đồ thị page-level** `--graph_preset sparse-graph`: `sem_threshold = max(sem_threshold, 0.75)`, `adj_max_gap = min(adj_max_gap, 1)` (ít cạnh semantic hơn, giữ liền trang). Cả baseline GAT và region đều dùng preset này trong các run so sánh đã ghi lại.

---

## 3. Xây cache (một lần)

Ví dụ theo docstring `experiments/run_phase1.py` (điều chỉnh `output_dir` cho đúng folder dùng trong project, ví dụ `cache/mpdoc_val_full`):

```bash
python3 -m experiments.run_phase1 cache \
    --split validation \
    --model_name vidore/colpali-v1.2 \
    --model_type colpali \
    --output_dir cache/mpdoc_val_full \
    --top_k 20
```

Cache mỗi mẫu gồm `*_q.pt`, `*_p.pt`, `*_s0.pt`, `*_mask.pt` và `meta.json` (`RerankDataset`).

---

## 4. Hyperparameter (tổng hợp)

### 4.1. CLI `run_phase1 train` / `eval` (region)

| Tham số | Mặc định code | Ghi chú |
| --- | --- | --- |
| `--reranker_type` | `page` | Đặt `region` cho EvidenceGraph-RAG |
| `--graph_preset` | `default` | So sánh chính: `sparse-graph` |
| `--sem_threshold` | `0.65` | Sau preset → **0.75** nếu `sparse-graph` |
| `--adj_max_gap` | `1` | Giữ nguyên với preset |
| `--grid_rows`, `--grid_cols` | `2`, `2` | Sweep chính: lưới **2×2** |
| `--sem_threshold_region` | `0.70` | Ngưỡng cosine cho cạnh region (và cross-page region) |
| `--cross_page_region_edges` | tắt | Bật bằng flag |
| `--num_epochs` | `10` | Multi-seed “best”: **20** |
| `--batch_size` | `32` | |
| `--lr` | `1e-4` | Multi-seed “best”: **5e-5** |
| `--hidden_dim`, `--num_layers`, `--num_heads` | `256`, `2`, `4` | Giống GAT page |
| `--lambda_mix_start`, `--lambda_mix_end` | `0.20`, `0.60` | Multi-seed “best`: **0.15**, **0.55** |
| `--lambda_mix_warmup_steps` | `800` | Multi-seed “best`: **1200** |
| `--top_k` (train) | `20` | |
| `--k` (eval) | `10` | |

### 4.2. `RegionGATConfig` (model; không hết lộ ra CLI)

| Trường | Mặc định |
| --- | --- |
| `input_dim` | `128` |
| `hidden_dim` / `output_dim` | `256` / `64` |
| `num_heads` / `num_layers` | `4` / `2` |
| `dropout` | `0.1` |
| `include_typed_nodes` | `True` |
| `typed_node_scale` | `1.0` |

---

## 5. Lệnh đã dùng (mẫu tái hiện)

### 5.1. Multi-seed doc-disjoint + train + eval (cấu hình “best” theo ghi chép 2026-03-30)

Cấu hình: `reranker_type=region`, lưới 2×2, `sem_threshold_region=0.70`, không cross-page region, `sparse-graph`, `num_epochs=20`, `lr=5e-5`, lịch `lambda_mix` 0.15→0.55 (warmup 1200 bước), seeds `42..46`, root `multi_seed_runs/mpdoc_val_disjoint_region_best`.

```bash
python3 -m experiments.run_phase1_multi_seed \
    --candidate_cache cache/mpdoc_val_full \
    --root multi_seed_runs/mpdoc_val_disjoint_region_best \
    --graph_preset sparse-graph \
    --reranker_type region \
    --seeds 42,43,44,45,46 \
    --num_epochs 20 \
    --train_batch_size 32 \
    --top_k 20 \
    --eval_k 10 \
    --lr 5e-5 \
    --grid_rows 2 \
    --grid_cols 2 \
    --sem_threshold_region 0.70 \
    --lambda_mix_start 0.15 \
    --lambda_mix_end 0.55 \
    --lambda_mix_warmup_steps 1200
```

**Gộp kết quả** (mặc định cuối script; hoặc chạy tay):

```bash
python3 -m experiments.summarize_phase1_multi_seed \
    --glob 'multi_seed_runs/mpdoc_val_disjoint_region_best/seed_*/eval/phase1_results.json' \
    --out_json multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_results.json \
    --out_md multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_report.md
```

Artifact có thêm bản **mean ± std** (trong repo):  
`aggregated_report_with_percentiles.md`, `aggregated_results_with_percentiles.json` (cùng thư mục root).

### 5.2. Baseline X-PageRerank (GAT), cùng gia đình split

```bash
python3 -m experiments.run_phase1_multi_seed \
    --candidate_cache cache/mpdoc_val_full \
    --root multi_seed_runs/mpdoc_val_disjoint \
    --graph_preset sparse-graph \
    --reranker_type page \
    --seeds 42,43,44,45,46
```

(Tùy chỉnh `--num_epochs`, `--lr`, v.v. cho đúng run baseline đã lưu; bảng kết quả dưới lấy từ `aggregated_report_with_percentiles.md` hiện có.)

### 5.3. Sweep cục bộ (seed 42, lưới 2×2)

Đã có thư mục kết quả eval dưới `results/region_s42_g2_thr{0.65,0.70,0.75}_cross{off,on}/` (mỗi nơi `phase1_results.json`). Huấn luyện/đánh giá tương ứng qua `run_phase1 train` / `eval` với `--reranker_type region`, `--grid_rows 2`, `--grid_cols 2`, chỉnh `--sem_threshold_region` và tùy chọn `--cross_page_region_edges`.

---

## 6. Kết quả (bảng)

### 6.1. Sweep seed 42 — EvidenceGraph-RAG (R@1, R@5, MRR@10, nDCG@10)

Đọc từ `results/region_s42_g2_thr*_cross*/phase1_results.json` (metric của method `EvidenceGraph-RAG (page+region)`).

| Cấu hình | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| thr=0.65, cross off | 0.7165 | 0.9813 | 0.8437 | 0.8836 |
| thr=0.65, cross **on** | 0.7353 | 0.9813 | 0.8530 | 0.8904 |
| **thr=0.70, cross off** | **0.7568** | **0.9813** | **0.8630** | **0.8977** |
| thr=0.70, cross on | 0.7396 | 0.9842 | 0.8544 | 0.8913 |
| thr=0.75, cross off | 0.7468 | 0.9856 | 0.8579 | 0.8939 |
| thr=0.75, cross on | 0.7065 | 0.9856 | 0.8372 | 0.8787 |

**Lựa chọn cho multi-seed**: `sem_threshold_region=0.70`, **tắt** cross-page region (cao nhất R@1 / MRR@10 / nDCG@10 trong lưới sweep này).

### 6.2. Multi-seed (5 seed) — EvidenceGraph-RAG “best config” từng seed

`multi_seed_runs/mpdoc_val_disjoint_region_best/seed_*/eval/phase1_results.json`.

| Seed | Recall@1 | Recall@5 | MRR@10 | nDCG@10 | Coverage@5 | SupportF1@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7612 | 0.9813 | 0.8646 | 0.8988 | 0.9813 | 0.6410 |
| 43 | 0.7030 | 0.9973 | 0.8442 | 0.8845 | 0.9973 | 0.6189 |
| 44 | 0.6599 | 0.9783 | 0.8097 | 0.8582 | 0.9783 | 0.6085 |
| 45 | 0.6820 | 0.9793 | 0.8208 | 0.8662 | 0.9793 | 0.5696 |
| 46 | 0.6563 | 0.9897 | 0.8169 | 0.8641 | 0.9897 | 0.6266 |

### 6.3. Multi-seed — tổng hợp mean ± std (báo cáo tự động)

Từ `multi_seed_runs/mpdoc_val_disjoint_region_best/aggregated_report_with_percentiles.md` và `multi_seed_runs/mpdoc_val_disjoint/aggregated_report_with_percentiles.md` (cùng seeds 42–46).

**Retrieval**

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.5462 ± 0.0255 | 0.8848 ± 0.0245 | 1.0000 ± 0.0000 | 0.6937 ± 0.0198 | 0.7673 ± 0.0153 |
| ColPali + MLP | 0.5399 ± 0.0329 | 0.8906 ± 0.0146 | 1.0000 ± 0.0000 | 0.6931 ± 0.0187 | 0.7672 ± 0.0140 |
| **EvidenceGraph-RAG (page+region)** | **0.6925 ± 0.0427** | **0.9852 ± 0.0081** | **1.0000 ± 0.0000** | **0.8312 ± 0.0227** | **0.8744 ± 0.0168** |
| X-PageRerank (GAT) | 0.6145 ± 0.0382 | 0.9248 ± 0.0068 | 1.0000 ± 0.0000 | 0.7497 ± 0.0245 | 0.8107 ± 0.0183 |

**Support-page**

| Method | Coverage@5 | Coverage@10 | AllSupportHit@10 | SupportF1@10 |
| --- | --- | --- | --- | --- |
| ColPali (stage-1) | 0.8848 ± 0.0245 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |
| ColPali + MLP | 0.8906 ± 0.0146 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |
| **EvidenceGraph-RAG** | **0.9852 ± 0.0081** | **1.0000 ± 0.0000** | **0.0000 ± 0.0000** | **0.6129 ± 0.0270** |
| X-PageRerank (GAT) | 0.9248 ± 0.0068 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6129 ± 0.0270 |

**Chênh lệch mean (EvidenceGraph − GAT)** trên cùng báo cáo: ΔRecall@1 **+0.0780**, ΔRecall@5 **+0.0604**, ΔMRR@10 **+0.0815**, ΔnDCG@10 **+0.0637**, ΔCoverage@5 **+0.0604** (xem `docs/experiment_note_2026-03-30_region_multiseed.md`).

### 6.4. Ghi chú về các báo cáo region cũ hơn

`docs/region_multiseed_experiments_report.md` so sánh aggregate **khác** (`aggregated_report_region.md`, hyperparameter/chương trình huấn luyện sớm hơn) và cho thấy lúc đó region **chưa** vượt GAT trên R@1 / MRR@10. Sau khi chỉnh lịch `lambda_mix`, LR, số epoch và sweep `sem_threshold_region` / cross-page, bộ `mpdoc_val_disjoint_region_best` phản ánh kết quả mạnh hơn trong bảng §6.3.

---

## 7. Lưu ý vận hành

- **Ngắt tay (KeyboardInterrupt)** giữa các cấu hình trong một sweep dài: các run **đã hoàn thành** trước khi dừng vẫn hợp lệ (ghi nhận trong `docs/experiment_note_2026-03-30_region_multiseed.md`).
- **Khớp cache**: `--candidate_cache` phải là thư mục thật có `meta.json`; `top_k` khi train/eval khớp lúc build cache.
- Một số file dưới `multi_seed_runs/` có thể bị công cụ IDE chặn đọc trực tiếp; metric vẫn truy cập qua shell hoặc JSON như trên.

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

---

## 8. Checklist để đạt mức paper submission chắc tay hơn

Mục tiêu của checklist này là tách rõ:

- cái gì đã đủ để viết paper,
- cái gì còn thiếu nếu muốn claim mạnh hơn,
- và thứ tự ưu tiên để không bị sa vào chạy thí nghiệm lan man.

### 8.1. Must-have trước khi submit

1. Xác nhận lại claim chính của paper. Nên chốt một claim hẹp, có thể bảo vệ tốt, ví dụ: `query-conditioned local evidence graph improves top-rank reranking on multi-page document QA`. Tránh claim quá rộng kiểu “đã giải quyết cross-page reasoning tổng quát” nếu chưa có QA end-to-end và multi-support subset đủ mạnh.

2. Chốt protocol chính thức cho tất cả số liệu. Cần cố định seed, cố định family split, cố định quy tắc chọn config, và ghi rõ run nào là baseline, run nào là tuned region model.

3. Bổ sung ablation tối thiểu để chứng minh lợi ích của graph. Tối thiểu nên có `stage-1`, `+ MLP reranker`, `page graph` không region, `page + region`, `page + region + typed nodes`, và `page + region + typed nodes + cross-page edges` nếu có.

4. Làm ablation theo từng thành phần kiến trúc. Cần kiểm tra bỏ `query node`, bỏ `typed nodes`, bỏ `semantic edges`, bỏ `adjacency edges`, đổi `grid_rows x grid_cols` giữa `2x2` và `3x3`, sweep `sem_threshold_region`, và sweep `lambda_mix` cùng `warmup`.

5. Chứng minh tính ổn định theo nhiều seed. Báo cáo `mean ± std`, thêm median hoặc percentile nếu cần, và nếu có thể thì thêm paired comparison theo từng seed.

6. Thêm kiểm định thống kê tối thiểu. Dùng bootstrap CI hoặc paired test trên per-query/per-seed scores, rồi báo cáo rõ chênh lệch mean và độ tin cậy.

7. Làm rõ tính hợp lệ của metric support. Báo cáo phân bố `num_support_pages`, nêu rõ `AllSupportHit@k = 0` hiện tại là do split chủ yếu single-support hay do pipeline, và nếu muốn dùng metric này như claim thì cần một multi-support subset thật sự.

### 8.2. Should-have để paper thuyết phục hơn

1. Thêm end-to-end QA metrics. Nên có `EM`, `F1`, hoặc `ANLS` tùy benchmark; nếu pipeline chưa sinh câu trả lời thì ít nhất phải có downstream proxy rõ ràng để chứng minh reranking có ích cho QA thật.

2. Thêm dataset thứ hai. Ít nhất một benchmark nữa ngoài MP-DocVQA, ví dụ DUDE, để chứng minh phương pháp không chỉ khớp riêng một split.

3. Thêm hard subset analysis. Nên tách multi-page vs single-page, page-distance lớn vs nhỏ, question có cross-reference/bảng/figure/caption, và nếu có thể thì 2-hop vs 3+ hop.

4. Thêm latency / memory / throughput. Cần đo graph build time, inference latency/query, và peak memory theo `top_k` để giữ câu chuyện “drop-in post-retrieval module”.

5. Thêm qualitative case studies. Nên có 3 đến 5 ví dụ thành công, 3 đến 5 ví dụ thất bại, và một vài hình minh họa top-k pages cùng evidence graph.

### 8.3. Nice-to-have nếu muốn lên level cao hơn

1. So sánh với thêm baseline mạnh hơn. Có thể là một reranker transformer không graph, một biến thể heterogeneous graph khác, hoặc một baseline reasoning/graph retrieval gần với literature hơn.

2. Thêm phân tích lỗi có taxonomy. Nên gom lỗi theo nhóm như false positive do OCR noise, false negative do support phân tán, fail vì cross-page link yếu, và fail vì query không cần graph.

3. Báo cáo khả năng tổng quát hóa. Nếu protocol cho phép, hãy train trên một split và test trên split khác, hoặc ít nhất trình bày cross-seed generalization rõ hơn.

4. Thêm calibration hoặc confidence analysis. Có thể xem phân phối score của page support và độ tin cậy khi graph giúp hay làm hỏng ranking.

### 8.4. Decision gate trước submission

Nếu muốn tự hỏi “đã đủ paper chưa?”, dùng gate sau:

1. Nếu chỉ có retrieval metric và support coverage thì bài báo đang ở mức `good technical report`.
2. Nếu có retrieval, ablation đầy đủ, và multi-seed ổn định thì bài báo đã ở mức `submission được`.
3. Nếu có thêm QA end-to-end, dataset thứ hai, hard subset, và efficiency thì bài báo ở mức `chắc tay hơn nhiều`.

### 8.5. Ưu tiên chạy tiếp theo

Thứ tự mình khuyên là:

1. Chốt ablation table chính thức.
2. Chạy multi-support subset hoặc dataset thứ hai.
3. Bổ sung QA metrics.
4. Đo latency / memory.
5. Làm case study và failure analysis.

### 8.6. Kết luận thực dụng

Hiện tại, kết quả đã đủ để viết phần `method` và `retrieval experiment` của paper. Điểm còn thiếu để submission `chắc tay` là evidence-level validation thật sự, ablation đầy đủ, một benchmark bổ sung hoặc hard subset, và end-to-end QA metrics. Nói ngắn gọn: **paper có thể viết, nhưng để paper mạnh thì vẫn nên làm thêm một vòng thí nghiệm theo checklist này**.

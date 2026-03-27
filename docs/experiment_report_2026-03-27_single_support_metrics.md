# Báo cáo thực nghiệm — 2026-03-27

## Thông tin chung

| Mục | Nội dung |
|-----|----------|
| **Ngày** | 27/03/2026 |
| **Người thực hiện** | Người vận hành pipeline trên repo `colpali` (workspace); hỗ trợ ghi chép: trợ lý AI |
| **Mục đích** | Ghi nhận phát hiện về **suy biến metric** khi mọi mẫu trong tập đánh giá có `num_support_pages = 1`, và định hướng xử lý (không nhầm với lỗi implement thuần túy) |

## Bối cảnh quan sát

Sau khi chạy đánh giá Phase 1, file kết quả:

`results/phase2_val_based_1500_test/phase1_results.json`

trong phần phân tích theo từng truy vấn cho thấy:

- Mọi dòng `per_query` đều có `"num_support_pages": 1`.
- Khối `analysis.by_support_count` chỉ có bucket `"1"` với `n_queries: 237` (không có nhóm 2, 3, … trang hỗ trợ).

Điều này **không phải** do code gán sai số hỗ trợ một cách ngẫu nhiên: `num_support_pages` được tính từ tập gold thực tế, `len(set(gold_pages))` (xem luồng trong `experiments/run_phase1.py`, hàm phân tích `per_query_first_hit`).

## Vì sao Coverage@k và Recall@k trùng số?

Định nghĩa trong code:

- **Recall@k (stage retrieval)** — biến thể **any-hit**: tỷ lệ câu có **ít nhất một** trang gold nằm trong top-*k* (`experiments/eval/eval_retrieval.py`, `recall_at_k`).
- **Coverage@k** — trung bình trên các câu của tỷ lệ  
  `|pred_topk ∩ gold| / |gold|` (`experiments/eval/eval_support_pages.py`, `evidence_coverage`).

Khi **mỗi câu chỉ có đúng một trang gold** (`|gold| = 1`):

- `|pred ∩ gold| / |gold|` bằng `1` nếu trang đó có trong top-*k*, và `0` nếu không — tức cùng một hàm chỉ báo với **any-hit Recall@k** trên từng câu.
- Trung bình macro trên toàn tập vì vậy **khớp** Recall@k any-hit.

Kết luận: đây là **hệ quả toán học** của bộ dữ liệu đánh giá (toàn câu 1-trang-gold), không phải lỗi in sai hai cột.

## Vì sao AllSupportHit@k = 0.0 cho mọi phương pháp?

Metric **AllSupportHit@k** (`both_support_hit_at_k` trong `experiments/eval/eval_support_pages.py`) **chỉ tính trên các câu có ít nhất 2 trang hỗ trợ** (`min_support_pages=2` mặc định): cần mọi trang gold đều xuất hiện trong top-*k*.

Trên tập 237 câu đều `num_support_pages = 1`:

- Không có câu “đủ điều kiện” (eligible = 0).
- Sau chỉnh sửa Experiment 2, hàm trả về **0.0** thay vì `NaN` — bảng kết quả hiển thị `0.0` là **đúng với định nghĩa**, nhưng **không mang nghĩa** “mô hình luôn thất bại ở multi-page”; đơn giản là **không có câu multi-support để đo**.

## Công việc đã làm (ghi nhận theo phiên làm việc)

1. **Phân tích** file `phase1_results.json` (run `phase2_val_based_1500_test`).
2. **Đối chiếu code** `eval_retrieval.py`, `eval_support_pages.py`, `run_phase1.py` để xác nhận định nghĩa metric và nguồn `num_support_pages`.
3. **Kết luận**: cần tách bạch “lỗi code” và “giới hạn tập đánh giá / lựa chọn split”.

*(Chưa triển khai thay đổi code mới trong ngày này — báo cáo tập trung vào chẩn đoán và khuyến nghị.)*

## Pipeline và lệnh (tái lập mức khái niệm)

Tên thư mục kết quả gợi ý quy trình: cache từ split `val`, tách train/val/test **doc-disjoint & question-disjoint**, huấn luyện GAT (có thể với `--graph_preset sparse-graph`), rồi `eval` trỏ vào cache **test**. Minh họa lệnh (đường dẫn cần khớp máy bạn):

```bash
# 1) Cache ứng viên (đã dùng chunking / maxsim tuỳ máy)
export PARQUET_DIR="/path/to/MP-DocVQA/parquet/data"
python -m experiments.run_phase1 cache \
  --split val --offline \
  --parquet_dir "$PARQUET_DIR" \
  --model_name vidore/colSmol-256M --model_type colidefics3 \
  --output_dir cache/val_candidate_full \
  --top_k 10 --max_pages 30 --batch_size 1 \
  --cache_chunk_rows 20 --maxsim_doc_chunk 4

# 2) Tách split rời doc + question
python -m experiments.train.make_disjoint_splits \
  --candidate_cache cache/val_candidate_full \
  --output_root cache/phase2_disjoint \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --seed 42

# 3) Train
python -m experiments.run_phase1 train \
  --train_cache cache/phase2_disjoint/train \
  --val_cache cache/phase2_disjoint/val \
  --output_dir checkpoints/phase2_gat \
  --graph_preset sparse-graph

# 4) Eval → ghi JSON (ví dụ)
python -m experiments.run_phase1 eval \
  --cache_dir cache/phase2_disjoint/test \
  --checkpoint checkpoints/phase2_gat/best.pt \
  --output_dir results/phase2_val_based_1500_test \
  --graph_preset sparse-graph
```

Các tham số chính xác (số mẫu ~1500, tên thư mục cache) phụ thuộc lần chạy thực tế trên máy; cần đối chiếu `meta.json` / `split_summary.json` trong từng cache.

## Dataset / tập đánh giá (theo `phase1_results.json`)

| Thuộc tính | Giá trị ghi nhận |
|------------|------------------|
| Số truy vấn | 237 |
| Phân bố `num_support_pages` | Chỉ nhóm **1** (237 câu) |
| Hệ quả | Coverage@k ≈ Recall@k any-hit; AllSupportHit@k không có nền để đo (0.0) |

**Ám chỉ tên thư mục** `val_based_1500_test`: có thể tham khảo ~1500 mẫu khi **tạo cache / split**, nhưng **tập cuối cùng đưa vào eval** (sau lọc, lỗi cache, v.v.) còn 237 câu — cần mở `meta.json` của cache test để khớp con số.

## Kết quả chỉ số chính (trích từ cùng file JSON)

**Retrieval** (làm tròn ~3 chữ số thập phân):

| Phương pháp | Recall@1 | MRR@10 | nDCG@10 |
|-------------|----------|--------|---------|
| ColPali (stage-1) | 0.506 | 0.675 | 0.754 |
| ColPali + MLP reranker | 0.485 | 0.656 | 0.739 |
| X-PageRerank (ablation: no graph) | 0.667 | 0.775 | 0.830 |
| X-PageRerank (GAT) | 0.717 | 0.800 | 0.848 |

Với `k = 1`, trong bảng **support_pages**, **Coverage@k** trùng **Recall@k** từng phương pháp (xem mục “Vì sao Coverage@k và Recall@k trùng số?”). **AllSupportHit@1/5/10** = `0.0` vì không có câu multi-support (xem mục AllSupportHit).

**Support-page (ví dụ @10, GAT):** Coverage@10 = 1.0, SupportF1@10 ≈ 0.613 — khi mỗi câu 1 gold, SupportR@10 = 1 nếu mọi câu đều hit trong top-10 (đúng với Recall@10 = 1).

## Khuyến nghị (để metric “cross-page” có ý nghĩa)

1. **Kiểm tra `meta.json` của cache test** trước khi eval: đếm số mẫu có `len(page_indices chỉ ground-truth)` / trường tương đương ≥ 2 (tuỳ cách lưu trong meta). Nếu toàn bộ là 1-trang-gold, nên ghi rõ trong báo cáo rằng AllSupportHit và sự khác biệt Coverage vs Recall **không được kiểm chứng**.
2. **Lọc ứng viên khi build cache** (hoặc lọc split test) để giữ một tỷ lệ tối thiểu câu **đa trang hỗ trợ**, nếu MP-DocVQA cho phép nhận diện từ annotation.
3. **Cải tiến pipeline (tùy chọn, chưa làm trong ngày 27/03)**: khi `evaluate_support_pages` / `run_phase1 eval` phát hiện không có câu nào `len(gold) >= 2`, log cảnh báo rõ ràng và ghi vào JSON (ví dụ `metric_warnings`) để tránh hiểu nhầm `AllSupportHit@k = 0.0`.

## Tham chiếu nhanh code

- Định nghĩa Coverage: `evidence_coverage` trong `experiments/eval/eval_support_pages.py`.
- Định nghĩa Recall any-hit: `recall_at_k` trong `experiments/eval/eval_retrieval.py`.
- AllSupportHit / ngưỡng ≥2 trang: `both_support_hit_at_k` trong `experiments/eval/eval_support_pages.py`.
- `num_support_pages` trong JSON phân tích: `len(gold_set)` trong `per_query_first_hit` (`experiments/run_phase1.py`).
</think>


<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace
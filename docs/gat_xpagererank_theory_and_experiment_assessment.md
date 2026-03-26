# GAT + X-PageRerank: Ly thuyet hoan chinh va danh gia thuc nghiem

## 1. Bai toan va dong co
Trong bai toan page retrieval cho MP-DocVQA, stage-1 (ColPali/ColQwen) tra ve top-K page theo do tuong dong query-page. Cach nay manh o matching truc tiep, nhung co han che:
- Chua khai thac quan he giua cac trang trong cung tai lieu.
- Nhieu cau hoi can bang chung lien trang, can xep lai thu tu duoc uu tien.

X-PageRerank giai quyet bang cach:
1) Lay top-K tu stage-1
2) Xay graph theo query va quan he giua page
3) Dung GAT de message passing
4) Tinh score moi va tron voi score goc de rerank.

## 2. Mo hinh ly thuyet

### 2.1 Node va feature
- Node page: dai dien moi page trong top-K.
- Node query: dai dien query, ket noi den toan bo page.
- Feature page:
  - Embedding page da pool
  - (Tuy chon) vector sai khac voi query: `page - query`
  - (Tuy chon) diem stage-1 da normalize.

### 2.2 Edge
Cac loai canh chinh:
- Semantic edge: noi page co cosine similarity cao.
- Adjacency edge: noi cac page gan nhau theo thu tu trang.
- Query edge: query <-> page.

Y nghia:
- Semantic edge truyen tin bang chung noi dung.
- Adjacency edge truyen tin cau truc tai lieu.
- Query edge de dieu kien hoa toan bo graph theo cau hoi.

### 2.3 GAT update
Moi layer GAT tinh attention theo canh:
- He so canh i->j duoc hoc boi `a^T [W h_i || W h_j]`.
- Softmax tren lang gieng cua i de lay trong so tong hop.
- Multi-head de tang bieu dien.

Sau L lop, moi page co bieu dien da "nhin" duoc thong tin tu page lien quan va query.

### 2.4 Scoring va tron score
- Scoring head sinh `delta_i` cho tung page.
- Final score:
  - `final_i = (1 - lam) * s0_i + lam * delta_i`
  - `lam = sigmoid(lambda_mix)`.

Y nghia:
- Giu tri thuc retrieval stage-1 (`s0`) de tranh pha vo xep hang goc.
- Bo sung dieu chinh theo graph (`delta`) de sua cac vi tri kho.

### 2.5 Objective train
Thuong ket hop:
- Listwise loss: day support pages len trong top-K.
- Pairwise margin loss: buoc positive > negative voi margin.

Ket hop nay can bang ranking toan cuc va phan biet cap doi kho.

## 3. Vi sao can cai tien "sparse-graph"
Tu ket qua thuc nghiem truoc do, GAT thua no-graph o metric dau bang (`Recall@1`) va ranking depth (`MRR`, `nDCG`) tren split nho. Nguyen nhan kha di:
- Graph qua day => nhieu canh nhieu, thong tin nhieu nhieu.
- Attention bi "pha loang", page manh bi trung hoa boi page kem lien quan.
- Query-edge full connect khien mo hinh de hoc shortcut khong on dinh.

Do do, cai tien hop ly la lam graph "thua nhung chat":
- Tang nguong semantic (`sem_threshold`) de loai canh yeu.
- Han che adjacency chi cho canh can ke (`adj_max_gap <= 1`).

Muc tieu:
- Giam noise message passing.
- Tang kha nang giu dung page top-1.
- Cai thien tinh on dinh khi tap val nho.

## 4. Vi sao can sua metric AllSupportHit
`AllSupportHit@k` duoc dinh nghia tren tap query co >= N support pages (thuong N=2). Neu split hien tai chu yeu 1 support page, so query hop le bang 0.

Neu tra `NaN`:
- Kho theo doi tren dashboard/JSON.
- Kho so sanh giua cac lan chay.

Tra `0.0` khi khong co mau hop le:
- Khong lam vo pipeline bao cao.
- Van de doc va tong hop.
- Co the bo sung them thong tin `eligible_count` o buoc tiep theo de minh bach.

## 5. Danh gia thuc nghiem hien tai va giai thich
Ket qua nen (val doc-disjoint hien tai):
- GAT: Recall@1 = 0.3871, MRR@10 = 0.6371, nDCG@10 = 0.7288
- No-graph: Recall@1 = 0.5161, MRR@10 = 0.7124, nDCG@10 = 0.7849

Giai thich:
- Tren split nho, graph dense de overfit/noisy.
- No-graph (ban chat gan MLP + feature tot) lai it rui ro hon.
- Vi vay huong sparse-graph la cai tien co co so ly thuyet va phu hop dau hieu thuc nghiem.

## 6. Gia thuyet kiem chung cho vong tiep theo
Sau khi bat `--graph_preset sparse-graph`, ky vong:
1) Recall@1 cua GAT tang ro nhat.
2) Recall@5 giu nguyen (khong giam so voi moc 1.0 neu co).
3) MRR@10, nDCG@10 cai thien so voi GAT default.

Neu chua cai thien:
- Giam tiep do day canh semantic (tang threshold).
- Dieu chinh weight loss (uu tien listwise hon).
- Regularize `lambda_mix` de khong de graph score lan at `s0`.

## 7. Ket luan
Hai cai tien trong Experiment 2 la hop ly ca ly thuyet lan thuc tien:
- Metric fix giai quyet van de bao cao va so sanh ket qua.
- Sparse-graph preset giai quyet van de cot loi cua GAT tren split nho: nhieu canh gay nhieu.

Huong tiep theo nen la chay lai train/eval voi sparse-graph, bao cao ket qua doi chieu cung no-graph tren cung split, va danh gia theo nhieu seed de dam bao tinh on dinh.

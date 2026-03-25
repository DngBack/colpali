Tên đề tài

Cross-Page Evidence Reranking for Visual Late-Interaction Retrieval in Long PDF Documents

Tên ngắn

X-PageRerank

Một câu mô tả

X-PageRerank là một stage-2 query-conditioned reranker đặt trên ColPali/ColQwen-style visual retrievers, nhằm tổng hợp bằng chứng xuyên trang cho các câu hỏi nhiều bước trên tài liệu PDF dài. Nó không xây layered component graph trên toàn corpus như LILaC, mà chỉ xây evidence graph cục bộ trên top-k trang/region đã retrieve, nhờ đó giữ được tính thực dụng, hiệu quả, và khác biệt học thuật rõ ràng. ColPali vốn là một retriever late-interaction trên ảnh trang tài liệu với code công khai; LILaC thì đi theo hướng layered component graph và subgraph retrieval tổng quát cho multimodal multihop retrieval.

1. Tại sao đề tài này vẫn đáng làm sau LILaC?
Điểm tránh trùng

LILaC giải bài toán open-domain multimodal multihop retrieval bằng:

layered component graph coarse/fine,
subgraph retrieval,
query decomposition,
beam-search traversal.

Đề tài của bạn sau khi chỉnh lại sẽ không cạnh tranh trực diện ở trục đó. Nó khác ở 4 điểm:

(a) Backbone cố định là visual late-interaction retriever

Ta lấy ColPali/ColQwen làm stage-1 retriever, thay vì phát triển một framework retrieval graph tổng quát từ đầu. Repo ColPali hiện đã hỗ trợ training/inference cho các ColVision models như ColPali và ColQwen2.

(b) Graph chỉ xây ở query-time trên top-k candidates

Không có corpus-wide component graph. Điều này làm bài toán gọn hơn rất nhiều và khác bản chất với LILaC.

(c) Trọng tâm là cross-page evidence composition

Không nhắm vào multimodal component retrieval nói chung, mà nhắm vào một failure mode rất cụ thể của visual page retrieval:

retrieve được từng trang tốt, nhưng không biết ghép bằng chứng giữa nhiều trang để trả lời câu hỏi multi-hop.

(d) Claim chính là drop-in augmentation

Đề tài của bạn là:

thêm một post-retrieval reranker đơn giản, hiệu quả, dễ gắn vào ColPali.

Đây là một story thực dụng và sạch hơn.

2. Bài toán nghiên cứu
Thiết lập

Cho một tập tài liệu PDF dài, mỗi tài liệu gồm nhiều trang ảnh:

𝐷
=
{
𝑝
1
,
𝑝
2
,
…
,
𝑝
𝑛
}
D={p
1
	​

,p
2
	​

,…,p
n
	​

}

với 
𝑝
𝑖
p
i
	​

 là ảnh của trang thứ 
𝑖
i.

Cho truy vấn 
𝑞
q, stage-1 retriever 
𝑅
R kiểu ColPali trả về top-
𝐾
K trang:

𝑃
𝐾
(
𝑞
)
=
{
𝑝
𝑖
1
,
…
,
𝑝
𝑖
𝐾
}
P
K
	​

(q)={p
i
1
	​

	​

,…,p
i
K
	​

	​

}

Mục tiêu không chỉ là tìm một trang đúng, mà là tìm một tập bằng chứng xuyên trang:

𝐸
(
𝑞
)
=
{
𝑒
1
,
𝑒
2
,
…
,
𝑒
𝑚
}
E(q)={e
1
	​

,e
2
	​

,…,e
m
	​

}

sao cho:

các evidence units liên quan thật sự đến câu hỏi,
chúng có thể nằm trên nhiều trang khác nhau,
việc tổng hợp chúng giúp tăng retrieval và downstream QA.
Khó khăn

ColPali rất mạnh ở page-level visual relevance, nhưng không được thiết kế chuyên biệt để:

nối biểu đồ ở trang A với đoạn giải thích ở trang B,
nối caption ở trang X với bảng ở trang Y,
ghép heading, figure, paragraph, footnote thành một chain bằng chứng.
3. Giả thuyết trung tâm
Hypothesis

Cross-page reasoning nên được đặt ở query-time, không phải index-time toàn cục.

Cụ thể:

Index-time chỉ cần chuẩn bị các evidence units có cấu trúc.
Query-time mới xây graph cục bộ trên top-k pages/regions và lan truyền thông tin giữa chúng.

Điều này có ba lợi ích:

tránh chi phí graph toàn corpus,
giữ nguyên backbone retrieval mạnh có sẵn,
trực tiếp tối ưu cho câu hỏi hiện tại, tức là query-conditioned reasoning.
4. Ý tưởng phương pháp
4.1 Tổng quan pipeline
Bước 1 — Stage-1 visual retrieval

Dùng ColPali/ColQwen để retrieve top-
𝐾
K pages theo late interaction. ColPali là một VLM retriever tạo multi-vector embeddings cho ảnh trang và matching bằng late interaction theo tinh thần ColBERT.

Bước 2 — Evidence unit extraction

Trên top-
𝐾
K trang, trích ra các evidence units:

page node
region node
caption-like block
table-like region
figure-like region
paragraph-like block nếu có parser/OCR nhẹ
Bước 3 — Query-conditioned evidence graph

Xây graph cục bộ:

𝐺
𝑞
=
(
𝑉
𝑞
,
𝐸
𝑞
)
G
q
	​

=(V
q
	​

,E
q
	​

)

với:

𝑉
𝑞
V
q
	​

: nodes thuộc top-
𝐾
K pages
𝐸
𝑞
E
q
	​

: semantic edges + structural edges + cross-page edges
Bước 4 — Graph reranking

Chạy một module nhẹ để tính:

score mới cho page,
score cho evidence set,
hoặc score cho evidence chain.
Bước 5 — Downstream answerer

Đưa top evidence vào VLM/LLM answerer.

5. Thiết kế chi tiết
5.1 Evidence units
Cấp đơn giản nhất

Node = page

Bản này chạy nhanh nhất và vẫn chứng minh được ý tưởng cross-page propagation.

Cấp tốt hơn

Node = page + salient region

Region có thể lấy từ:

fixed grid patches,
saliency hotspots,
OCR/layout blocks,
parser blocks nếu có pipeline parse.
5.2 Các loại cạnh
Semantic edge

Nối hai nodes nếu embedding similarity cao:

𝑤
𝑖
𝑗
𝑠
𝑒
𝑚
=
cos
⁡
(
ℎ
𝑖
,
ℎ
𝑗
)
w
ij
sem
	​

=cos(h
i
	​

,h
j
	​

)
Page adjacency edge

Nối trang liền kề:

(
𝑖
,
𝑗
)
∈
𝐸
if 
∣
𝑝
𝑎
𝑔
𝑒
(
𝑖
)
−
𝑝
𝑎
𝑔
𝑒
(
𝑗
)
∣
=
1
(i,j)∈Eif ∣page(i)−page(j)∣=1
Same-section edge

Nếu parser hoặc heading detection cho thấy cùng section.

Figure-caption / table-paragraph edge

Nếu region loại figure gần caption hoặc table gần explanatory paragraph.

Query relevance edge

Nối query representation với các node để graph mang tính query-conditioned.

6. Hàm điểm
Stage-1 score
𝑠
0
(
𝑞
,
𝑝
𝑖
)
s
0
	​

(q,p
i
	​

)

là điểm ColPali.

Graph-refined node representation

Sau 
𝐿
L bước message passing:

ℎ
𝑖
(
𝐿
)
=
GNN
(
ℎ
𝑖
(
0
)
,
𝐺
𝑞
,
𝑞
)
h
i
(L)
	​

=GNN(h
i
(0)
	​

,G
q
	​

,q)
Final reranking score
𝑠
(
𝑞
,
𝑝
𝑖
)
=
𝑠
0
(
𝑞
,
𝑝
𝑖
)
+
𝜆
⋅
𝑔
(
𝑞
,
ℎ
𝑖
(
𝐿
)
)
s(q,p
i
	​

)=s
0
	​

(q,p
i
	​

)+λ⋅g(q,h
i
(L)
	​

)

hoặc ở mức evidence set:

𝑠
(
𝑞
,
𝐸
)
=
∑
𝑒
∈
𝐸
𝛼
𝑒
 
𝑔
(
𝑞
,
ℎ
𝑒
(
𝐿
)
)
+
𝛽
⋅
Coverage
(
𝐸
)
s(q,E)=
e∈E
∑
	​

α
e
	​

g(q,h
e
(L)
	​

)+β⋅Coverage(E)

Trong đó Coverage thưởng cho evidence đến từ nhiều loại node hoặc nhiều trang nếu chúng bổ trợ nhau.

7. Novelty statement
Đóng góp chính
Một formulation mới cho cross-page visual retrieval như một bài toán post-retrieval evidence composition thay vì corpus-wide graph retrieval.
Một query-conditioned evidence graph nhẹ xây trên top-
𝐾
K candidates, tương thích trực tiếp với ColPali/ColQwen.
Một cơ chế graph reranking hiệu quả cho multi-page, multi-hop document QA.
Một protocol đánh giá cross-page retrieval gồm retrieval metrics và evidence-composition metrics.
Câu novelty gọn

We do not replace visual late-interaction retrieval; we make it compositional across pages.

Câu này rất sạch.

8. So với LILaC: khác ở đâu?
LILaC
layered component graph coarse/fine,
multimodal multihop retrieval tổng quát,
subgraph retrieval,
query decomposition + beam traversal,
không đóng khung như một drop-in ColPali reranker.
Đề tài của bạn
base retriever cố định là ColPali/ColQwen,
graph chỉ query-time trên top-k,
tập trung vào cross-page composition trong PDF dài,
không cần query decomposition bằng LLM,
nhắm đến đơn giản hơn, nhanh hơn, dễ gắn hơn.
Một câu phòng thủ với reviewer

Unlike LILaC’s corpus-level layered subgraph retrieval framework, our method targets a narrower but practically important setting: post-retrieval cross-page evidence aggregation for visual late-interaction document retrievers.

9. Datasets nên dùng
Main 1 — MP-DocVQA

Rất phù hợp vì benchmark này được thiết kế cho multi-page document VQA, nơi hệ thống cần tìm trang liên quan trước khi trả lời; paper gốc nhấn mạnh bài toán nhiều trang và đưa ra một cách scoring relevance giữa các trang.

Main 2 — DUDE

Dùng để kiểm tra độ robust trên tài liệu đa dạng hơn. Đây là benchmark phù hợp cho document understanding / retrieval thực tế hơn.

Optional 3 — SlideVQA hoặc benchmark PDF dài khác

Chỉ thêm nếu bạn cần chứng minh tính tổng quát.

Khuyến nghị thực dụng

Bản đầu:

main: MP-DocVQA
secondary: DUDE

Không nên mở quá nhiều benchmark ngay từ đầu.

10. Metrics
Retrieval metrics
Recall@1 / 5 / 10
MRR@10
nDCG@10
Cross-page evidence metrics

Đây là phần nên thêm để paper có bản sắc:

BothSupportHit@k

Nếu câu hỏi cần 2 trang hỗ trợ, coi là thành công khi cả hai trang đều nằm trong top-k.

Support-Page F1

Đo overlap giữa predicted support pages và gold support pages.

Evidence Coverage

Tỉ lệ evidence types hoặc support units được thu hồi.

Downstream QA
EM
F1
ANLS nếu benchmark dùng
11. Baselines
Core baselines
BM25 / text parser baseline
ColPali top-k
ColPali + naive concat top-k pages
ColPali + MLP reranker
X-PageRerank
Ablations
page-only graph
page+region graph
semantic edges only
semantic + structural edges
no graph, only top-k aggregation
static graph vs query-conditioned graph
Reviewer-critical baseline

Bạn bắt buộc phải có:

ColPali top-1
ColPali top-3
ColPali top-5
ColPali + naive concat

Nếu không, reviewer sẽ bảo graph thắng chỉ vì xem nhiều trang hơn.

12. Thiết kế thực nghiệm nhanh gọn
Phase 1 — bản chạy được trong thời gian ngắn
dùng ColPali làm page retriever
node = page
graph = page graph
edge = semantic similarity + page adjacency
model = 2-layer GAT
task = page reranking trên MP-DocVQA
Phase 2 — bản tốt hơn
thêm region nodes trên top pages
thêm structural edges
thêm support-page metrics
Phase 3 — bản paper mạnh hơn
thêm DUDE
thêm latency/efficiency analysis
thêm qualitative evidence chains
13. Figure và table plan
Figure 1

Pipeline:
query → ColPali retrieve top-k pages → build query-conditioned graph → rerank pages/evidence → answer

Figure 2

Ví dụ định tính:

chart ở page 5
explanation ở page 12
graph propagation nối hai evidence này
Figure 3

Performance theo query type:

single-hop
cross-page multi-hop
chart+text
table+paragraph
Table 1

Main retrieval results:

ColPali
ColPali + naive concat
ColPali + MLP reranker
X-PageRerank
Table 2

Ablation:

page graph
page+region graph
edge type variants
query-conditioned vs static
Table 3

Efficiency:

latency/query
memory
top-k sensitivity
14. Tuyên bố đóng góp kiểu mini paper
Abstract-style version

We study a failure mode of visual late-interaction document retrievers: while they localize relevant pages effectively, they do not explicitly compose evidence across multiple pages for multi-hop document questions. We propose X-PageRerank, a query-conditioned cross-page evidence reranker built on top of ColPali-style visual retrievers. Instead of constructing a corpus-level multimodal graph, our method builds a lightweight evidence graph only over top-k retrieved pages or regions, enabling efficient cross-page message passing and reranking. This design preserves the simplicity of visual retrieval while making it compositional across pages. We evaluate on multi-page document QA benchmarks and show improved retrieval and answer accuracy, especially for questions requiring distributed evidence across charts, tables, captions, and explanatory text. Supported by public ColPali code and multi-page VQA benchmarks, the approach is practical to implement and scales better than more general graph-retrieval frameworks.

15. Repo triển khai tối thiểu
Modules
retrieval/colpali_infer.py
retrieval/index_pages.py
graph/build_query_graph.py
graph/edges.py
models/page_gat_reranker.py
models/region_graph_reranker.py
train/train_reranker.py
eval/eval_retrieval.py
eval/eval_support_pages.py
data/mpdocvqa_loader.py
data/dude_loader.py
Thứ tự code
chạy ColPali inference
lưu top-k pages
viết graph builder page-level
train page GAT reranker
thêm region-level nodes
thêm ablations
16. Phán đoán cuối cùng
Tôi chốt hướng này là hợp lý vì:
không trùng trực diện với LILaC nữa,
khai thác đúng điểm yếu của ColPali,
triển khai nhanh nhờ code và benchmark có sẵn,
paper story sạch: từ page retrieval sang cross-page evidence composition. ColPali có code và paper công khai; MP-DocVQA đã là benchmark nhiều trang rõ ràng; LILaC xác nhận đây là một hướng retrieval nhiều bước đáng nghiên cứu nhưng cũng cho thấy bạn cần tránh formulation graph tổng quát của họ.
Tên đề tài tôi khuyên dùng nhất

Cross-Page Evidence Reranking for Visual Late-Interaction Retrieval in Long PDF Documents

Nếu muốn, lượt tiếp theo tôi sẽ viết luôn cho bạn bản mini paper hoàn chỉnh 1.5–2 trang với các mục:

Abstract
Introduction
Related Work
Method
Experiment Plan
Expected Contributions
theo đúng văn phong paper.
Viết cho tôi một bản paper đầy đủ chi tiết. 
X-page Rerank Paper Draft
Cross-Page Evidence Reranking for Visual Late-Interaction Retrieval in Long PDF Documents
Abstract

Visual late-interaction retrievers such as ColPali have recently demonstrated strong performance for page-level retrieval over document images. However, they remain limited when answering queries that require compositional reasoning across multiple pages, such as linking a figure on one page with explanatory text on another. In this work, we propose X-PageRerank, a query-conditioned cross-page evidence reranking framework that augments visual late-interaction retrievers with lightweight graph-based reasoning. Instead of constructing a corpus-level multimodal graph, our method builds a local evidence graph over top-k retrieved pages and regions, enabling efficient message passing across pages. This design preserves the scalability and simplicity of existing retrievers while enabling cross-page evidence composition. Experiments on multi-page document QA benchmarks demonstrate improvements in retrieval and answer accuracy, especially for multi-hop queries. Our approach provides a practical and modular solution for compositional document retrieval.

1. Introduction

Modern document understanding systems increasingly rely on neural retrieval to identify relevant evidence before downstream reasoning. Visual late-interaction retrievers, such as ColPali, have shown strong performance by directly operating on document images without requiring full OCR pipelines. These models encode page images into multi-vector representations and perform fine-grained matching with queries via late interaction.

Despite these advances, current systems exhibit a key limitation: they treat each page independently. This assumption breaks down for queries that require cross-page reasoning, where relevant evidence is distributed across multiple pages. For example, answering a question may require combining a chart on one page with a textual explanation located elsewhere in the document.

We identify this limitation as a lack of compositionality in visual retrieval. While retrievers can localize relevant pages, they lack mechanisms to aggregate and propagate information across pages.

To address this gap, we propose X-PageRerank, a lightweight, query-conditioned cross-page evidence reranking framework. Our key idea is simple: rather than modifying the retrieval backbone, we construct a local evidence graph over top-k retrieved pages and perform message passing to enable cross-page reasoning.

Our contributions are:

A new formulation of cross-page retrieval as a post-retrieval evidence composition problem.
A query-conditioned evidence graph built over top-k retrieved pages and regions.
A lightweight graph-based reranking module that improves multi-hop retrieval.
A comprehensive evaluation protocol for cross-page retrieval on document QA benchmarks.
2. Related Work
Neural Retrieval

Dense and late-interaction retrieval models such as ColBERT and ColPali have significantly improved retrieval performance by enabling token-level matching. ColPali extends this paradigm to document images using vision-language encoders.

Document Understanding

Models such as LayoutLMv3 and DocLLM incorporate layout and visual structure for document understanding tasks. However, these approaches focus on extraction and classification rather than retrieval.

Multimodal and Multi-hop Retrieval

Recent work explores graph-based retrieval and multi-hop reasoning across heterogeneous data sources. Notably, LILaC introduces a layered component graph and subgraph retrieval framework. In contrast, our work focuses on efficient post-retrieval reasoning over visual document pages, without requiring corpus-wide graph construction.

3. Problem Formulation

Given a document composed of pages:

D = {p1, p2, ..., pn}

and a query q, a visual retriever R returns top-k pages:

P_k(q) = {p_i1, ..., p_iK}

Our goal is to identify a set of evidence units E(q) across these pages that collectively support answering q.

Unlike traditional retrieval, where relevance is evaluated per page, we consider compositional relevance across pages.

4. Method
4.1 Overview

Our method consists of four stages:

Visual retrieval (ColPali)
Evidence extraction
Query-conditioned graph construction
Graph-based reranking
4.2 Evidence Units

We define nodes in the graph as:

Page nodes
Region nodes (optional)

Each node is represented as:

v_i = (h_i, type_i, page_i)

where h_i is the embedding.

4.3 Graph Construction

We construct a query-conditioned graph G_q = (V_q, E_q) over top-k pages.

Edges include:

Semantic similarity edges
Page adjacency edges
Structural edges (figure-caption, etc.)
4.4 Graph Reranking

We apply L-layer message passing:

h_i^(l+1) = sigma( sum_j alpha_ij W h_j^(l) )

Final score:

s(q, p_i) = s0(q, p_i) + lambda * g(q, h_i^L)

5. Experiments
5.1 Datasets
MP-DocVQA
DUDE
5.2 Metrics
Recall@k
MRR@10
nDCG@10
Support-page F1
5.3 Baselines
BM25
ColPali
ColPali + concat
MLP reranker
6. Results

We expect:

Improved Recall@k for multi-page queries
Higher support-page accuracy
Gains in downstream QA
7. Ablation Studies
Graph vs no graph
Page vs region nodes
Edge types
8. Discussion

Our method demonstrates that cross-page reasoning can be achieved without expensive corpus-level graph construction.

9. Conclusion

We propose X-PageRerank, a practical and efficient method for cross-page visual retrieval. By introducing lightweight graph-based reasoning at query time, we enable compositional retrieval across document pages.

Appendix (Planned)
Implementation details
Hyperparameters
Additional examples
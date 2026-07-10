# Ý nghĩa các dataset trong chương trình

Chương trình sử dụng 3 bộ dữ liệu để minh họa 3 thuật toán gom nhóm khác nhau:

- Seeds dùng cho KMeans.
- User Knowledge Modeling dùng cho Hierarchical clustering.
- Spiral dùng cho DBSCAN.

Các cột nhãn như `variety_name`, `UNS`, `true_label` chỉ dùng để đối chiếu kết quả sau khi gom nhóm. Những cột này không được dùng làm đầu vào cho thuật toán.

## 1. Seeds Dataset

Nguồn: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/236/seeds

Seeds là bộ dữ liệu về hạt lúa mì. Mỗi dòng dữ liệu tương ứng với một hạt lúa mì, được mô tả bằng các đặc trưng hình học lấy từ ảnh X-ray.

Dataset này có 210 mẫu, thuộc 3 giống lúa mì khác nhau:

- Kama
- Rosa
- Canadian

Mỗi giống có 70 mẫu.

Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `area` | Diện tích hạt |
| `perimeter` | Chu vi hạt |
| `compactness` | Độ đặc/chặt của hình dạng hạt |
| `kernel_length` | Chiều dài hạt |
| `kernel_width` | Chiều rộng hạt |
| `asymmetry` | Độ bất đối xứng |
| `groove_length` | Chiều dài rãnh hạt |
| `variety_name` | Giống lúa mì thật, chỉ dùng để đối chiếu |

Trong chương trình, Seeds được dùng để minh họa thuật toán KMeans.

Lý do phù hợp:

- Dữ liệu hoàn toàn là số.
- Có 3 nhóm tự nhiên tương ứng với 3 giống lúa mì.
- Các cụm tương đối gọn, phù hợp với ý tưởng mỗi cụm có một tâm centroid.
- KMeans cần chọn trước số cụm `k`, ở đây có thể đặt `k=3`.

Nói dễ hiểu, chương trình lấy các số đo hình dạng của hạt, sau đó xem KMeans có tự gom được các hạt cùng giống vào cùng cụm hay không. Cột `variety_name` không được dùng để gom nhóm, chỉ dùng sau đó để kiểm tra kết quả có hợp lý không.

## 2. User Knowledge Modeling Dataset

Nguồn: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling

User Knowledge Modeling là bộ dữ liệu về mức độ hiểu biết của sinh viên hoặc người học đối với chủ đề Electrical DC Machines.

Dataset này có 403 mẫu và 5 thuộc tính đầu vào. Mỗi dòng là một người học, được mô tả bằng các chỉ số liên quan đến thời gian học, mức độ lặp lại và kết quả kiểm tra.

Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `STG` | Mức độ thời gian học tài liệu mục tiêu |
| `SCG` | Mức độ lặp lại hoặc số lần học tài liệu mục tiêu |
| `STR` | Mức độ thời gian học các tài liệu liên quan |
| `LPR` | Kết quả kiểm tra trên các tài liệu liên quan |
| `PEG` | Kết quả kiểm tra trên tài liệu mục tiêu |
| `UNS` | Mức độ kiến thức thật của người học, ví dụ Very Low, Low, Middle, High |

Trong chương trình, dataset này được dùng để minh họa Hierarchical clustering.

Lý do phù hợp:

- Mỗi người học có nhiều đặc trưng hành vi học tập.
- Có thể xem từng người học ban đầu là một cụm riêng.
- Hierarchical clustering gộp dần những người có hồ sơ học tập giống nhau.
- Dendrogram giúp trình bày trực quan quá trình gộp cụm.

Nói dễ hiểu, chương trình dùng các chỉ số học tập như thời gian học, số lần lặp lại và điểm kiểm tra để gom những sinh viên có kiểu học hoặc tình trạng kiến thức giống nhau. Cột `UNS` là nhãn thật về mức kiến thức, chỉ dùng để đối chiếu sau khi gom nhóm.

Trong code của chương trình, dataset đầy đủ có 403 dòng, nhưng phần demo hierarchical clustering lấy mẫu 90 dòng để dendrogram dễ nhìn và tránh quá nặng.

## 3. Spiral Dataset

Nguồn: UEF/SIPU clustering benchmark  
Link: https://cs.joensuu.fi/sipu/datasets/spiral.txt

Spiral là dataset tổng hợp, không phải dữ liệu đời thực như Seeds hay User Knowledge. Nó được tạo ra để kiểm tra thuật toán phân cụm trên dữ liệu có hình dạng đặc biệt.

Theo trang benchmark UEF/SIPU, Spiral thuộc nhóm shape datasets, có:

```text
N = 312 điểm
k = 3 cụm
D = 2 chiều
```

Mỗi dòng thường gồm:

| Cột | Ý nghĩa |
|---|---|
| `x` | Tọa độ x của điểm |
| `y` | Tọa độ y của điểm |
| `true_label` | Nhãn cụm thật |

Điểm đặc biệt của Spiral là các cụm có dạng xoắn ốc, không phải dạng tròn hoặc gọn quanh tâm.

Trong chương trình, Spiral được dùng để minh họa DBSCAN.

Lý do phù hợp:

- DBSCAN gom cụm dựa trên mật độ và sự liên thông giữa các điểm.
- DBSCAN không yêu cầu cụm có dạng tròn.
- DBSCAN không cần biết trước số cụm.
- DBSCAN xử lý tốt các cụm có hình dạng cong, xoắn, không lồi.

Dataset này cũng được dùng để so sánh với KMeans. KMeans thường chia điểm theo khoảng cách đến centroid, nên với cụm xoắn, nó dễ chia sai hình dạng cụm.

Nói dễ hiểu, Spiral cho thấy điểm mạnh của DBSCAN. Nếu nhìn bằng mắt, ta thấy 3 nhánh xoắn rõ ràng. DBSCAN có thể lần theo mật độ điểm trên từng nhánh, còn KMeans có xu hướng cắt không gian thành các vùng quanh tâm, nên không giữ được hình dạng xoắn.

## So sánh nhanh

| Dataset | Dữ liệu nói về gì? | Dùng cho thuật toán | Vì sao phù hợp |
|---|---|---|---|
| Seeds | Hình dạng hạt lúa mì | KMeans | Có 3 nhóm tương đối gọn, phù hợp centroid |
| User Knowledge | Hồ sơ học tập của người học | Hierarchical clustering | Dễ biểu diễn quá trình gộp cụm bằng dendrogram |
| Spiral | Điểm 2D dạng xoắn tổng hợp | DBSCAN | Cụm không lồi, phù hợp gom nhóm theo mật độ |

## Kết luận

Seeds là dữ liệu sinh học có các cụm tương đối gọn, phù hợp để minh họa KMeans. User Knowledge là dữ liệu hành vi học tập, phù hợp để xem quan hệ phân cấp và quá trình gộp cụm của Hierarchical clustering. Spiral là dữ liệu benchmark tổng hợp, dùng để chứng minh DBSCAN xử lý tốt các cụm có hình dạng phức tạp.

# Ghi chú dữ liệu và xử lý cho Chủ đề 2

## Phân chia công việc

- Phần dữ liệu và phương pháp: chọn bộ dữ liệu, tiền xử lý, chọn tham số mặc định, xuất kết quả trực quan hóa.
- Phần web app: xây dựng giao diện upload/chọn giải thuật/hiển thị kết quả và triển khai.
- Phần báo cáo: cơ sở lý thuyết, ảnh minh họa, nhận xét kết quả và trích dẫn nguồn dữ liệu.

## Bộ dữ liệu được chọn

| Minh họa | Bộ dữ liệu | Số dòng | Thuộc tính dùng để gom nhóm | Nhãn chỉ dùng để đối chiếu | Nguồn |
|---|---:|---:|---|---|---|
| KMeans | Seeds | 210 | 7 thuộc tính số đo hạt lúa mì | `variety_name` | UCI Seeds, DOI 10.24432/C5H30K |
| Hierarchical clustering | User Knowledge Modeling | 403 dòng đầy đủ / 90 dòng mẫu | 5 chỉ số hành vi học tập | `UNS` | UCI User Knowledge Modeling, DOI 10.24432/C5231X |
| DBSCAN | Spiral | 312 | 2 tọa độ | `true_label` | UEF/SIPU clustering benchmark |

## Tiền xử lý chung

1. Chỉ chọn các cột số làm thuộc tính đầu vào.
2. Chuyển giá trị không hợp lệ thành missing value.
3. Điền missing value bằng trung vị.
4. Chuẩn hóa dữ liệu bằng `StandardScaler`.
5. Tạo `plot_x` và `plot_y` để vẽ 2D. Nếu dữ liệu có hơn 2 thuộc tính thì dùng PCA để chiếu xuống 2 chiều.

## Tham số mặc định

- Seeds + KMeans: `k=3`, random seed 42, lặp đến khi centroid không còn dịch chuyển. Bảng bước chạy và centroid nằm trong không gian đã chuẩn hóa, không phải đơn vị đo gốc.
- User Knowledge + hierarchical clustering: Ward linkage, cắt cây thành `k=4` cụm. App có thể dùng file đầy đủ 403 dòng, nhưng dendrogram và bảng merge dùng mẫu 90 dòng để dễ đọc.
- Spiral + DBSCAN: `eps=0.30`, `min_samples=5` sau khi chuẩn hóa.

Với tham số mặc định, Spiral không có điểm nhiễu. Bộ này được dùng để minh họa cụm không lồi theo mật độ. Nếu cần minh họa noise/outlier, nên thêm một phiên bản dữ liệu có nhiễu thay vì đổi bộ Spiral sạch này.

## File kết quả để trực quan hóa

Contract chính: `outputs/topic2/handoff/topic2_visualization_contract.json`.
Một bản sao nằm ở `docs/topic2_visualization_contract.json` để web app có thể đọc cấu trúc file.

Mỗi bộ minh họa chính có:

- `*_points.csv`: dữ liệu gốc + `cluster`, `plot_x`, `plot_y`.
- `*_cluster_summary.csv`: kích thước từng cụm và nhãn thật chiếm nhiều nhất nếu có.
- `*_metrics.json`: tham số và chỉ số đánh giá.
- ảnh scatter và ảnh kích thước cụm.

Artifact bổ sung:

- `seeds_kmeans_steps.csv`, `seeds_kmeans_centroids.csv`: các bước chạy KMeans và lịch sử centroid.
- `user_knowledge_hierarchical_merges.csv`, `user_knowledge_distance_matrix_preview.csv`, dendrogram PNG: dùng để giải thích hierarchical clustering.
- `spiral_dbscan_k_distance.csv`, k-distance PNG: dùng để giải thích cách chọn `eps` cho DBSCAN.
- `spiral_kmeans_comparison_points.csv`, PNG so sánh: cho thấy KMeans không phù hợp với cụm dạng xoắn.

## Nhận xét ngắn

KMeans rõ nhất trên Seeds vì dữ liệu có các cụm số tương đối gọn. Hierarchical clustering rõ nhất trên User Knowledge vì dendrogram cho thấy các hồ sơ học tập được gộp từng bước. DBSCAN rõ nhất trên Spiral vì cụm được xác định theo mật độ và hình dạng liên thông, không theo khoảng cách đến centroid.

Với Spiral, silhouette không phải chỉ số chính. Silhouette ưu tiên cụm lồi/gọn, nên có thể thấp dù DBSCAN khôi phục đúng nhãn spiral. Khi nhận xét Spiral, nên dùng scatter plot và adjusted Rand index (ARI).

## Nguồn và trích dẫn

- Seeds: Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., & Lukasik, S. (2010). Seeds [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5H30K
- User Knowledge Modeling: Kahraman, H., Colak, I., & Sagiroglu, S. (2009). User Knowledge Modeling [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5231X
- Spiral: University of Eastern Finland / SIPU clustering benchmark datasets. Trang benchmark ghi bộ dữ liệu shape có cột thứ ba là nhãn; Spiral có N=312, k=3, D=2.

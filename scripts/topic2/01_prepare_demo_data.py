#!/usr/bin/env python3
"""Prepare Topic 2 clustering demo datasets and visualization artifacts.

Outputs:
  * data/topic2/examples/*.csv
  * outputs/topic2/results/*.csv, *.json
  * outputs/topic2/plots/*.png
  * outputs/topic2/handoff/topic2_visualization_contract.json
  * docs/topic2_dataset_notes.md

Run from repo root:
    python scripts/topic2/01_prepare_demo_data.py
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from topic2_clustering.analysis import (  # noqa: E402
    prepare_numeric_frame,
    run_dbscan,
    run_hierarchical,
    run_kmeans,
)
from topic2_clustering.paths import DOCS_DIR, EXAMPLES_DIR, HANDOFF_DIR, PLOTS_DIR, RESULTS_DIR, ensure_dirs  # noqa: E402
from topic2_clustering.plots import save_bar, save_cluster_scatter, save_dendrogram, save_k_distance  # noqa: E402


def fetch_zip_member(url: str, member_hint: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        match = next((n for n in names if member_hint.lower() in n.lower()), None)
        if match is None:
            raise RuntimeError(f"Could not find {member_hint!r} in {url}; members={names}")
        return zf.read(match)


def load_seeds() -> pd.DataFrame:
    cached = EXAMPLES_DIR / "seeds_kmeans.csv"
    if cached.exists():
        return pd.read_csv(cached)
    raw = fetch_zip_member("https://archive.ics.uci.edu/static/public/236/seeds.zip", "seeds_dataset")
    cols = ["area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry", "groove_length", "variety"]
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, names=cols)
    df["variety_name"] = df["variety"].map({1: "Kama", 2: "Rosa", 3: "Canadian"})
    return df


def load_user_knowledge() -> pd.DataFrame:
    cached = EXAMPLES_DIR / "user_knowledge_hierarchical.csv"
    if cached.exists():
        df = pd.read_csv(cached)
        if "UNS" in df.columns:
            df["UNS"] = df["UNS"].astype(str).str.strip().replace({"Very Low": "very_low"})
        return df
    raw = fetch_zip_member("https://archive.ics.uci.edu/static/public/257/user+knowledge+modeling.zip", ".xls")
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    frames = []
    for split, df in sheets.items():
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        needed = ["STG", "SCG", "STR", "LPR", "PEG", "UNS"]
        if all(c in df.columns for c in needed):
            frames.append(df[needed].assign(source_sheet=split))
    if not frames:
        raise RuntimeError("User Knowledge workbook did not contain expected columns")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["STG", "SCG", "STR", "LPR", "PEG"])
    out["UNS"] = out["UNS"].astype(str).str.strip().replace({"Very Low": "very_low"})
    out["student_id"] = [f"S{i:03d}" for i in range(1, len(out) + 1)]
    return out[["student_id", "STG", "SCG", "STR", "LPR", "PEG", "UNS", "source_sheet"]]


def load_spiral() -> pd.DataFrame:
    cached = EXAMPLES_DIR / "spiral_dbscan.csv"
    if cached.exists():
        return pd.read_csv(cached)
    url = "https://cs.joensuu.fi/sipu/datasets/spiral.txt"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, comment="#")
    if df.shape[1] < 3:
        raise RuntimeError("Expected Spiral dataset to have x, y, label columns")
    df = df.iloc[:, :3]
    df.columns = ["x", "y", "true_label"]
    return df


def metric_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records).sort_values(["dataset", "algorithm"])


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def save_artifacts(prefix: str, result: dict[str, object], *, plot_title: str) -> dict:
    points = result["points"]
    summary = result["summary"]
    points_path = RESULTS_DIR / f"{prefix}_points.csv"
    summary_path = RESULTS_DIR / f"{prefix}_cluster_summary.csv"
    metrics_path = RESULTS_DIR / f"{prefix}_metrics.json"
    scatter_path = PLOTS_DIR / f"{prefix}_scatter.png"
    bar_path = PLOTS_DIR / f"{prefix}_cluster_sizes.png"

    points.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_json(metrics_path, result["metrics"])
    save_cluster_scatter(points, scatter_path, title=plot_title)
    save_bar(summary, bar_path, title=f"{plot_title}: cluster sizes")

    return {
        "points_csv": str(points_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "metrics_json": str(metrics_path.relative_to(ROOT)),
        "scatter_png": str(scatter_path.relative_to(ROOT)),
        "cluster_sizes_png": str(bar_path.relative_to(ROOT)),
    }


def main() -> None:
    ensure_dirs()

    seeds = load_seeds()
    knowledge = load_user_knowledge()
    spiral = load_spiral()

    seeds_path = EXAMPLES_DIR / "seeds_kmeans.csv"
    knowledge_path = EXAMPLES_DIR / "user_knowledge_hierarchical.csv"
    spiral_path = EXAMPLES_DIR / "spiral_dbscan.csv"
    seeds.to_csv(seeds_path, index=False)
    knowledge.to_csv(knowledge_path, index=False)
    spiral.to_csv(spiral_path, index=False)

    metrics = []
    handoff: dict[str, object] = {
        "schema_version": 1,
        "generated_for": "Handoff trực quan hóa cho Chủ đề 2",
        "coordinate_columns": {
            "plot_x": "Tọa độ x để vẽ 2D. Nếu dữ liệu có đúng 2 thuộc tính thì đây là x đã chuẩn hóa; nếu nhiều hơn 2 thuộc tính thì là thành phần PCA-1.",
            "plot_y": "Tọa độ y để vẽ 2D. Nếu dữ liệu có đúng 2 thuộc tính thì đây là y đã chuẩn hóa; nếu nhiều hơn 2 thuộc tính thì là thành phần PCA-2.",
        },
        "common_point_columns": ["cluster", "plot_x", "plot_y"],
        "datasets": {},
    }

    seeds_features = ["area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry", "groove_length"]
    seeds_prepared = prepare_numeric_frame(seeds, seeds_features, label_column="variety_name")
    seeds_result = run_kmeans(seeds_prepared, k=3)
    seeds_extra = save_artifacts("seeds_kmeans", seeds_result, plot_title="Seeds dataset — KMeans k=3")
    seeds_result["trace"].to_csv(RESULTS_DIR / "seeds_kmeans_steps.csv", index=False)
    seeds_result["centroids"].to_csv(RESULTS_DIR / "seeds_kmeans_centroids.csv", index=False)
    metrics.append({"dataset": "seeds", **seeds_result["metrics"]})
    handoff["datasets"]["seeds_kmeans"] = {
        "purpose": "Minh họa KMeans: cụm tương đối gọn và phù hợp với centroid",
        "source_csv": str(seeds_path.relative_to(ROOT)),
        "generated_from_csv": str(seeds_path.relative_to(ROOT)),
        "row_count": int(len(seeds)),
        "feature_columns": seeds_features,
        "label_column_for_evaluation_only": "variety_name",
        "default_algorithm": "kmeans",
        "default_parameters": {"k": 3, "scale": "StandardScaler", "impute": "median"},
        "required_point_columns": [*seeds.columns.tolist(), "cluster", "plot_x", "plot_y"],
        "centroid_note": "Centroid trong seeds_kmeans_centroids.csv và bảng bước chạy nằm trong không gian đã chuẩn hóa (đầu ra StandardScaler), không phải đơn vị đo gốc.",
        "extra_artifact_schemas": {
            "steps_csv": {
                "columns": ["step", "action", "inertia", "changed_assignments", "centroid_shift"],
                "note": "changed_assignments trống ở bước 1 vì chưa có nhãn trước đó; centroid_shift trống ở các dòng gán điểm vào cụm.",
            },
            "centroids_csv": {
                "columns": ["step", "cluster"] + seeds_features,
                "note": "Tọa độ centroid nằm trong không gian đã chuẩn hóa. Mỗi dòng là một centroid ở một bước.",
            },
        },
        "artifacts": {**seeds_extra, "steps_csv": "outputs/topic2/results/seeds_kmeans_steps.csv", "centroids_csv": "outputs/topic2/results/seeds_kmeans_centroids.csv"},
    }

    knowledge_features = ["STG", "SCG", "STR", "LPR", "PEG"]
    knowledge_sample = knowledge.sample(n=min(90, len(knowledge)), random_state=42).sort_values("student_id").reset_index(drop=True)
    knowledge_sample.to_csv(EXAMPLES_DIR / "user_knowledge_hierarchical_sample90.csv", index=False)
    knowledge_prepared = prepare_numeric_frame(knowledge_sample, knowledge_features, label_column="UNS")
    knowledge_result = run_hierarchical(
        knowledge_prepared,
        k=4,
        method="ward",
        row_labels=knowledge_sample["student_id"].astype(str).tolist(),
    )
    knowledge_extra = save_artifacts("user_knowledge_hierarchical", knowledge_result, plot_title="User Knowledge — hierarchical clustering")
    knowledge_result["merges"].to_csv(RESULTS_DIR / "user_knowledge_hierarchical_merges.csv", index=False)
    dist_preview = knowledge_result["distance_matrix"].iloc[:25, :25].copy()
    dist_preview.index.name = "student_id"
    dist_preview.to_csv(RESULTS_DIR / "user_knowledge_distance_matrix_preview.csv")
    save_dendrogram(
        knowledge_result["linkage_matrix"],
        knowledge_sample["student_id"].tolist(),
        PLOTS_DIR / "user_knowledge_hierarchical_dendrogram.png",
        title="User Knowledge dendrogram (sample of 90)",
    )
    metrics.append({"dataset": "user_knowledge_sample90", **knowledge_result["metrics"]})
    handoff["datasets"]["user_knowledge_hierarchical"] = {
        "purpose": "Minh họa hierarchical clustering: dễ xem từng bước gộp hồ sơ học tập",
        "source_csv": str(knowledge_path.relative_to(ROOT)),
        "generated_from_csv": "data/topic2/examples/user_knowledge_hierarchical_sample90.csv",
        "row_count": int(len(knowledge_sample)),
        "full_source_row_count": int(len(knowledge)),
        "demo_sample_csv": "data/topic2/examples/user_knowledge_hierarchical_sample90.csv",
        "feature_columns": knowledge_features,
        "label_column_for_evaluation_only": "UNS",
        "default_algorithm": "hierarchical",
        "default_parameters": {"k": 4, "linkage": "ward", "scale": "StandardScaler", "impute": "median"},
        "required_point_columns": [*knowledge_sample.columns.tolist(), "cluster", "plot_x", "plot_y"],
        "extra_artifact_schemas": {
            "merges_csv": {
                "columns": ["step", "left", "right", "distance", "new_cluster_size", "members_preview"],
                "note": "left/right hiển thị tối đa 8 student_id đầu tiên mỗi phía. distance là khoảng cách gộp theo linkage. members_preview hiển thị tối đa 12 phần tử đầu của cụm mới.",
            },
            "distance_preview_csv": {
                "columns": ["student_id"] + knowledge_sample["student_id"].iloc[:25].astype(str).tolist(),
                "note": "25x25 symmetric standardized-Euclidean distance preview. index column is student_id.",
            },
        },
        "artifacts": {**knowledge_extra, "merges_csv": "outputs/topic2/results/user_knowledge_hierarchical_merges.csv", "distance_preview_csv": "outputs/topic2/results/user_knowledge_distance_matrix_preview.csv", "dendrogram_png": "outputs/topic2/plots/user_knowledge_hierarchical_dendrogram.png"},
    }

    spiral_features = ["x", "y"]
    spiral_prepared = prepare_numeric_frame(spiral, spiral_features, label_column="true_label")
    spiral_result = run_dbscan(spiral_prepared, eps=0.30, min_samples=5)
    spiral_extra = save_artifacts("spiral_dbscan", spiral_result, plot_title="Spiral dataset — DBSCAN")
    spiral_result["k_distance"].to_csv(RESULTS_DIR / "spiral_dbscan_k_distance.csv", index=False)
    save_k_distance(spiral_result["k_distance"], PLOTS_DIR / "spiral_dbscan_k_distance.png", title="Spiral k-distance curve (k=5)")
    kmeans_on_spiral = run_kmeans(spiral_prepared, k=3)
    kmeans_on_spiral["points"].to_csv(RESULTS_DIR / "spiral_kmeans_comparison_points.csv", index=False)
    save_cluster_scatter(kmeans_on_spiral["points"], PLOTS_DIR / "spiral_kmeans_comparison.png", title="Spiral dataset — KMeans comparison")
    metrics.append({"dataset": "spiral", **spiral_result["metrics"]})
    metrics.append({"dataset": "spiral_comparison", **kmeans_on_spiral["metrics"]})
    handoff["datasets"]["spiral_dbscan"] = {
        "purpose": "Minh họa DBSCAN: cụm không lồi và liên thông theo mật độ",
        "source_csv": str(spiral_path.relative_to(ROOT)),
        "generated_from_csv": str(spiral_path.relative_to(ROOT)),
        "row_count": int(len(spiral)),
        "feature_columns": spiral_features,
        "label_column_for_evaluation_only": "true_label",
        "default_algorithm": "dbscan",
        "default_parameters": {"eps": 0.30, "min_samples": 5, "scale": "StandardScaler", "impute": "median"},
        "required_point_columns": [*spiral.columns.tolist(), "cluster", "plot_x", "plot_y", "point_type"],
        "metric_note": "Silhouette không phải chỉ số chính cho cụm dạng xoắn vì chỉ số này ưu tiên cụm lồi/gọn. Với demo này, nên dùng ARI với true_label và scatter plot.",
        "extra_artifact_schemas": {
            "k_distance_csv": {
                "columns": ["rank", "k_distance"],
                "note": "Các điểm được sắp theo k-distance tăng dần (khoảng cách đến láng giềng thứ 5). Vùng khuỷu là gợi ý tốt để chọn eps.",
            },
            "kmeans_comparison_points_csv": {
                "columns": ["x", "y", "true_label", "cluster", "plot_x", "plot_y"],
                "note": "KMeans (k=3) chạy trên cùng dữ liệu Spiral. So sánh cluster với true_label để thấy KMeans không phù hợp với cụm không lồi.",
            },
        },
        "artifacts": {**spiral_extra, "k_distance_csv": "outputs/topic2/results/spiral_dbscan_k_distance.csv", "k_distance_png": "outputs/topic2/plots/spiral_dbscan_k_distance.png", "kmeans_comparison_points_csv": "outputs/topic2/results/spiral_kmeans_comparison_points.csv", "kmeans_comparison_png": "outputs/topic2/plots/spiral_kmeans_comparison.png"},
    }

    all_metrics = metric_frame(metrics)
    all_metrics.to_csv(RESULTS_DIR / "topic2_algorithm_metrics.csv", index=False)
    handoff["metrics_csv"] = "outputs/topic2/results/topic2_algorithm_metrics.csv"
    handoff["notes_md"] = "docs/topic2_dataset_notes.md"
    write_json(HANDOFF_DIR / "topic2_visualization_contract.json", handoff)
    write_json(DOCS_DIR / "topic2_visualization_contract.json", handoff)

    notes = f"""# Ghi chú dữ liệu và xử lý cho Chủ đề 2

## Phân chia công việc

- Phần dữ liệu và phương pháp: chọn bộ dữ liệu, tiền xử lý, chọn tham số mặc định, xuất kết quả trực quan hóa.
- Phần web app: xây dựng giao diện upload/chọn giải thuật/hiển thị kết quả và triển khai.
- Phần báo cáo: cơ sở lý thuyết, ảnh minh họa, nhận xét kết quả và trích dẫn nguồn dữ liệu.

## Bộ dữ liệu được chọn

| Minh họa | Bộ dữ liệu | Số dòng | Thuộc tính dùng để gom nhóm | Nhãn chỉ dùng để đối chiếu | Nguồn |
|---|---:|---:|---|---|---|
| KMeans | Seeds | {len(seeds)} | {len(seeds_features)} thuộc tính số đo hạt lúa mì | `variety_name` | UCI Seeds, DOI 10.24432/C5H30K |
| Hierarchical clustering | User Knowledge Modeling | {len(knowledge)} dòng đầy đủ / {len(knowledge_sample)} dòng mẫu | {len(knowledge_features)} chỉ số hành vi học tập | `UNS` | UCI User Knowledge Modeling, DOI 10.24432/C5231X |
| DBSCAN | Spiral | {len(spiral)} | 2 tọa độ | `true_label` | UEF/SIPU clustering benchmark |

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
"""
    (DOCS_DIR / "topic2_dataset_notes.md").write_text(notes)

    print("Prepared Topic 2 demo data")
    print(all_metrics.to_string(index=False))
    print("Handoff:", HANDOFF_DIR / "topic2_visualization_contract.json")
    print("Tracked contract copy:", DOCS_DIR / "topic2_visualization_contract.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster

from topic2_clustering.analysis import (
    prepare_numeric_frame,
    preprocessing_report,
    run_dbscan,
    run_hierarchical,
    run_kmeans,
    k_distance_table,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "topic2_visualization_contract.json"


st.set_page_config(
    page_title="CT312 Chủ đề 2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_json(rel_path: str) -> dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text())


@st.cache_data
def load_csv(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel_path)


def local_path(rel_path: str) -> Path:
    return ROOT / rel_path


def require_file(rel_path: str) -> bool:
    if local_path(rel_path).exists():
        return True
    st.error("Thiếu file kết quả. Chạy lại hai lệnh tạo dữ liệu và kiểm tra handoff.")
    st.code("python scripts/topic2/01_prepare_demo_data.py\npython scripts/topic2/02_validate_handoff.py")
    st.code(rel_path)
    return False


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, fmt(value), help=help_text)


def scatter(points: pd.DataFrame, *, title: str, color_col: str = "cluster", hover_cols: list[str] | None = None) -> None:
    hover_cols = hover_cols or []
    fig = px.scatter(
        points,
        x="plot_x",
        y="plot_y",
        color=points[color_col].astype(str),
        hover_data=[c for c in hover_cols if c in points.columns],
        title=title,
        opacity=0.82,
        height=520,
    )
    fig.update_traces(marker={"size": 8, "line": {"width": 0.4, "color": "#ffffff"}})
    fig.update_layout(legend_title_text=color_col, margin={"l": 12, "r": 12, "t": 56, "b": 12})
    st.plotly_chart(fig, use_container_width=True)


def upload_data_errors(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Return messages for uploaded data that cannot be clustered safely."""
    errors: list[str] = []
    if frame.empty:
        return ["File CSV không có dòng dữ liệu. Hãy tải lên file có ít nhất 2 dòng dữ liệu."]
    if len(frame) < 2:
        errors.append("Dữ liệu chỉ có 1 dòng. Cả ba thuật toán cần ít nhất 2 dòng để tính khoảng cách.")
    if not feature_columns:
        errors.append("Chưa chọn cột số để gom nhóm. Hãy chọn ít nhất một cột đặc trưng dạng số.")
        return errors

    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    infinite_columns = [
        column
        for column in feature_columns
        if np.isinf(numeric[column].dropna().to_numpy()).any()
    ]
    if infinite_columns:
        errors.append(
            "Các cột " + ", ".join(f"`{column}`" for column in infinite_columns)
            + " chứa giá trị vô cực (`inf` hoặc `-inf`). Hãy thay các giá trị này bằng số hữu hạn hoặc để trống."
        )

    empty_columns = [
        column
        for column in feature_columns
        if numeric[column].replace([np.inf, -np.inf], np.nan).notna().sum() == 0
    ]
    if empty_columns:
        errors.append(
            "Các cột " + ", ".join(f"`{column}`" for column in empty_columns)
            + " không có giá trị số hợp lệ. Hãy chọn cột khác hoặc bổ sung dữ liệu số."
        )
    return errors

def show_preprocessing(report):

    st.subheader("Các bước tiền xử lý")

    st.markdown("""
    1. Chuyển dữ liệu sang kiểu số

    2. Kiểm tra giá trị thiếu

    3. Điền Missing bằng Median

    4. Chuẩn hóa bằng StandardScaler
    
    5. PCA (nếu số chiều >2)
    """)

    st.write("### Giá trị thiếu")
    st.dataframe(report.missing_summary)

    st.write("### Thống kê dữ liệu")
    st.dataframe(report.numeric_summary)

    st.write("### StandardScaler")

    st.dataframe(report.scaler_summary)

    if report.use_pca:
        st.success("PCA sẽ được dùng để giảm dữ liệu xuống 2 chiều để trực quan hóa.")
    else:
        st.info("Không cần PCA.")

def show_upload_errors(errors: list[str]) -> None:
    st.error("Không thể chạy thuật toán với dữ liệu hiện tại.")
    for message in errors:
        st.warning(message)


def bar(summary: pd.DataFrame, *, title: str) -> None:
    fig = px.bar(summary, x="cluster_name", y="n_rows", text="n_rows", title=title, height=360)
    fig.update_layout(xaxis_title="cụm", yaxis_title="số dòng", margin={"l": 12, "r": 12, "t": 56, "b": 12})
    st.plotly_chart(fig, use_container_width=True)


def metrics_row(metrics: dict[str, Any], keys: list[tuple[str, str]]) -> None:
    cols = st.columns(len(keys))
    for col, (label, key) in zip(cols, keys):
        with col:
            card(label, metrics.get(key))


def read_artifacts(dataset_key: str) -> tuple[dict[str, Any], dict[str, str]]:
    contract = load_json("docs/topic2_visualization_contract.json")
    spec = contract["datasets"][dataset_key]
    return spec, spec["artifacts"]


def show_source_note(spec: dict[str, Any]) -> None:
    st.caption(
        f"Cột dùng để gom nhóm: {', '.join(spec['feature_columns'])}. "
        f"Cột nhãn `{spec['label_column_for_evaluation_only']}` chỉ dùng để đối chiếu, không dùng làm đầu vào."
    )


def overview() -> None:
    st.header("Tổng quan")
    st.write("Ba bộ dữ liệu được chọn để làm rõ ba cách gom nhóm khác nhau.")

    metrics = load_csv("outputs/topic2/results/topic2_algorithm_metrics.csv")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("KMeans")
        st.write("Seeds: dữ liệu số gọn, phù hợp để minh họa tâm cụm và khoảng cách đến centroid.")
    with col2:
        st.subheader("Hierarchical clustering")
        st.write("User Knowledge: phù hợp để xem dendrogram và từng bước gộp cụm.")
    with col3:
        st.subheader("DBSCAN")
        st.write("Spiral: cụm có hình dạng xoắn, giúp thấy điểm mạnh của gom nhóm theo mật độ.")


def kmeans_view() -> None:
    spec, art = read_artifacts("seeds_kmeans")
    st.header("KMeans — Seeds")
    show_source_note(spec)
    metrics = load_json(art["metrics_json"])
    metrics_row(metrics, [("Số dòng", "n_rows"), ("k", "k"), ("Inertia", "inertia"), ("Silhouette", "silhouette"), ("ARI với nhãn", "adjusted_rand_vs_label")])

    points = load_csv(art["points_csv"])
    summary = load_csv(art["summary_csv"])
    steps = load_csv(art["steps_csv"])
    centroids = load_csv(art["centroids_csv"])

    left, right = st.columns([2, 1])
    with left:
        scatter(points, title="Kết quả gom nhóm Seeds", hover_cols=["variety_name", "area", "perimeter"])
    with right:
        bar(summary, title="Kích thước cụm")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Các bước KMeans")
    st.caption(spec["centroid_note"])
    assign_steps = steps[steps["action"].str.startswith("assign")]
    fig = px.line(assign_steps, x="step", y="inertia", markers=True, title="Inertia qua từng bước gán cụm", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(steps, use_container_width=True, hide_index=True)

    with st.expander("Lịch sử centroid"):
        st.dataframe(centroids, use_container_width=True, hide_index=True)


def hierarchical_view() -> None:
    spec, art = read_artifacts("user_knowledge_hierarchical")
    st.header("Hierarchical clustering — User Knowledge")
    st.info("Các kết quả minh họa dùng mẫu 90 dòng. File nguồn đầy đủ có 403 dòng.")
    show_source_note(spec)
    metrics = load_json(art["metrics_json"])
    metrics_row(metrics, [("Số dòng", "n_rows"), ("k", "k"), ("Linkage", "linkage"), ("Silhouette", "silhouette"), ("ARI với nhãn", "adjusted_rand_vs_label")])

    points = load_csv(art["points_csv"])
    summary = load_csv(art["summary_csv"])
    merges = load_csv(art["merges_csv"])
    distance = load_csv(art["distance_preview_csv"])

    left, right = st.columns([2, 1])
    with left:
        scatter(points, title="Cụm hồ sơ học tập", hover_cols=["student_id", "UNS", "STG", "SCG", "PEG"])
    with right:
        bar(summary, title="Kích thước cụm")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Dendrogram")
    if require_file(art["dendrogram_png"]):
        st.image(str(local_path(art["dendrogram_png"])), use_container_width=True)

    st.subheader("Các bước gộp cụm")
    st.dataframe(merges, use_container_width=True, hide_index=True)

    with st.expander("Ma trận khoảng cách rút gọn"):
        st.dataframe(distance, use_container_width=True, hide_index=True)


def dbscan_view() -> None:
    spec, art = read_artifacts("spiral_dbscan")
    st.header("DBSCAN — Spiral")
    show_source_note(spec)
    st.caption(spec["metric_note"])
    metrics = load_json(art["metrics_json"])
    metrics_row(metrics, [("Số dòng", "n_rows"), ("eps", "eps"), ("min_samples", "min_samples"), ("Số cụm", "n_clusters_excluding_noise"), ("Điểm nhiễu", "noise_points"), ("ARI với nhãn", "adjusted_rand_vs_label")])

    points = load_csv(art["points_csv"])
    summary = load_csv(art["summary_csv"])
    kdist = load_csv(art["k_distance_csv"])
    comparison = load_csv(art["kmeans_comparison_points_csv"])

    left, right = st.columns([2, 1])
    with left:
        scatter(points, title="Kết quả DBSCAN", hover_cols=["true_label", "point_type"])
    with right:
        bar(summary, title="Kích thước cụm")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.write("Loại điểm")
        st.dataframe(points["point_type"].value_counts().rename_axis("point_type").reset_index(name="n_rows"), hide_index=True)

    # st.subheader("Chọn eps")
    # fig = px.line(kdist, x="rank", y="k_distance", title="Đường k-distance", height=360)
    # st.plotly_chart(fig, use_container_width=True)

    st.subheader("So sánh với KMeans")
    st.write("KMeans chia Spiral theo khoảng cách đến centroid, nên không giữ được hình dạng xoắn của cụm.")
    scatter(comparison, title="KMeans trên cùng bộ dữ liệu Spiral", hover_cols=["true_label"])


def raw_files_view() -> None:
    st.header("File handoff")
    contract = load_json("docs/topic2_visualization_contract.json")
    rows = []
    for dataset, spec in contract["datasets"].items():
        for name, rel_path in spec["artifacts"].items():
            path = local_path(rel_path)
            rows.append({
                "dataset": dataset,
                "artifact": name,
                "path": rel_path,
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else None,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Visualization contract"):
        st.json(contract)


def main() -> None:
    st.title("CT312 Chủ đề 2")
    st.caption("Ứng dụng trực quan hóa cho bộ dữ liệu gom nhóm.")

    if not CONTRACT_PATH.exists():
        st.error("Thiếu visualization contract. Hãy tạo lại bộ dữ liệu trước.")
        st.code("python scripts/topic2/01_prepare_demo_data.py\npython scripts/topic2/02_validate_handoff.py")
        return

    page = st.sidebar.radio(
        "Điều hướng",
        ["Tổng quan", "Upload"],
        index=0,
    )

    if page == "Tổng quan":
        overview()
    elif page == "Upload":
        upload_view()

def _render_upload_header(feature_cols: list[str], label_col: str | None) -> None:
    st.write(f"Cột dùng để gom nhóm: {', '.join(feature_cols)}")
    if label_col:
        st.caption(f"Cột nhãn `{label_col}` chỉ dùng để đối chiếu, không dùng làm đầu vào.")
    else:
        st.caption("Không dùng cột nhãn để đối chiếu.")


def _render_upload_kmeans(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    metrics_row(metrics, [("Số dòng", "n_rows"), ("k", "k"), ("Inertia", "inertia"), ("Silhouette", "silhouette"), ("ARI với nhãn", "adjusted_rand_vs_label")])

    points = result["points"]
    summary = result["summary"]
    steps = result["trace"]
    centroids = result["centroids"]
    elbow = result["elbow"]

    left, right = st.columns([2, 1])
    with left:
        scatter(points, title="Kết quả gom nhóm từ dữ liệu upload", hover_cols=[c for c in ["true_label", "point_type"] if c in points.columns])
    with right:
        bar(summary, title="Kích thước cụm")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Các bước KMeans")

    st.subheader("Tham khảo chọn số cụm bằng Elbow Method")

    fig = px.line(
        elbow,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow Method",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Điểm khuỷu tay là vị trí mà Inertia bắt đầu giảm chậm hơn. "
        "Giá trị k tại vị trí này thường được chọn làm số cụm ban đầu."
    )

    assign_steps = steps[steps["action"].str.startswith("assign")]
    fig = px.line(assign_steps, x="step", y="inertia", markers=True, title="Inertia qua từng bước gán cụm", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(steps, use_container_width=True, hide_index=True)

    with st.expander("Lịch sử centroid"):
        st.dataframe(centroids, use_container_width=True, hide_index=True)


def _render_upload_hierarchical(result: dict[str, object], row_labels: list[str]) -> None:
    metrics = result["metrics"]
    metrics_row(metrics, [("Số dòng", "n_rows"), ("k", "k"), ("Linkage", "linkage"), ("Silhouette", "silhouette"), ("ARI với nhãn", "adjusted_rand_vs_label")])

    points = result["points"]
    summary = result["summary"]
    merges = result["merges"]
    distance = result["distance_matrix"]
    linkage_matrix = result["linkage_matrix"]

    left, right = st.columns([2, 1])
    with left:
        scatter(points, title="Cụm hồ sơ từ dữ liệu upload", hover_cols=[c for c in ["student_id", "UNS", "STG", "SCG", "PEG"] if c in points.columns])
    with right:
        bar(summary, title="Kích thước cụm")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Dendrogram")
    max_cluster = len(points)

    selected_k = st.slider(
        "Số cụm cần cắt",
        min_value=2,
        max_value=max_cluster,
        value=metrics["k"],
    )

    n = linkage_matrix.shape[0] + 1

    # linkage có n-1 lần merge
    # Muốn còn k cụm thì lấy merge thứ n-k
    cut_height = linkage_matrix[n - selected_k - 1, 2]

    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, labels=row_labels[: len(points)], ax=ax, no_labels=True)

    ax.axhline(
        y=cut_height,
        color="red",
        linestyle="--",
        linewidth=2,
    )

    ax.text(
        0,
        cut_height,
        f"k = {selected_k}",
        color="red",
        fontsize=10,
        verticalalignment="bottom",
    )
    ax.set_ylabel("distance")
    ax.set_xlabel("")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Các bước gộp cụm")
    st.dataframe(merges, use_container_width=True, hide_index=True)

    with st.expander("Ma trận khoảng cách rút gọn"):
        st.dataframe(distance, use_container_width=True)


def _render_upload_dbscan(prepared: object, result: dict[str, object]) -> None:
    metrics = result["metrics"]
    metrics_row(metrics, [("Số dòng", "n_rows"), ("eps", "eps"), ("min_samples", "min_samples"), ("Số cụm", "n_clusters_excluding_noise"), ("Điểm nhiễu", "noise_points"), ("ARI với nhãn", "adjusted_rand_vs_label")])

    points = result["points"]
    summary = result["summary"]
    kdist = result["k_distance"]
    comparison = run_kmeans(prepared, k=min(3, len(prepared.X_scaled)))["points"]

    left, right = st.columns([2, 1])
    with left:
        scatter(points, title="Kết quả DBSCAN từ dữ liệu upload", hover_cols=[c for c in ["true_label", "point_type"] if c in points.columns])
    with right:
        bar(summary, title="Kích thước cụm")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if "point_type" in points.columns:
            st.write("Loại điểm")
            st.dataframe(points["point_type"].value_counts().rename_axis("point_type").reset_index(name="n_rows"), hide_index=True)

    st.subheader("Chọn eps")
    fig = px.line(kdist, x="rank", y="k_distance", title="Đường k-distance", height=360)
    st.plotly_chart(fig, use_container_width=True)

    # st.subheader("So sánh với KMeans")
    # st.write("KMeans chia dữ liệu theo khoảng cách đến centroid, nên thường không giữ tốt cấu trúc không lồi.")
    # scatter(comparison, title="KMeans trên cùng dữ liệu upload", hover_cols=[c for c in ["true_label", "point_type"] if c in comparison.columns])

    st.subheader("So sánh DBSCAN và KMeans")

    st.write(
        "DBSCAN giữ được các cụm có hình dạng bất kỳ và phát hiện điểm nhiễu, "
        "trong khi KMeans chia dữ liệu dựa trên khoảng cách tới centroid."
    )

    left, right = st.columns(2)

    with left:
        scatter(
            points,
            title="DBSCAN",
            hover_cols=[
                c for c in ["true_label", "point_type"]
                if c in points.columns
            ],
        )

    with right:
        scatter(
            comparison,
            title="KMeans",
            hover_cols=[
                c for c in ["true_label", "point_type"]
                if c in comparison.columns
            ],
        )

def upload_view() -> None:
    st.header("Upload dữ liệu")
    uploaded = st.file_uploader("Chọn file CSV và EXCEL", type=["csv","xlsx"])

    if uploaded is None:
        return

    try:
        # df = pd.read_csv(uploaded)
        file_type = Path(uploaded.name).suffix.lower()

        if file_type == ".csv":
            df = pd.read_csv(uploaded)

        elif file_type == ".xlsx":
            df = pd.read_excel(uploaded)

        else:
            st.error("Định dạng file không được hỗ trợ.")
            return
    except pd.errors.EmptyDataError:
        st.error("Không thể đọc CSV: file không có cột hoặc không có dữ liệu.")
        return
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error("Không thể đọc CSV vì định dạng hoặc mã hóa file không hợp lệ.")
        st.warning(f"Chi tiết lỗi: {error}")
        return
    except Exception as error:
        st.error("Không thể đọc file CSV đã tải lên.")
        st.warning(f"Chi tiết lỗi: {error}")
        return

    if df.empty:
        st.error("File CSV không có dòng dữ liệu. Hãy tải lên file có ít nhất 2 dòng dữ liệu.")
        return

    # st.dataframe(df.head(20), use_container_width=True)
    st.header("Phân tích dữ liệu")
    st.dataframe(df.dtypes.rename("Kiểu dữ liệu"))

    st.subheader("Xem trước dữ liệu")
    st.write(f"Kích thước dữ liệu: {df.shape[0]} dòng × {df.shape[1]} cột")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.error("CSV không có cột số để gom nhóm.")
        st.warning("Các cột văn bản như tên, giới tính hoặc thành phố chưa được mã hóa; hãy chọn hoặc thêm cột số.")
        return

    feature_cols = st.multiselect(
        "Chọn cột dùng để gom nhóm",
        numeric_cols,
        default=numeric_cols[:2],
    )
    if not feature_cols:
        st.error("Chưa chọn cột số để gom nhóm. Hãy chọn ít nhất một cột đặc trưng.")
        return

    report = preprocessing_report(df, feature_cols)

    show_preprocessing(report)

    errors = upload_data_errors(df, feature_cols)
    if errors:
        show_upload_errors(errors)
        return

    missing_counts = df[feature_cols].apply(pd.to_numeric, errors="coerce").isna().sum()
    if missing_counts.any():
        affected = [f"`{column}` ({count} ô)" for column, count in missing_counts.items() if count]
        st.info("Giá trị thiếu hoặc không phải số sẽ được điền bằng trung vị: " + ", ".join(affected) + ".")

    algorithm = st.selectbox(
        "Chọn giải thuật",
        ["KMeans", "Hierarchical", "DBSCAN"],
    )

    label_col = st.selectbox(
        "Cột nhãn để đối chiếu, nếu có",
        ["Không có"] + df.columns.tolist(),
    )
    label_col = None if label_col == "Không có" else label_col
    _render_upload_header(feature_cols, label_col)

    if algorithm == "KMeans":
        st.header("KMeans — Upload")
        st.caption("Centroid và inertia được tính trên dữ liệu đã chuẩn hóa.")

        left, right = st.columns([1, 2])

        with left:
            k = st.number_input(
                "Số cụm k",
                min_value=2,
                value=3,
                step=1,
            )

        with right:
            st.info(
                "**Phương pháp Elbow** giúp tham khảo số cụm phù hợp.\n\n"
                "Quan sát đồ thị Inertia theo k và chọn vị trí "
                "mà đường cong bắt đầu giảm chậm (điểm khuỷu tay). "
                "Đây chỉ là gợi ý, bạn vẫn có thể chọn giá trị k khác."
            )
        if int(k) > len(df):
            show_upload_errors([
                f"KMeans yêu cầu số cụm k không vượt quá số dòng dữ liệu ({len(df)}), nhưng bạn đã chọn k={int(k)}."
            ])
            return
        try:
            prepared = prepare_numeric_frame(df, feature_cols, label_column=label_col)
            result = run_kmeans(prepared, k=int(k))
        except (ValueError, TypeError) as error:
            st.error("Không thể tiền xử lý hoặc chạy KMeans với dữ liệu hiện tại.")
            st.warning(f"Chi tiết lỗi: {error}")
            return
        _render_upload_kmeans(result)
    elif algorithm == "Hierarchical":
        st.header("Hierarchical clustering — Upload")
        max_rows = len(df)
        row_count = st.number_input(
            "Số dòng dùng để tính",
            min_value=2,
            max_value=max_rows,
            value=min(90, max_rows),
            step=1,
        )
        sampled_df = df.head(int(row_count)).copy()
        st.info(f"Chỉ tính và vẽ trên {len(sampled_df)} dòng đầu tiên trong file upload.")
        # st.dataframe(sampled_df.head(20), use_container_width=True)
        k = st.number_input("Số cụm k", min_value=2, value=3, step=1)
        if int(k) > len(sampled_df):
            show_upload_errors([
                f"Hierarchical clustering yêu cầu k không vượt quá số dòng được chọn ({len(sampled_df)}), nhưng bạn đã chọn k={int(k)}."
            ])
            return
        try:
            sampled_prepared = prepare_numeric_frame(sampled_df, feature_cols, label_column=label_col)
            row_labels = [str(i) for i in sampled_df.index]
            result = run_hierarchical(sampled_prepared, k=int(k), method="ward", row_labels=row_labels)
        except (ValueError, TypeError) as error:
            st.error("Không thể tiền xử lý hoặc chạy Hierarchical clustering với dữ liệu hiện tại.")
            st.warning(f"Chi tiết lỗi: {error}")
            return
        _render_upload_hierarchical(result, row_labels)
    else:
        st.header("DBSCAN — Upload")
        st.caption(
            "Silhouette không phải chỉ số chính cho cụm dạng xoắn; "
            "nên ưu tiên scatter plot và ARI."
        )

        try:
            prepared = prepare_numeric_frame(
            df,
            feature_cols,
            label_column=label_col,
        )
        except (ValueError, TypeError) as error:
            st.error("Không thể tiền xử lý dữ liệu.")
            st.warning(f"Chi tiết lỗi: {error}")
            return

        st.subheader("Tham khảo chọn eps")

        preview_min_samples = 5

        preview = k_distance_table(
            prepared.X_scaled,
            k=preview_min_samples,
        )

        fig = px.line(
            preview,
            x="rank",
            y="k_distance",
            markers=True,
            title="Đường k-distance",
            height=360,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
                "Quan sát điểm chuyển tiếp của đường k-distance. Giá trị k-distance tại vị trí này thường được chọn làm eps."
        )

        eps = st.number_input(
            "eps",
            min_value=0.01,
            value=0.30,
            step=0.01,
        )

        min_samples = st.number_input(
            "min_samples",
            min_value=2,
            value=5,
            step=1,
        )

        if int(min_samples) > len(df):
            show_upload_errors([
                f"DBSCAN yêu cầu min_samples không vượt quá số dòng dữ liệu ({len(df)}), "
                f"nhưng bạn đã chọn min_samples={int(min_samples)}."
            ])
            return

        try:
            result = run_dbscan(
                prepared,
                eps=float(eps),
                min_samples=int(min_samples),
            )
        except (ValueError, TypeError) as error:
            st.error("Không thể chạy DBSCAN với dữ liệu hiện tại.")
            st.warning(f"Chi tiết lỗi: {error}")
            return

        _render_upload_dbscan(prepared, result)

if __name__ == "__main__":
    main()

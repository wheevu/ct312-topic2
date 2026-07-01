from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


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
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel_path, encoding="utf-8")


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

    st.subheader("Chọn eps")
    fig = px.line(kdist, x="rank", y="k_distance", title="Đường k-distance", height=360)
    st.plotly_chart(fig, use_container_width=True)

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

    overview_tab, kmeans_tab, hierarchical_tab, dbscan_tab, files_tab = st.tabs([
        "Tổng quan",
        "KMeans",
        "Hierarchical",
        "DBSCAN",
        "Files",
    ])

    with overview_tab:
        overview()
    with kmeans_tab:
        kmeans_view()
    with hierarchical_tab:
        hierarchical_view()
    with dbscan_tab:
        dbscan_view()
    with files_tab:
        raw_files_view()


if __name__ == "__main__":
    main()

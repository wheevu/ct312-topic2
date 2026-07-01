"""Shared analysis helpers for Topic 2 clustering demos.

The functions here return visualization-ready tables instead of app-specific
objects so a separate web teammate can consume CSV/JSON directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str | None
    X_scaled: np.ndarray
    coords: np.ndarray


@dataclass(frozen=True)
class KMeansTraceResult:
    labels: np.ndarray
    centroids: np.ndarray
    trace: pd.DataFrame
    centroid_history: pd.DataFrame
    inertia: float


def prepare_numeric_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_column: str | None = None,
) -> PreparedData:
    """Median-impute, standardize, and produce 2D coordinates for plotting."""
    clean = frame.copy()
    X = clean[feature_columns].apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    if X_scaled.shape[1] == 1:
        coords = np.column_stack([X_scaled[:, 0], np.zeros(len(X_scaled))])
    elif X_scaled.shape[1] == 2:
        coords = X_scaled
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    return PreparedData(clean, feature_columns, label_column, X_scaled, coords)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    non_noise = labels != -1
    usable_labels = labels[non_noise]
    if len(set(usable_labels.tolist())) < 2:
        return None
    if len(usable_labels) < 3:
        return None
    if len(set(usable_labels.tolist())) >= len(usable_labels):
        return None
    return float(silhouette_score(X[non_noise], usable_labels))


def labeled_points(prepared: PreparedData, labels: np.ndarray) -> pd.DataFrame:
    out = prepared.frame.copy()
    out["cluster"] = labels.astype(int)
    out["plot_x"] = prepared.coords[:, 0]
    out["plot_y"] = prepared.coords[:, 1]
    return out


def cluster_summary(labels: np.ndarray, true_labels: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        row = {
            "cluster": int(label),
            "cluster_name": "noise" if label == -1 else f"cluster {label}",
            "n_rows": int(mask.sum()),
            "share_pct": round(float(mask.mean() * 100), 2),
        }
        if true_labels is not None:
            vc = true_labels[mask].astype(str).value_counts()
            row["top_true_label"] = vc.index[0] if len(vc) else ""
            row["top_true_label_count"] = int(vc.iloc[0]) if len(vc) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def kmeans_trace(
    X: np.ndarray,
    *,
    k: int,
    feature_columns: list[str],
    random_state: int = 42,
    max_iter: int = 100,
) -> KMeansTraceResult:
    """Teaching KMeans run used as the official demo output."""
    rng = np.random.default_rng(random_state)
    init_idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[init_idx].copy()
    trace_rows = []
    centroid_rows = []
    prev_labels: np.ndarray | None = None
    labels = np.full(len(X), -1, dtype=int)

    for step in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        inertia = float(np.sum((X - centroids[labels]) ** 2))
        trace_rows.append({
            "step": step + 1,
            "action": "assign points to nearest centroid",
            "inertia": round(inertia, 6),
            "changed_assignments": None if prev_labels is None else int((labels != prev_labels).sum()),
        })
        for c, center in enumerate(centroids):
            centroid_rows.append({"step": step + 1, "cluster": c, **{name: float(v) for name, v in zip(feature_columns, center)}})

        new_centroids = centroids.copy()
        for c in range(k):
            if np.any(labels == c):
                new_centroids[c] = X[labels == c].mean(axis=0)
        shift = float(np.linalg.norm(new_centroids - centroids))
        trace_rows.append({
            "step": step + 1,
            "action": "update centroids from assigned points",
            "inertia": round(inertia, 6),
            "changed_assignments": None,
            "centroid_shift": round(shift, 6),
        })
        if shift < 1e-6:
            break
        prev_labels = labels.copy()
        centroids = new_centroids

    final_distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    final_labels = final_distances.argmin(axis=1)
    final_inertia = float(np.sum((X - centroids[final_labels]) ** 2))
    return KMeansTraceResult(final_labels, centroids, pd.DataFrame(trace_rows), pd.DataFrame(centroid_rows), final_inertia)


def run_kmeans(prepared: PreparedData, *, k: int = 3) -> dict[str, object]:
    traced = kmeans_trace(prepared.X_scaled, k=k, feature_columns=prepared.feature_columns)
    labels = traced.labels
    true = prepared.frame[prepared.label_column] if prepared.label_column else None
    return {
        "labels": labels,
        "points": labeled_points(prepared, labels),
        "summary": cluster_summary(labels, true),
        "trace": traced.trace,
        "centroids": traced.centroid_history,
        "metrics": {
            "algorithm": "kmeans",
            "k": k,
            "n_rows": int(len(labels)),
            "n_features": int(prepared.X_scaled.shape[1]),
            "inertia": traced.inertia,
            "silhouette": safe_silhouette(prepared.X_scaled, labels),
            "adjusted_rand_vs_label": float(adjusted_rand_score(true, labels)) if true is not None else None,
        },
    }


def hierarchical_merges(X: np.ndarray, row_labels: list[str], *, method: str = "ward") -> pd.DataFrame:
    Z = linkage(X, method=method)
    members: dict[int, list[str]] = {i: [row_labels[i]] for i in range(len(row_labels))}
    rows = []
    next_id = len(row_labels)
    for step, (a, b, distance, size) in enumerate(Z, start=1):
        ia, ib = int(a), int(b)
        merged = members[ia] + members[ib]
        rows.append({
            "step": step,
            "left": ", ".join(members[ia][:8]),
            "right": ", ".join(members[ib][:8]),
            "distance": round(float(distance), 6),
            "new_cluster_size": int(size),
            "members_preview": ", ".join(merged[:12]),
        })
        members[next_id] = merged
        next_id += 1
    return pd.DataFrame(rows)


def run_hierarchical(
    prepared: PreparedData,
    *,
    k: int = 4,
    method: str = "ward",
    row_labels: list[str] | None = None,
) -> dict[str, object]:
    model = AgglomerativeClustering(n_clusters=k, linkage=method)
    labels = model.fit_predict(prepared.X_scaled)
    true = prepared.frame[prepared.label_column] if prepared.label_column else None
    row_labels = row_labels or [str(i) for i in prepared.frame.index]
    return {
        "labels": labels,
        "points": labeled_points(prepared, labels),
        "summary": cluster_summary(labels, true),
        "merges": hierarchical_merges(prepared.X_scaled, row_labels, method=method),
        "distance_matrix": pd.DataFrame(squareform(pdist(prepared.X_scaled)), index=row_labels, columns=row_labels),
        "linkage_matrix": linkage(prepared.X_scaled, method=method),
        "metrics": {
            "algorithm": "hierarchical",
            "k": k,
            "linkage": method,
            "n_rows": int(len(labels)),
            "n_features": int(prepared.X_scaled.shape[1]),
            "silhouette": safe_silhouette(prepared.X_scaled, labels),
            "adjusted_rand_vs_label": float(adjusted_rand_score(true, labels)) if true is not None else None,
        },
    }


def dbscan_point_types(X: np.ndarray, labels: np.ndarray, model: DBSCAN) -> list[str]:
    core = set(model.core_sample_indices_.tolist())
    types = []
    for i, label in enumerate(labels):
        if label == -1:
            types.append("noise")
        elif i in core:
            types.append("core")
        else:
            types.append("border")
    return types


def k_distance_table(X: np.ndarray, *, k: int) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, -1])
    return pd.DataFrame({"rank": np.arange(1, len(kth) + 1), "k_distance": kth})


def run_dbscan(prepared: PreparedData, *, eps: float = 0.30, min_samples: int = 5) -> dict[str, object]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(prepared.X_scaled)
    true = prepared.frame[prepared.label_column] if prepared.label_column else None
    points = labeled_points(prepared, labels)
    points["point_type"] = dbscan_point_types(prepared.X_scaled, labels, model)
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)
    return {
        "labels": labels,
        "points": points,
        "summary": cluster_summary(labels, true),
        "k_distance": k_distance_table(prepared.X_scaled, k=min_samples),
        "metrics": {
            "algorithm": "dbscan",
            "eps": eps,
            "min_samples": min_samples,
            "n_rows": int(len(labels)),
            "n_features": int(prepared.X_scaled.shape[1]),
            "n_clusters_excluding_noise": len(clusters),
            "noise_points": int((labels == -1).sum()),
            "noise_pct": round(float((labels == -1).mean() * 100), 2),
            "silhouette_excluding_noise": safe_silhouette(prepared.X_scaled, labels),
            "adjusted_rand_vs_label": float(adjusted_rand_score(true, labels)) if true is not None else None,
        },
    }

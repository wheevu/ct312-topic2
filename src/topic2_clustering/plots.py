"""Plot helpers for Topic 2 generated artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def save_cluster_scatter(points: pd.DataFrame, path: Path, *, title: str, color_col: str = "cluster") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = sorted(points[color_col].unique().tolist())
    for label in labels:
        sub = points[points[color_col] == label]
        marker = "x" if label == -1 or str(label).lower() == "noise" else "o"
        ax.scatter(sub["plot_x"], sub["plot_y"], s=28, alpha=0.78, marker=marker, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("plot_x")
    ax.set_ylabel("plot_y")
    ax.legend(title=color_col, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar(summary: pd.DataFrame, path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["cluster_name"].astype(str), summary["n_rows"], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("cluster")
    ax.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_k_distance(kdist: pd.DataFrame, path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kdist["rank"], kdist["k_distance"], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("points sorted by k-distance")
    ax.set_ylabel("k-distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_dendrogram(linkage_matrix, labels: list[str], path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

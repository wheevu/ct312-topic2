#!/usr/bin/env python3
"""Validate Topic 2 visualization handoff files.

Run after `01_prepare_demo_data.py`:
    python scripts/topic2/02_validate_handoff.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "topic2_visualization_contract.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Validation failed: {message}")


def check_csv(path: Path, required_columns: list[str], row_count: int | None = None) -> None:
    require(path.exists(), f"missing CSV: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in required_columns if c not in frame.columns]
    require(not missing, f"{path} missing columns: {missing}")
    if row_count is not None:
        require(len(frame) == row_count, f"{path} row count {len(frame)} != expected {row_count}")


def main() -> None:
    require(CONTRACT.exists(), f"missing contract: {CONTRACT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract.get("schema_version") == 1, "schema_version must be 1")

    for name, spec in contract["datasets"].items():
        source = ROOT / spec["source_csv"]
        generated_from = ROOT / spec.get("generated_from_csv", spec["source_csv"])
        require(source.exists(), f"{name}: missing source_csv {source}")
        require(generated_from.exists(), f"{name}: missing generated_from_csv {generated_from}")

        artifacts = spec["artifacts"]
        points = ROOT / artifacts["points_csv"]
        check_csv(points, spec["required_point_columns"], spec["row_count"])
        check_csv(ROOT / artifacts["summary_csv"], ["cluster", "cluster_name", "n_rows", "share_pct"])
        require((ROOT / artifacts["metrics_json"]).exists(), f"{name}: missing metrics JSON")
        require((ROOT / artifacts["scatter_png"]).exists(), f"{name}: missing scatter PNG")
        require((ROOT / artifacts["cluster_sizes_png"]).exists(), f"{name}: missing cluster size PNG")

    metrics = ROOT / contract["metrics_csv"]
    check_csv(metrics, ["dataset", "algorithm", "n_rows", "n_features"])
    require((ROOT / contract["notes_md"]).exists(), "missing notes markdown")
    print("Topic 2 handoff validation passed")


if __name__ == "__main__":
    main()

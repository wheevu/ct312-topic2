"""Path constants for Topic 2 clustering demos."""

from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT: Path = _project_root()
DATA_DIR: Path = ROOT / "data" / "topic2"
RAW_DIR: Path = DATA_DIR / "raw"
EXAMPLES_DIR: Path = DATA_DIR / "examples"
DOCS_DIR: Path = ROOT / "docs"
OUTPUT_DIR: Path = ROOT / "outputs" / "topic2"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
RESULTS_DIR: Path = OUTPUT_DIR / "results"
HANDOFF_DIR: Path = OUTPUT_DIR / "handoff"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RAW_DIR, EXAMPLES_DIR, DOCS_DIR, OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR, HANDOFF_DIR):
        directory.mkdir(parents=True, exist_ok=True)

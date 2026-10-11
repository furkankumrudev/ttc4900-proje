"""Deney çıktıları: metrik json'ları ve tahmin dosyaları (tüm fazlarda ortak)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ttc4900.paths import METRICS_DIR, PROJECT_ROOT


def git_commit() -> str:
    """Kısa HEAD hash'i; çalışma ağacı kirliyse '-dirty' eki. git yoksa 'unknown'."""
    try:
        run = dict(cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **run).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], **run).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def now() -> datetime:
    return datetime.now(UTC)


def to_jsonable(obj):
    """tuple/numpy tiplerini json'a uygun hale getirir (ör. ngram_range=(1, 2) -> [1, 2])."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_metrics(obj: dict, name: str, out_dir: Path = METRICS_DIR, when=None) -> Path:
    """`<out_dir>/<name>_<YYYYMMDD-HHMMSS>.json` yazar. `split` alanı zorunludur (CLAUDE.md)."""
    if not obj.get("split"):
        raise ValueError("Metrik json'u hangi split'i kullandığını 'split' alanıyla belirtmeli.")
    when = when or now()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{when:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

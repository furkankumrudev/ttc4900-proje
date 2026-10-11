"""Proje yolları: hepsi proje köküne göre çözülür (yerel ve Colab'da aynı çalışır)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS / "splits"
FIGURES_DIR = OUTPUTS / "figures"
METRICS_DIR = OUTPUTS / "metrics"
MODELS_DIR = OUTPUTS / "models"
DEDUP_DIR = OUTPUTS / "dedup"

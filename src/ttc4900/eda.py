"""Keşifsel veri analizi. Metin istatistikleri yalnızca train'den; test'ten sınıf sayıları."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ttc4900.data import (  # noqa: E402
    drop_exact_duplicates,
    load_raw,
    load_train_test,
    update_report,
)
from ttc4900.paths import FIGURES_DIR  # noqa: E402

DPI = 150


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=DPI)
    plt.close(fig)


def _summary(x: pd.Series) -> dict:
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p95": float(np.percentile(x, 95)),
        "min": int(x.min()),
        "max": int(x.max()),
    }


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    raw = load_raw()
    clean, _ = drop_exact_duplicates(raw)
    train, test = load_train_test()
    order = sorted(raw["category"].unique())

    # 1) ham vs temiz sınıf dağılımı
    both = pd.concat([raw.assign(veri="ham"), clean.assign(veri="temiz")])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=both, x="category", hue="veri", order=order, ax=ax)
    ax.set(title="Sınıf dağılımı: ham ve temiz veri", xlabel="Sınıf", ylabel="Örnek sayısı")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))  # çubukların üstüne binmesin
    _save(fig, "eda_class_distribution.png")

    # 2) train vs test sınıf oranları
    props = pd.DataFrame(
        {
            "train": train["category"].value_counts(normalize=True),
            "test": test["category"].value_counts(normalize=True),
        }
    ).loc[order]
    long = props.reset_index(names="category").melt("category", var_name="split", value_name="oran")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long, x="category", y="oran", hue="split", ax=ax)
    ax.set(title="Sınıf oranları: train vs test", xlabel="Sınıf", ylabel="Oran")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    _save(fig, "eda_split_balance.png")

    # 3) kelime sayısı (yalnızca train)
    tr = train.assign(n_words=train["text"].str.split().str.len(), n_chars=train["text"].str.len())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=tr, x="category", y="n_words", order=order, ax=ax)
    ax.set_yscale("log")
    ax.set(title="Sınıf başına kelime sayısı (train, log ölçek)", xlabel="Sınıf", ylabel="Kelime")
    _save(fig, "eda_text_length.png")

    update_report(
        {
            "eda": {
                "scope_note": (
                    "Metin uzunluğu istatistikleri yalnızca train'den hesaplandı; "
                    "test'ten yalnızca sınıf sayıları (eda_split_balance) kullanıldı."
                ),
                "n_train": len(tr),
                "n_words": _summary(tr["n_words"]),
                "n_chars": _summary(tr["n_chars"]),
                "median_words_by_class": {
                    str(k): float(v) for k, v in tr.groupby("category")["n_words"].median().items()
                },
            }
        }
    )
    print("EDA tamam:", FIGURES_DIR)


if __name__ == "__main__":
    main()

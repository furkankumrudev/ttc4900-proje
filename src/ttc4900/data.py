"""Veri yükleme, tekrar temizliği, tek seferlik split ve yakın-tekrar denetimi.

Sonraki fazların tek giriş noktası: `load_train_test()`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from ttc4900.paths import DEDUP_DIR, OUTPUTS, SPLITS_DIR

DATASET_ID = "savasy/ttc4900"
DATASET_REVISION = "236e17b0c497ceb5437c175cb1ffaae201bfb287"
SPLIT_PATH = SPLITS_DIR / "split_v1.json"
REPORT_PATH = OUTPUTS / "data_report.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
NEAR_DUP_THRESHOLD = 0.9

_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- yükleme
def load_raw() -> pd.DataFrame:
    """Hub'dan (sabit revision) ham veriyi yükler. row_id = orijinal Hub satır numarası."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    feat = ds.features["category"]
    labels = list(ds["category"])
    return pd.DataFrame(
        {
            "row_id": np.arange(len(ds)),
            "label": labels,
            "category": [feat.int2str(x) for x in labels],
            "text": list(ds["text"]),
        }
    )


def label_names() -> list[str]:
    """ClassLabel isimleri (indeks = label)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    return list(ds.features["category"].names)


# --------------------------------------------------------------------------- tekrarlar
def normalize_text(s: str) -> str:
    """Tekrar anahtarı için normalizasyon (modele giden metni DEĞİŞTİRMEZ).

    NFC -> Türkçe küçük harf (İ->i, I->ı, sonra lower) -> boşlukları tek boşluğa indir.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.replace("İ", "i").replace("I", "ı").lower()
    return _WS.sub(" ", s).strip()


def _preview(s: str, n: int = 100) -> str:
    return _WS.sub(" ", s).strip()[:n]


def count_raw_exact_duplicates(df: pd.DataFrame) -> int:
    """Normalizasyonsuz, birebir aynı metinlerin fazlalık satır sayısı (ilk kopya hariç)."""
    return int(df["text"].duplicated().sum())


def _drop_rec(r, reason: str, kept) -> dict:
    return {
        "row_id": int(r.row_id),
        "category": r.category,
        "reason": reason,
        "kept_row_id": kept,
        "text_preview": _preview(r.text),
    }


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize anahtara göre tekrarları atar.

    Etiketler aynıysa en küçük row_id kalır (reason=exact_duplicate, kept_row_id=kalan).
    Etiketler çelişiyorsa grubun tamamı atılır (reason=conflicting_labels, kept_row_id=NA).
    """
    key = df["text"].map(normalize_text)
    keep_ids: list[int] = []
    dropped: list[dict] = []
    for _, g in df.assign(_key=key).groupby("_key", sort=False):
        g = g.sort_values("row_id")
        if len(g) == 1:
            keep_ids.append(int(g["row_id"].iloc[0]))
        elif g["label"].nunique() == 1:
            kept = int(g["row_id"].iloc[0])
            keep_ids.append(kept)
            for r in g.iloc[1:].itertuples():
                dropped.append(_drop_rec(r, "exact_duplicate", kept))
        else:
            for r in g.itertuples():
                dropped.append(_drop_rec(r, "conflicting_labels", pd.NA))
    clean = df[df["row_id"].isin(keep_ids)].sort_values("row_id").reset_index(drop=True)
    cols = ["row_id", "category", "reason", "kept_row_id", "text_preview"]
    dropped_df = pd.DataFrame(dropped, columns=cols)
    if len(dropped_df):
        dropped_df = dropped_df.sort_values("row_id").reset_index(drop=True)
    dropped_df["kept_row_id"] = dropped_df["kept_row_id"].astype("Int64")
    return clean, dropped_df


# --------------------------------------------------------------------------- split
def make_split(
    clean: pd.DataFrame,
    path: Path = SPLIT_PATH,
    overwrite: bool = False,
    n_raw: int | None = None,
) -> dict:
    """Stratified split'i row_id'ler üzerinde üretip JSON'a yazar (bir kez!)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} zaten var; split yalnızca bir kez üretilir.")
    train_ids, test_ids = train_test_split(
        clean["row_id"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean["label"].to_numpy(),
    )
    names = clean[["label", "category"]].drop_duplicates().sort_values("label")["category"].tolist()
    split = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "n_raw": int(n_raw) if n_raw is not None else None,
        "n_clean": int(len(clean)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": "label",
        "label_names": names,
        "train_ids": sorted(int(i) for i in train_ids),
        "test_ids": sorted(int(i) for i in test_ids),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return split


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_train_test(path: Path = SPLIT_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_df, test_df): tüm fazların tek giriş noktası. Sütunlar: row_id,label,category,text."""
    split = load_split(path)
    assert split["revision"] == DATASET_REVISION, "Split farklı bir veri revision'ı için üretilmiş."
    raw = load_raw().set_index("row_id", drop=False)
    train = raw.loc[split["train_ids"]].reset_index(drop=True)
    test = raw.loc[split["test_ids"]].reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------- yakın tekrar
def find_near_duplicates(
    clean: pd.DataFrame,
    split: dict,
    threshold: float = NEAR_DUP_THRESHOLD,
    block: int = 500,
) -> pd.DataFrame:
    """Kosinüs benzerliği > threshold olan çiftleri listeler. HİÇBİR satırı silmez.

    Not: bu TF-IDF tüm veriye (train+test) fit edilir, ancak yalnızca denetim içindir;
    model özelliği değildir ve hiçbir modele girmez.
    """
    texts = clean["text"].map(normalize_text).tolist()
    X = TfidfVectorizer().fit_transform(texts).tocsr()  # L2 norm: X @ X.T = kosinüs
    ids = clean["row_id"].to_numpy()
    labels = clean["label"].to_numpy()
    raw_texts = clean["text"].tolist()
    split_of = {i: "train" for i in split["train_ids"]} | {i: "test" for i in split["test_ids"]}
    n = X.shape[0]
    cols = np.arange(n)
    rows = []
    for start in range(0, n, block):
        sims = (X[start : start + block] @ X.T).toarray()
        gi = np.arange(start, start + sims.shape[0])[:, None]
        for r, c in np.argwhere((sims > threshold) & (cols[None, :] > gi)):
            i, j = start + r, c
            sa, sb = split_of.get(int(ids[i])), split_of.get(int(ids[j]))
            rows.append(
                {
                    "row_id_a": int(ids[i]),
                    "row_id_b": int(ids[j]),
                    "similarity": float(sims[r, c]),
                    "label_a": int(labels[i]),
                    "label_b": int(labels[j]),
                    "split_a": sa,
                    "split_b": sb,
                    "same_label": bool(labels[i] == labels[j]),
                    "cross_split": sa != sb,
                    "text_a_preview": _preview(raw_texts[i]),
                    "text_b_preview": _preview(raw_texts[j]),
                }
            )
    cols_out = [
        "row_id_a",
        "row_id_b",
        "similarity",
        "label_a",
        "label_b",
        "split_a",
        "split_b",
        "same_label",
        "cross_split",
        "text_a_preview",
        "text_b_preview",
    ]
    out = pd.DataFrame(rows, columns=cols_out)
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- rapor
def update_report(updates: dict, path: Path = REPORT_PATH) -> dict:
    """Var olan raporu okuyup alanları birleştirir (başka modülün alanlarını ezmez)."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def main() -> None:
    raw = load_raw()
    clean, dropped = drop_exact_duplicates(raw)

    if SPLIT_PATH.exists():
        split = load_split()
        assert split["revision"] == DATASET_REVISION
        assert split["n_clean"] == len(clean), "n_clean split ile uyuşmuyor"
        got = sorted(split["train_ids"] + split["test_ids"])
        assert got == clean["row_id"].tolist(), "split id'leri temiz kümeyle uyuşmuyor"
        print("Mevcut split kullanıldı:", SPLIT_PATH)
    else:
        split = make_split(clean, n_raw=len(raw))
        print("Yeni split yazıldı:", SPLIT_PATH)

    near = find_near_duplicates(clean, split)
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(DEDUP_DIR / "dropped_rows.csv", index=False, encoding="utf-8")
    near.to_csv(DEDUP_DIR / "near_duplicates.csv", index=False, encoding="utf-8")

    cat = clean.set_index("row_id")["category"]
    involved = set(near["row_id_a"]) | set(near["row_id_b"])
    update_report(
        {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "n_raw": len(raw),
            "raw_exact_duplicate_rows": count_raw_exact_duplicates(raw),
            "n_dropped_by_reason": {
                str(k): int(v) for k, v in dropped["reason"].value_counts().items()
            },
            "n_clean": len(clean),
            "n_train": len(split["train_ids"]),
            "n_test": len(split["test_ids"]),
            "class_distribution": {
                "raw": _counts(raw["category"]),
                "clean": _counts(clean["category"]),
                "train": _counts(cat.loc[split["train_ids"]]),
                "test": _counts(cat.loc[split["test_ids"]]),
            },
            "near_duplicates": {
                "threshold": NEAR_DUP_THRESHOLD,
                "n_pairs": len(near),
                "n_rows_involved": len(involved),
                "n_cross_label_pairs": int((~near["same_label"]).sum()),
                "n_cross_split_pairs": int(near["cross_split"].sum()),
                "examples": near.head(10).to_dict(orient="records"),
                "tfidf_fit_scope": (
                    "Denetim TF-IDF'i tüm temiz veriye (train+test) fit edildi; yalnızca "
                    "yakın-tekrar tespiti içindir, model özelliği değildir."
                ),
                "note": "Yakın tekrarlar silinmedi; yalnızca raporlandı.",
            },
        }
    )
    print("Rapor:", REPORT_PATH)


if __name__ == "__main__":
    main()

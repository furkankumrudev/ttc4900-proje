"""Veri yükleme, tekrar temizliği, tek seferlik split'ler ve yakın-tekrar denetimi.

Sonraki fazların tek giriş noktaları: `load_train_test()` ve `make_cv()`.
v1 = düz stratified split (yalnızca "sızıntı etkisi" baseline'ı için), v2 = yakın-tekrar
gruplarını bölmeyen resmî split.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ttc4900.paths import DEDUP_DIR, OUTPUTS, SPLITS_DIR

DATASET_ID = "savasy/ttc4900"
DATASET_REVISION = "236e17b0c497ceb5437c175cb1ffaae201bfb287"
SPLIT_V1_PATH = SPLITS_DIR / "split_v1.json"
SPLIT_V2_PATH = SPLITS_DIR / "split_v2.json"
SPLIT_PATH = SPLIT_V2_PATH  # resmî split: grup-bilinçli v2
REPORT_PATH = OUTPUTS / "data_report.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# 0.9 ilk tercihti; (0.8, 0.9] aralığında da aynı haberin yeniden yazımları bölünüyordu
# (ör. 867/958 Teksas kazası), en büyük grup 0.8'de de 5 kaldığı için eşik 0.8'e indirildi.
NEAR_DUP_THRESHOLD = 0.8
SENSITIVITY_THRESHOLDS = (0.8, 0.9, 0.95)
V2_N_FOLDS = round(1 / TEST_SIZE)  # v2 üretimi: fold 0 = test (~%20)
CV_SPLITS = 5  # train içi CV fold sayısı (v2 üretiminden bağımsız)
TFIDF_DESC = (
    "TfidfVectorizer() varsayılanları, normalize_text çıktısı üzerinde, tüm temiz veriye fit "
    "(yalnızca denetim/gruplama; model özelliği değil)"
)

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
def _label_names_of(clean: pd.DataFrame) -> list[str]:
    return clean[["label", "category"]].drop_duplicates().sort_values("label")["category"].tolist()


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_split(
    clean: pd.DataFrame,
    path: Path = SPLIT_V1_PATH,
    overwrite: bool = False,
    n_raw: int | None = None,
) -> dict:
    """v1: düz stratified split'i row_id'ler üzerinde üretip JSON'a yazar (bir kez!)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} zaten var; split yalnızca bir kez üretilir.")
    train_ids, test_ids = train_test_split(
        clean["row_id"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean["label"].to_numpy(),
    )
    split = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "n_raw": int(n_raw) if n_raw is not None else None,
        "n_clean": int(len(clean)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": "label",
        "label_names": _label_names_of(clean),
        "train_ids": sorted(int(i) for i in train_ids),
        "test_ids": sorted(int(i) for i in test_ids),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(split, path)
    return split


def make_split_v2(
    clean: pd.DataFrame,
    groups: np.ndarray,
    path: Path = SPLIT_V2_PATH,
    overwrite: bool = False,
    n_raw: int | None = None,
) -> dict:
    """v2: yakın-tekrar gruplarını bölmeyen stratified split (bir kez!).

    Sezgi: aynı grubun tüm üyeleri aynı tarafa gider; sınıf oranları mümkün olduğunca
    korunur (tam eşitlik garanti değil). StratifiedGroupKFold(V2_N_FOLDS, shuffle) fold 0 = test.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} zaten var; split yalnızca bir kez üretilir.")
    groups = np.asarray(groups)
    ids = clean["row_id"].to_numpy()
    y = clean["label"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=V2_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(iter(sgkf.split(np.zeros(len(clean)), y, groups)))
    sizes = pd.Series(groups).value_counts()  # grup -> üye sayısı
    hist = sizes.value_counts().sort_index()  # boyut -> grup sayısı
    comps = [sorted(int(i) for i in ids[groups == g]) for g in sizes.index[sizes.to_numpy() >= 2]]
    comps.sort()  # deterministik sıra
    split = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "n_raw": int(n_raw) if n_raw is not None else None,
        "n_clean": int(len(clean)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": "label",
        "method": (
            f"StratifiedGroupKFold(n_splits={V2_N_FOLDS}, shuffle=True, "
            f"random_state={RANDOM_STATE}), fold 0 = test"
        ),
        "threshold": NEAR_DUP_THRESHOLD,
        "tfidf_params": TFIDF_DESC,
        "label_names": _label_names_of(clean),
        "n_groups": int(len(sizes)),
        "group_size_hist": {str(int(k)): int(v) for k, v in hist.items()},
        "components": comps,
        "train_ids": sorted(int(i) for i in ids[tr_idx]),
        "test_ids": sorted(int(i) for i in ids[te_idx]),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(split, path)
    return split


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_train_test(path: Path = SPLIT_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_df, test_df): tüm fazların tek giriş noktası. Sütunlar: row_id,label,category,text."""
    split = load_split(path)
    if split["revision"] != DATASET_REVISION:
        raise ValueError("Split farklı bir veri revision'ı için üretilmiş.")
    raw = load_raw()
    if split.get("n_raw") is not None and len(raw) != split["n_raw"]:
        raise ValueError(f"n_raw uyuşmuyor: split={split['n_raw']}, veri={len(raw)}")
    if set(split["train_ids"]) & set(split["test_ids"]):
        raise ValueError("train ve test id'leri ayrık değil.")
    ids = np.array(split["train_ids"] + split["test_ids"])
    if len(ids) and (ids.min() < 0 or ids.max() >= len(raw)):
        raise ValueError("Split id'leri range(n_raw) dışında.")
    raw = raw.set_index("row_id", drop=False)
    train = raw.loc[split["train_ids"]].reset_index(drop=True)
    test = raw.loc[split["test_ids"]].reset_index(drop=True)
    return train, test


def cv_groups(train_df: pd.DataFrame, path: Path = SPLIT_PATH) -> np.ndarray:
    """train_df satırlarına hizalı CV grup id'leri (split dosyasındaki `components`'tan).

    Bileşen üyeleri bileşenin en küçük row_id'sini, tekil satırlar -(row_id+1) alır
    (negatif olduğu için bileşen id'leriyle çakışmaz).
    """
    split = load_split(path)
    if "components" not in split:
        raise ValueError(
            f"{path} grup bilgisi (components) içermiyor; v1 grupsuzdur. "
            "Gruplu CV için split_v2.json kullanın."
        )
    foreign = set(train_df["row_id"].astype(int)) - set(split["train_ids"])
    if foreign:
        raise ValueError(
            f"{len(foreign)} satır {Path(path).name} train'inde değil (test ya da başka split?)."
        )
    gid = {i: comp[0] for comp in split["components"] for i in comp}
    return np.array([gid.get(int(r), -(int(r) + 1)) for r in train_df["row_id"]], dtype=np.int64)


def make_cv(
    train_df: pd.DataFrame,
    path: Path = SPLIT_PATH,
    n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[StratifiedGroupKFold, np.ndarray]:
    """Faz 2+ CV'nin TEK giriş noktası; düz StratifiedKFold kullanılmaz.

    Kullanım: `cv, g = make_cv(train)`; `cv.split(train, train["label"], g)`.
    Yakın-tekrar grupları train/validation arasında bölünmez, CV skoru şişmez.
    `random_state` yalnızca tekrarlı CV içindir (her tekrar farklı bölümleme); g değişmez.
    `g`, çağrıldığı andaki `train_df` satır sırasına hizalıdır: CV'ye aynı DataFrame verilmeli.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cv, cv_groups(train_df, path)


# --------------------------------------------------------------------------- yakın tekrar
def similarity_pairs(clean: pd.DataFrame, threshold: float, block: int = 500) -> pd.DataFrame:
    """Kosinüs benzerliği > threshold olan (i<j) çiftler: row_id_a, row_id_b, similarity.

    Split'ten bağımsızdır. TF-IDF L2-normlu olduğundan X @ X.T = kosinüs; bellek için
    blok blok hesaplanır. TF-IDF tüm veriye fit edilir; yalnızca denetim/gruplama içindir.
    """
    texts = clean["text"].map(normalize_text).tolist()
    X = TfidfVectorizer().fit_transform(texts).tocsr()
    ids = clean["row_id"].to_numpy()
    n = X.shape[0]
    cols = np.arange(n)
    rows = []
    for start in range(0, n, block):
        sims = (X[start : start + block] @ X.T).toarray()
        gi = np.arange(start, start + sims.shape[0])[:, None]
        for r, c in np.argwhere((sims > threshold) & (cols[None, :] > gi)):
            rows.append((int(ids[start + r]), int(ids[c]), float(sims[r, c])))
    out = pd.DataFrame(rows, columns=["row_id_a", "row_id_b", "similarity"])
    return out.sort_values(
        ["similarity", "row_id_a", "row_id_b"], ascending=[False, True, True]
    ).reset_index(drop=True)


def near_dup_groups(
    clean: pd.DataFrame, threshold: float = NEAR_DUP_THRESHOLD, pairs: pd.DataFrame | None = None
) -> np.ndarray:
    """Yakın-tekrar grafının bağlı bileşenleri: clean satır sırasına hizalı grup id'leri.

    Düğüm = satır, kenar = benzerlik > threshold; zincir A-B, B-C tek grup olur
    (geçişlilik). Tekil satırlar kendi grubudur. Gruplar TÜM temiz veri üzerinden ve
    etiket KULLANILMADAN kurulur. `pairs` verilirse (daha düşük eşikte hesaplanmış olabilir)
    threshold ile süzülür.
    """
    if pairs is None:
        pairs = similarity_pairs(clean, threshold)
    pairs = pairs[pairs["similarity"] > threshold]
    pos = pd.Series(np.arange(len(clean)), index=clean["row_id"].to_numpy())
    a = pos.loc[pairs["row_id_a"].to_numpy()].to_numpy()
    b = pos.loc[pairs["row_id_b"].to_numpy()].to_numpy()
    n = len(clean)
    adj = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    _, lab = connected_components(adj, directed=False)
    return lab


def find_near_duplicates(
    clean: pd.DataFrame,
    split: dict,
    threshold: float = NEAR_DUP_THRESHOLD,
    block: int = 500,
    pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Kosinüs > threshold çiftlerini etiket/split bilgisiyle listeler. HİÇBİR satırı silmez.

    `pairs` verilirse yeniden hesaplamadan süzülür.
    """
    if pairs is None:
        pairs = similarity_pairs(clean, threshold, block)
    out = pairs[pairs["similarity"] > threshold].copy()
    lab = clean.set_index("row_id")["label"]
    txt = clean.set_index("row_id")["text"]
    split_of = {i: "train" for i in split["train_ids"]} | {i: "test" for i in split["test_ids"]}
    out["label_a"] = lab.loc[out["row_id_a"]].astype(int).to_numpy()
    out["label_b"] = lab.loc[out["row_id_b"]].astype(int).to_numpy()
    out["split_a"] = out["row_id_a"].map(split_of)
    out["split_b"] = out["row_id_b"].map(split_of)
    out["same_label"] = out["label_a"] == out["label_b"]
    out["cross_split"] = out["split_a"] != out["split_b"]
    out["text_a_preview"] = [_preview(txt.loc[i]) for i in out["row_id_a"]]
    out["text_b_preview"] = [_preview(txt.loc[i]) for i in out["row_id_b"]]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- rapor
def update_report(updates: dict, path: Path = REPORT_PATH) -> dict:
    """Var olan raporu okuyup alanları birleştirir (başka modülün alanlarını ezmez)."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report.update(updates)
    _write_json(report, path)
    return report


def _counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def _check_split(split: dict, clean: pd.DataFrame, name: str) -> None:
    """Mevcut split dosyasının temiz veriyle tutarlılığı."""
    if split["revision"] != DATASET_REVISION:
        raise ValueError(f"{name}: revision uyuşmuyor")
    if split["n_clean"] != len(clean):
        raise ValueError(f"{name}: n_clean uyuşmuyor")
    if sorted(split["train_ids"] + split["test_ids"]) != clean["row_id"].tolist():
        raise ValueError(f"{name}: id birleşimi temiz kümeyle uyuşmuyor")


def _n_cross(pairs: pd.DataFrame, split: dict) -> int:
    """train ile test arasına bölünmüş çift sayısı."""
    te = set(split["test_ids"])
    return int((pairs["row_id_a"].isin(te) != pairs["row_id_b"].isin(te)).sum())


def _sensitivity(clean: pd.DataFrame, pairs_low: pd.DataFrame, splits: dict) -> list[dict]:
    """Eşik taraması: aynı pairs üzerinden süzülerek, TF-IDF tekrar hesaplanmadan."""
    lab = clean["label"].to_numpy()
    table = []
    for t in SENSITIVITY_THRESHOLDS:
        g = near_dup_groups(clean, t, pairs_low)
        sizes = np.bincount(g)
        multi = np.flatnonzero(sizes >= 2)
        mixed = sum(len(set(lab[g == k])) > 1 for k in multi)
        p = pairs_low[pairs_low["similarity"] > t]
        row = {
            "threshold": t,
            "n_pairs": int(len(p)),
            "n_groups_multi": int(len(multi)),
            "n_rows_in_groups": int(sizes[multi].sum()),
            "largest_group_size": int(sizes.max()),
            "n_mixed_label_groups": int(mixed),
        }
        row |= {f"n_cross_split_pairs_{k}": _n_cross(p, s) for k, s in splits.items()}
        table.append(row)
    return table


def _class_table(clean: pd.DataFrame, split: dict) -> dict[str, dict]:
    cat = clean.set_index("row_id")["category"]
    tr, te = _counts(cat.loc[split["train_ids"]]), _counts(cat.loc[split["test_ids"]])
    return {
        c: {
            "train": tr.get(c, 0),
            "test": te.get(c, 0),
            "test_ratio": te.get(c, 0) / (tr.get(c, 0) + te.get(c, 0)),
        }
        for c in sorted(set(tr) | set(te))
    }


def main() -> None:
    raw = load_raw()
    clean, dropped = drop_exact_duplicates(raw)

    # --- v1: varsa dokunma, yoksa üret
    if SPLIT_V1_PATH.exists():
        split1 = load_split(SPLIT_V1_PATH)
        _check_split(split1, clean, "split_v1")
        print("Mevcut v1 kullanıldı:", SPLIT_V1_PATH)
    else:
        split1 = make_split(clean, n_raw=len(raw))
        print("Yeni v1 yazıldı:", SPLIT_V1_PATH)

    # --- çiftler (en düşük eşikte bir kez) ve hassasiyet
    pairs_low = similarity_pairs(clean, min(SENSITIVITY_THRESHOLDS + (NEAR_DUP_THRESHOLD,)))
    groups = near_dup_groups(clean, NEAR_DUP_THRESHOLD, pairs_low)

    # --- v2: varsa yükle ve doğrula, yoksa üret
    if SPLIT_V2_PATH.exists():
        split2 = load_split(SPLIT_V2_PATH)
        _check_split(split2, clean, "split_v2")
        print("Mevcut v2 kullanıldı:", SPLIT_V2_PATH)
    else:
        split2 = make_split_v2(clean, groups, n_raw=len(raw))
        print("Yeni v2 yazıldı:", SPLIT_V2_PATH)

    sens = _sensitivity(clean, pairs_low, {"v1": split1, "v2": split2})

    # --- near_duplicates.csv (> NEAR_DUP_THRESHOLD, yani gruplamada kullanılan tüm kenarlar)
    near = find_near_duplicates(clean, split1, NEAR_DUP_THRESHOLD, pairs=pairs_low)
    near = near.rename(
        columns={"split_a": "split_v1_a", "split_b": "split_v1_b", "cross_split": "cross_split_v1"}
    )
    s2 = {i: "train" for i in split2["train_ids"]} | {i: "test" for i in split2["test_ids"]}
    near["split_v2_a"] = near["row_id_a"].map(s2)
    near["split_v2_b"] = near["row_id_b"].map(s2)
    near["cross_split_v2"] = near["split_v2_a"] != near["split_v2_b"]
    gid = pd.Series(groups, index=clean["row_id"].to_numpy())
    near["group_id"] = gid.loc[near["row_id_a"]].to_numpy()
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(DEDUP_DIR / "dropped_rows.csv", index=False, encoding="utf-8")
    near.to_csv(DEDUP_DIR / "near_duplicates.csv", index=False, encoding="utf-8")

    # --- grup ayrıntıları (NEAR_DUP_THRESHOLD)
    gs = pd.Series(groups).value_counts()
    big = int(gs.index[0])
    big_df = clean[groups == big]
    sizes_hist = {str(int(k)): int(v) for k, v in gs.value_counts().sort_index().items()}
    s_main = next(r for r in sens if r["threshold"] == NEAR_DUP_THRESHOLD)
    ct1, ct2 = _class_table(clean, split1), _class_table(clean, split2)
    cat = clean.set_index("row_id")["category"]
    involved = set(near["row_id_a"]) | set(near["row_id_b"])
    n_cross_v1, n_cross_v2 = int(near["cross_split_v1"].sum()), int(near["cross_split_v2"].sum())

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
            # üst düzey alanlar resmî split'i (v2) yansıtır; v1 `split_v2.comparison_v1` altında
            "official_split": "split_v2",
            "n_train": len(split2["train_ids"]),
            "n_test": len(split2["test_ids"]),
            "class_distribution": {
                "raw": _counts(raw["category"]),
                "clean": _counts(clean["category"]),
                "train": _counts(cat.loc[split2["train_ids"]]),
                "test": _counts(cat.loc[split2["test_ids"]]),
            },
            "near_duplicates": {
                "threshold": NEAR_DUP_THRESHOLD,
                "n_pairs": len(near),
                "n_rows_involved": len(involved),
                "n_cross_label_pairs": int((~near["same_label"]).sum()),
                "n_cross_split_pairs_v1": n_cross_v1,
                "n_cross_split_pairs_v2": n_cross_v2,
                "examples": near.head(10).to_dict(orient="records"),
                "tfidf_fit_scope": (
                    "Denetim TF-IDF'i tüm temiz veriye (train+test) fit edildi; yalnızca "
                    "yakın-tekrar tespiti içindir, model özelliği değildir."
                ),
                "note": "Yakın tekrarlar silinmedi; v2'de gruplanıp aynı tarafa konuldu.",
            },
            "grouping": {
                "threshold": NEAR_DUP_THRESHOLD,
                "sensitivity": sens,
                "group_size_hist": sizes_hist,
                "largest_group": {
                    "size": len(big_df),
                    "row_ids": [int(i) for i in big_df["row_id"]],
                    "class_distribution": _counts(big_df["category"]),
                    "previews": [_preview(t) for t in big_df["text"].head(3)],
                },
                "n_mixed_label_groups": s_main["n_mixed_label_groups"],
                "fit_scope": (
                    "Gruplar tüm temiz veri (train+test) üzerinden, etiket kullanılmadan, "
                    "TF-IDF kosinüs > eşik grafının bağlı bileşenleriyle kuruldu. " + TFIDF_DESC
                ),
            },
            "split_v2": {
                "method": split2["method"],
                "n_train": len(split2["train_ids"]),
                "n_test": len(split2["test_ids"]),
                "test_ratio": len(split2["test_ids"]) / len(clean),
                "by_class": ct2,
                "comparison_v1": {
                    "n_train": len(split1["train_ids"]),
                    "n_test": len(split1["test_ids"]),
                    "test_ratio": len(split1["test_ids"]) / len(clean),
                    "by_class": ct1,
                },
                "threshold": NEAR_DUP_THRESHOLD,
                "n_cross_split_pairs_v1": n_cross_v1,
                "n_cross_split_pairs_v2": n_cross_v2,
                "threshold_note": (
                    "İlk tercih 0.9 idi; (0.8, 0.9] aralığında 19 çift bölünmüş kalıyordu ve "
                    "~8'i aynı haberin yeniden yazımıydı. En büyük grup 0.8'de de 5 olduğundan "
                    "eşik 0.8'e indirildi. Eşik başına v1/v2 bölünmüş çiftler: "
                    "grouping.sensitivity."
                ),
            },
        }
    )
    print("Rapor:", REPORT_PATH)


if __name__ == "__main__":
    main()

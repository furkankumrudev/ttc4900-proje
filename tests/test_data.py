"""Ağ gerektirmeyen sentetik veri testleri (+ varsa gerçek split dosyası kontrolü)."""

import numpy as np
import pandas as pd
import pytest

from ttc4900 import data
from ttc4900.data import (
    DATASET_REVISION,
    SPLIT_V1_PATH,
    SPLIT_V2_PATH,
    drop_exact_duplicates,
    find_near_duplicates,
    load_split,
    make_cv,
    make_split,
    make_split_v2,
    near_dup_groups,
    normalize_text,
)
from ttc4900.paths import DEDUP_DIR


def _df(rows):
    return pd.DataFrame(
        [
            {"row_id": i, "label": lab, "category": f"c{lab}", "text": t}
            for i, (lab, t) in enumerate(rows)
        ]
    )


def test_normalize_text():
    assert normalize_text("İSTANBUL") == "istanbul"
    assert normalize_text("IŞIK") == "ışık"
    assert normalize_text("  a \n\t b   c ") == "a b c"


def test_drop_exact_duplicates():
    df = _df([(0, "Merhaba  dünya"), (1, "başka"), (0, "merhaba dünya"), (1, "x"), (2, "x")])
    clean, dropped = drop_exact_duplicates(df)
    assert clean["row_id"].tolist() == [0, 1]
    d = dropped.set_index("row_id")
    assert d.loc[2, "reason"] == "exact_duplicate" and d.loc[2, "kept_row_id"] == 0
    assert d.loc[3, "reason"] == d.loc[4, "reason"] == "conflicting_labels"
    assert pd.isna(d.loc[3, "kept_row_id"])
    assert len(dropped) == 3


def test_find_near_duplicates():
    base = " ".join(f"kelime{i}" for i in range(60))
    changed = base.replace("kelime30", "degisti")
    other = " ".join(f"farkli{i}" for i in range(60))
    clean = _df([(0, base), (0, changed), (1, other)])
    split = {"train_ids": [0, 2], "test_ids": [1]}
    near = find_near_duplicates(clean, split)
    assert len(near) == 1
    r = near.iloc[0]
    assert (r.row_id_a, r.row_id_b) == (0, 1) and r.same_label and r.cross_split


def test_make_split(tmp_path):
    rows = [(k, f"metin {k} {i}") for k in range(3) for i in range(50)]
    clean = _df(rows)
    p = tmp_path / "s.json"
    s = make_split(clean, path=p, n_raw=150)
    assert not set(s["train_ids"]) & set(s["test_ids"])
    assert sorted(s["train_ids"] + s["test_ids"]) == clean["row_id"].tolist()
    test_counts = clean[clean["row_id"].isin(s["test_ids"])]["label"].value_counts()
    assert all(abs(c - 10) <= 1 for c in test_counts)
    with pytest.raises(FileExistsError):
        make_split(clean, path=p)
    assert load_split(p)["train_ids"] == s["train_ids"]


@pytest.mark.skipif(not SPLIT_V1_PATH.exists(), reason="split_v1.json yok")
def test_real_split():
    s = load_split(SPLIT_V1_PATH)
    assert not set(s["train_ids"]) & set(s["test_ids"])
    assert s["revision"] == DATASET_REVISION
    ratio = len(s["test_ids"]) / s["n_clean"]
    assert abs(ratio - data.TEST_SIZE) < 0.01


# ----------------------------------------------------------------------- v2 / gruplar
def _pairs(*edges):
    return pd.DataFrame(
        [{"row_id_a": a, "row_id_b": b, "similarity": 0.95} for a, b in edges],
        columns=["row_id_a", "row_id_b", "similarity"],
    )


def test_near_dup_groups_chain():
    clean = _df([(0, "a")] * 5)
    g = near_dup_groups(clean, 0.9, pairs=_pairs((0, 1), (1, 2)))
    assert g[0] == g[1] == g[2]  # zincir A-B, B-C tek grup
    assert len({g[3], g[4], g[0]}) == 3  # D ve E tekil
    # eşiğin altındaki kenar yok sayılır
    low = pd.concat(
        [_pairs((0, 1)), pd.DataFrame([{"row_id_a": 1, "row_id_b": 2, "similarity": 0.5}])]
    )
    g2 = near_dup_groups(clean, 0.9, pairs=low)
    assert g2[0] == g2[1] != g2[2]


def _synthetic(n_per_class=60, n_classes=3, group_size=3):
    clean = _df([(k, f"metin {k} {i}") for k in range(n_classes) for i in range(n_per_class)])
    # her sınıfta ardışık group_size satır tek grup
    groups = np.array([i // group_size for i in range(len(clean))])
    return clean, groups


def test_make_split_v2_keeps_groups(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    s = make_split_v2(clean, groups, path=p, n_raw=len(clean))
    tr, te = set(s["train_ids"]), set(s["test_ids"])
    assert not tr & te and sorted(tr | te) == clean["row_id"].tolist()
    assert all(not (set(c) & tr and set(c) & te) for c in s["components"])
    assert all(len(c) >= 2 for c in s["components"]) and s["n_groups"] == len(set(groups))
    with pytest.raises(FileExistsError):
        make_split_v2(clean, groups, path=p)


def test_make_cv_does_not_split_groups(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    s = make_split_v2(clean, groups, path=p, n_raw=len(clean))
    train = clean[clean["row_id"].isin(s["train_ids"])].reset_index(drop=True)
    cv, g = make_cv(train, path=p)
    assert len(g) == len(train)
    for tr_idx, va_idx in cv.split(train, train["label"], g):
        assert not set(g[tr_idx]) & set(g[va_idx])


def test_cv_groups_rejects_foreign_rows(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    make_split_v2(clean, groups, path=p, n_raw=len(clean))
    with pytest.raises(ValueError, match="train'inde değil"):
        make_cv(clean, path=p)  # test satırları da içeriyor


def test_cv_groups_v1_error(tmp_path):
    clean, _ = _synthetic()
    p = tmp_path / "s1.json"
    make_split(clean, path=p, n_raw=len(clean))
    with pytest.raises(ValueError, match="components"):
        data.cv_groups(clean, path=p)


_NEAR_CSV = DEDUP_DIR / "near_duplicates.csv"
_real = pytest.mark.skipif(
    not (SPLIT_V1_PATH.exists() and SPLIT_V2_PATH.exists() and _NEAR_CSV.exists()),
    reason="gerçek split/near_duplicates dosyaları yok",
)


@_real
def test_real_v2():
    s1, s2 = load_split(SPLIT_V1_PATH), load_split(SPLIT_V2_PATH)
    near = pd.read_csv(_NEAR_CSV)
    assert s2["threshold"] == data.NEAR_DUP_THRESHOLD
    assert near["similarity"].min() > s2["threshold"]  # csv = gruplamadaki tüm kenarlar
    assert not near["cross_split_v2"].any()
    tr, te = set(s2["train_ids"]), set(s2["test_ids"])
    assert not tr & te
    assert all(not (set(c) & tr and set(c) & te) for c in s2["components"])
    n = s2["n_clean"]
    assert 0.18 <= len(te) / n <= 0.22
    names = s2["label_names"]
    assert s1["revision"] == s2["revision"]
    assert sorted(s1["train_ids"] + s1["test_ids"]) == sorted(s2["train_ids"] + s2["test_ids"])
    assert len(names) == 7


@_real
def test_real_v2_class_balance():
    from ttc4900.data import drop_exact_duplicates, load_raw

    s2 = load_split(SPLIT_V2_PATH)
    clean, _ = drop_exact_duplicates(load_raw())
    overall = len(s2["test_ids"]) / len(clean)
    test = clean[clean["row_id"].isin(s2["test_ids"])]
    for cat, grp in clean.groupby("category"):
        ratio = test["category"].eq(cat).sum() / len(grp)
        assert abs(ratio - overall) <= 0.03, cat

"""Ağ gerektirmeyen sentetik veri testleri (+ varsa gerçek split dosyası kontrolü)."""

import pandas as pd
import pytest

from ttc4900 import data
from ttc4900.data import (
    DATASET_REVISION,
    SPLIT_PATH,
    drop_exact_duplicates,
    find_near_duplicates,
    load_split,
    make_split,
    normalize_text,
)


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


@pytest.mark.skipif(not SPLIT_PATH.exists(), reason="split_v1.json yok")
def test_real_split():
    s = load_split()
    assert not set(s["train_ids"]) & set(s["test_ids"])
    assert s["revision"] == DATASET_REVISION
    ratio = len(s["test_ids"]) / s["n_clean"]
    assert abs(ratio - data.TEST_SIZE) < 0.01

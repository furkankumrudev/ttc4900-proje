"""Sentetik veri testleri + varsa gerçek split dosyası kontrolleri.

Hepsi ağsız çalışır; yalnızca test_real_v2_class_balance veriyi (HF önbelleği ya da ağ) ister
ve yüklenemezse atlanır.
"""

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


def _assert_groups_match_components(train: pd.DataFrame, g: np.ndarray, components) -> None:
    """g, train satırlarına hizalı ve split'teki bileşenlerle birebir aynı bölümlemeyi vermeli.

    Aynı bileşen -> aynı g; farklı bileşen ya da tekil satır -> farklı g. (Ayrık fold
    kontrolü bunu sınamaz: StratifiedGroupKFold verilen HER g için grupları ayırır.)
    """
    assert len(g) == len(train)
    comp_of = {i: k for k, c in enumerate(components) for i in c}
    # bileşen anahtarı: bileşen indeksi ya da tekil satır için ("tekil", row_id)
    key = [comp_of.get(int(r), ("tekil", int(r))) for r in train["row_id"]]
    pairs = set(zip(key, g.tolist(), strict=True))
    assert len(pairs) == len(set(key)) == len(set(g.tolist()))  # key <-> g birebir


def test_make_cv_groups_aligned_with_components(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    s = make_split_v2(clean, groups, path=p, n_raw=len(clean))
    train = clean[clean["row_id"].isin(s["train_ids"])]
    # karıştırılmış sıra: g satır sırasına hizalı olmalı, row_id sırasına değil
    train = train.sample(frac=1, random_state=0).reset_index(drop=True)
    cv, g = make_cv(train, path=p)
    _assert_groups_match_components(train, g, s["components"])
    for tr_idx, va_idx in cv.split(train, train["label"], g):
        assert not set(g[tr_idx]) & set(g[va_idx])


def test_make_cv_random_state(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    s = make_split_v2(clean, groups, path=p, n_raw=len(clean))
    train = clean[clean["row_id"].isin(s["train_ids"])].reset_index(drop=True)
    cv_a, g = make_cv(train, path=p)
    cv_b, _ = make_cv(train, path=p, random_state=data.RANDOM_STATE)
    cv_c, _ = make_cv(train, path=p, random_state=7)

    def folds(cv):
        return [va.tolist() for _, va in cv.split(train, train["label"], g)]

    assert folds(cv_a) == folds(cv_b)  # varsayılan = RANDOM_STATE, tekrarlanabilir
    assert folds(cv_a) != folds(cv_c)  # tekrarlı CV için farklı bölümleme


def test_cv_groups_rejects_foreign_rows(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    make_split_v2(clean, groups, path=p, n_raw=len(clean))
    with pytest.raises(ValueError, match="train'inde değil"):
        make_cv(clean, path=p)  # test satırları da içeriyor


def test_test_groups_aligned_and_rejects_train_rows(tmp_path):
    clean, groups = _synthetic()
    p = tmp_path / "s2.json"
    s = make_split_v2(clean, groups, path=p, n_raw=len(clean))
    test = clean[clean["row_id"].isin(s["test_ids"])].sample(frac=1, random_state=0)
    test = test.reset_index(drop=True)
    g = data.test_groups(test, path=p)
    _assert_groups_match_components(test, g, s["components"])
    with pytest.raises(ValueError, match="test'inde değil"):
        data.test_groups(clean, path=p)  # train satırları da içeriyor


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
def test_real_v2_csv_edges_within_components():
    """csv'deki her kenarın iki ucu split_v2'de aynı bileşende (csv <-> split bağımsız bağ)."""
    s2 = load_split(SPLIT_V2_PATH)
    near = pd.read_csv(_NEAR_CSV)
    comp_of = {i: k for k, c in enumerate(s2["components"]) for i in c}
    a = near["row_id_a"].map(comp_of)
    b = near["row_id_b"].map(comp_of)
    assert a.notna().all() and (a == b).all()
    # tersi: her bileşen en az bir kenarla bağlı (bileşen = csv kenarlarının bağlı bileşeni)
    assert set(a.astype(int)) == set(range(len(s2["components"])))


@_real
def test_real_make_cv_groups():
    """Gerçek split_v2 train'i üzerinde g <-> bileşen eşleşmesi (yalnızca json, ağ yok)."""
    s2 = load_split(SPLIT_V2_PATH)
    train = pd.DataFrame({"row_id": s2["train_ids"]}).sample(frac=1, random_state=0)
    _, g = make_cv(train.reset_index(drop=True))
    train_set = set(s2["train_ids"])
    comps = [c for c in s2["components"] if set(c) <= train_set]
    _assert_groups_match_components(train.reset_index(drop=True), g, comps)


@_real
def test_real_test_groups():
    """Gerçek split_v2 test'i: yalnızca row_id'ler (metin/etiket yok)."""
    s2 = load_split(SPLIT_V2_PATH)
    test = pd.DataFrame({"row_id": s2["test_ids"]})
    g = data.test_groups(test)
    test_set = set(s2["test_ids"])
    comps = [c for c in s2["components"] if set(c) <= test_set]
    _assert_groups_match_components(test, g, comps)


@_real
def test_real_v2_class_balance():
    """Etiketler split dosyasında yok; veri gerekir (HF önbelleği ya da ağ). Yoksa atlanır."""
    from ttc4900.data import drop_exact_duplicates, load_raw

    s2 = load_split(SPLIT_V2_PATH)
    try:
        raw = load_raw()
    except Exception as e:  # çevrimdışı ve önbellek yok
        pytest.skip(f"veri yüklenemedi: {type(e).__name__}")
    clean, _ = drop_exact_duplicates(raw)
    overall = len(s2["test_ids"]) / len(clean)
    test = clean[clean["row_id"].isin(s2["test_ids"])]
    for cat, grp in clean.groupby("category"):
        ratio = test["category"].eq(cat).sum() / len(grp)
        assert abs(ratio - overall) <= 0.03, cat


def _fake_hub(n=12):
    """Ağsız sahte Hub veri kümesi: gerçek `datasets.Dataset`, ClassLabel etiketli."""
    from datasets import ClassLabel, Dataset, Features, Value

    feats = Features({"category": ClassLabel(names=["a", "b"]), "text": Value("string")})
    return Dataset.from_dict(
        {"category": [i % 2 for i in range(n)], "text": [f"metin {i}" for i in range(n)]}, feats
    )


def _fake_split(tmp_path, n=12, test_ids=(1, 4, 7, 10)):
    p = tmp_path / "s.json"
    train_ids = [i for i in range(n) if i not in test_ids]
    split = {"revision": DATASET_REVISION, "n_raw": n, "train_ids": train_ids,
             "test_ids": list(test_ids)}  # fmt: skip
    data._write_json(split, p)
    return p, split


def test_load_train_never_selects_test_rows(tmp_path, monkeypatch):
    """load_train yalnızca train id'lerini Arrow'dan seçer; test satırı DataFrame'e girmez."""
    from datasets import Dataset

    ds = _fake_hub()
    monkeypatch.setattr(data, "_load_hub", lambda: ds)
    selected: list[list[int]] = []
    orig = Dataset.select

    def spy(self, indices, *a, **k):
        selected.append(list(indices))
        return orig(self, indices, *a, **k)

    monkeypatch.setattr(Dataset, "select", spy)
    p, split = _fake_split(tmp_path)
    train = data.load_train(p)
    assert selected == [split["train_ids"]]
    assert train["row_id"].tolist() == split["train_ids"]
    assert not set(train["row_id"]) & set(split["test_ids"])
    assert train["text"].tolist() == [f"metin {i}" for i in split["train_ids"]]
    assert train["category"].tolist() == ["a" if i % 2 == 0 else "b" for i in split["train_ids"]]


def test_load_train_matches_load_train_test(tmp_path, monkeypatch):
    ds = _fake_hub()
    monkeypatch.setattr(data, "_load_hub", lambda: ds)
    p, _ = _fake_split(tmp_path)
    tr, te = data.load_train_test(p)
    pd.testing.assert_frame_equal(data.load_train(p), tr)
    assert len(tr) + len(te) == len(ds)


def test_load_train_validates_split(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_load_hub", lambda: _fake_hub(10))  # n_raw 12 değil
    p, _ = _fake_split(tmp_path)
    with pytest.raises(ValueError, match="n_raw"):
        data.load_train(p)


@_real
def test_real_load_train_equals_train_part():
    """Gerçek veride load_train == load_train_test()[0] (satır sırası dahil; fold eşleşmesi)."""
    try:
        train = data.load_train()
    except Exception as e:  # çevrimdışı ve önbellek yok
        pytest.skip(f"veri yüklenemedi: {type(e).__name__}")
    pd.testing.assert_frame_equal(train, data.load_train_test()[0])
    assert train["row_id"].tolist() == load_split(SPLIT_V2_PATH)["train_ids"]

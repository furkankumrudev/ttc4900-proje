"""Faz 2 baseline'ları: ağsız, küçük oyuncak veri üzerinde."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from ttc4900 import baselines, report
from ttc4900.data import make_split_v2

N_CLASSES = 3


@pytest.fixture
def toy(tmp_path):
    """3 sınıf x 60 satır, ardışık 3 satır bir yakın-tekrar grubu; her metin benzersiz."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(N_CLASSES):
        for i in range(60):
            words = [f"sinif{k}"] * 2 + [f"ortak{j}" for j in rng.integers(0, 30, 6)]
            rows.append({"label": k, "category": f"c{k}", "text": " ".join(words) + f" id{i}x{k}"})
    clean = pd.DataFrame(rows).reset_index(names="row_id")
    groups = np.arange(len(clean)) // 3
    p = tmp_path / "s2.json"
    s = make_split_v2(clean, groups, path=p, n_raw=len(clean))
    train = clean[clean["row_id"].isin(s["train_ids"])].reset_index(drop=True)
    return train, p


def test_specs_are_pipelines_with_tfidf():
    for name, spec in baselines.SPECS.items():
        pipe = spec.make()
        assert isinstance(pipe.named_steps["tfidf"], TfidfVectorizer), name
        assert set(spec.grid) <= set(pipe.get_params()), name  # grid anahtarları geçerli


def test_tfidf_fit_only_on_outer_train(toy, monkeypatch):
    """Her TF-IDF fit'i bir dış fold'un train kısmının alt kümesinde olmalı (sızıntı yok)."""
    train, p = toy
    seen: list[set[str]] = []
    orig = TfidfVectorizer.fit_transform

    def spy(self, raw_documents, y=None):
        seen.append(set(raw_documents))
        return orig(self, raw_documents, y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", spy)
    folds, preds = baselines.nested_cv(
        baselines.SPECS["word_cnb"], train, list(range(N_CLASSES)), p, n_repeats=2,
        outer_splits=3, inner_splits=2, n_jobs=1,
    )  # fmt: skip
    text_of = dict(zip(train["row_id"], train["text"], strict=True))
    val_sets = [{text_of[i] for i in f["val_row_ids"]} for f in folds]
    assert seen
    for fit_set in seen:
        assert any(not (fit_set & v) for v in val_sets)  # bir dış fold'un val'ine hiç dokunmaz
    # her tekrarda her satır tam bir kez doğrulanır
    assert preds["y_pred"].shape == (2, len(train))
    for r in range(2):
        ids = sum((f["val_row_ids"] for f in folds if f["repeat"] == r), [])
        assert sorted(ids) == sorted(train["row_id"])


def test_run_baseline_writes_contract(toy, tmp_path):
    train, p = toy
    out = tmp_path / "metrics"
    path = baselines.run_baseline(
        "word_cnb", train, path=p, out_dir=out, n_repeats=2, outer_splits=3, inner_splits=2
    )
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["split"] == "s2" and m["test"] is None and m["phase"] == 2
    cv = m["cv"]
    assert len(cv["macro_f1_folds"]) == 6 and len(cv["folds"]) == 6
    assert np.isclose(cv["macro_f1_mean"], np.mean(cv["macro_f1_folds"]))
    assert set(cv["per_class_f1_mean"]) == {"c0", "c1", "c2"}
    assert cv["macro_f1_mean"] > 0.9  # sınıf kelimesi belirgin; düşükse pipeline bozuk
    assert set(m["final"]["best_params"]) == set(baselines.SPECS["word_cnb"].grid)
    bias = m["final"]["cv_best_score_optimistic"] - cv["macro_f1_mean"]
    assert np.isclose(cv["selection_bias"], bias)  # aynı eğitim boyutunda (5 fold) karşılaştırma
    npz = np.load(out / "preds" / "cv_word_cnb.npz")
    assert npz["y_pred"].shape == (2, len(train)) and (npz["y_pred"] >= 0).all()


def test_write_metrics_requires_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        report.write_metrics({"experiment": "x"}, "x", tmp_path)
    p = report.write_metrics({"split": "split_v2", "r": (1, 2)}, "x", tmp_path)
    assert json.loads(p.read_text(encoding="utf-8"))["r"] == [1, 2]

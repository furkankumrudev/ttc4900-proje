"""Faz 7 (error_analysis) testleri: yalnız sahte veriyle; gerçek test seti açılmaz."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from ttc4900 import error_analysis as ea
from ttc4900.final_eval import sha256_file

NAMES = ["siyaset", "dunya", "ekonomi", "kultur", "saglik", "spor", "teknoloji"]
K = len(NAMES)


def _world(n: int = 140, seed: int = 0):
    """Test satırları 0..n-1, train satırları 1000+; bir karışık etiketli grup train ile test'e
    yayılır, bir grup yalnız test'te ve etiketleri aynı."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % K
    texts = [f"metin{i} konu{y[i]} " + ("\u0091alıntı\u0092 " if i % 3 == 0 else "") + "son"
             for i in range(n)]  # fmt: skip
    test = pd.DataFrame({"row_id": np.arange(n), "label": y, "text": texts})
    preds = {}
    for j, m in enumerate(ea.MODELS):
        z = rng.normal(0, 1, (n, K))
        z[np.arange(n), y] += 2.0 + 0.3 * j  # çoğunlukla doğru
        scores = softmax(z, axis=1) if m.startswith("char_lr") else z
        pred = (np.log(scores) if m.startswith("char_lr") else z).argmax(1)
        preds[m] = {"row_id": np.arange(n), "y_true": y, "y_pred": pred, "scores": scores}
    components = [[0, 1000], [7, 14, 21]]  # 0 (test) ve 1000 (train) farklı etiketli
    labels = {int(i): int(y[i]) for i in range(n)} | {1000: (int(y[0]) + 1) % K}
    oof = {m: {1000: 2} for m in ea.PRIMARY}
    groups = np.array([0 if i in (0,) else 7 if i in (7, 14, 21) else -(i + 1) for i in range(n)])
    rates = ea.Rates(groups, B=50, seed=42)
    ref = {m: rates.rate(preds[m]["y_pred"] == y, np.ones(n, bool))["ci95"] for m in ea.PRIMARY}
    T = {"berturk_ft": 1.3, "char_lr": 0.9}
    return test, preds, T, components, labels, oof, groups, ref


def _analyze(**kw):
    test, preds, T, comps, labels, oof, groups, ref = _world()
    args = dict(test=test, preds=preds, T=T, components=comps, labels=labels, oof_pred=oof,
                names=NAMES, groups=groups, ref_accuracy_ci=ref, B=50, seed=42) | kw  # fmt: skip
    return test, ea.analyze(**args)


# --------------------------------------------------------------------------- erişim kuralları
def test_module_never_fits_or_predicts():
    src = Path(inspect.getsourcefile(ea)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(getattr(f, "attr", None) or getattr(f, "id", None))
    forbidden = {"fit", "predict", "predict_proba", "decision_function", "fit_transform",
                 "make_frozen", "fit_frozen", "load_bert", "Trainer", "train"}  # fmt: skip
    assert not called & forbidden
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not imported & {"make_frozen", "fit_frozen", "load_bert", "predict", "evaluate"}


def test_verify_preds_stops_on_sha_mismatch(tmp_path):
    p = tmp_path / "test_berturk_ft.npz"
    np.savez(p, a=np.arange(3))
    good = {"files": {"berturk_ft": {"sha256": sha256_file(p)}}}
    assert ea.verify_preds(good, tmp_path, models=("berturk_ft",)) == {"berturk_ft": sha256_file(p)}
    bad = {"files": {"berturk_ft": {"sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="durdu"):
        ea.verify_preds(bad, tmp_path, models=("berturk_ft",))


def test_rules_text():
    R = ea.ERROR_RULES
    assert "n < 20" in R["uncertainty"] and "CI verilmez" in R["uncertainty"]
    assert "p yok" in R["uncertainty"] and "row_id" in R["text"]
    assert ea.HIGH_CONF == 0.90 and ea.HIGH_CONF_SENS == 0.95 and ea.MAX_EXAMPLES == 2
    assert ea.C1_ONLY == "char_lr_c1clean" and ea.PRIMARY == ("berturk_ft", "char_lr")


# --------------------------------------------------------------------------- oranlar
def test_rates_ci_only_when_n_at_least_20():
    g = np.arange(100)
    r = ea.Rates(g, B=50, seed=0)
    k = np.arange(100) % 2 == 0
    small, big = np.arange(100) < 19, np.arange(100) < 20
    assert r.rate(k, small)["ci95"] is None and "CI verilmez" in r.rate(k, small)["ci_note"]
    assert r.rate(k, big)["ci95"] is not None and r.rate(k, big)["n"] == 20
    d = r.diff(k, small, ~small)
    assert d["ci95"] is None and d["diff"] is not None
    assert r.diff(k, big, ~big)["ci95"] is not None


def test_rates_point_equals_plain_ratio():
    g = np.repeat(np.arange(40), 2)
    r = ea.Rates(g, B=20, seed=0)
    k = np.arange(80) % 3 == 0
    m = np.arange(80) < 60
    assert r.rate(k, m)["rate"] == pytest.approx(k[m].mean())


# --------------------------------------------------------------------------- çekirdek
def test_pairs_rank_all_21_and_named():
    y = np.array([2, 2, 6, 0, 1, 1])
    p = np.array([6, 2, 2, 1, 0, 1])
    out = ea.pairs_item(y, p, NAMES)
    assert len(out["pairs"]) == 21 and out["n_errors"] == 4
    top = out["pairs"][0]
    assert {top["a"], top["b"]} in ({"ekonomi", "teknoloji"}, {"siyaset", "dunya"})
    assert [n["pair"] for n in out["named_pairs"]] == [["ekonomi", "teknoloji"],
                                                        ["siyaset", "dunya"]]  # fmt: skip


def test_mixed_groups_detected():
    labels = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    assert ea.mixed_groups([[1, 2], [3, 4], [5, 4]], labels) == [[3, 4]]


def test_analyze_end_to_end_no_text_in_output():
    test, out = _analyze()
    dump = json.dumps(out, ensure_ascii=False)
    for t in test["text"]:
        assert t not in dump
    assert "alıntı" not in dump and "\u0091" not in dump
    assert set(out) == {"confusion", "pairs", "disagreement", "high_conf_errors",
                        "mixed_label_groups", "c1", "examples"}  # fmt: skip
    assert out["mixed_label_groups"]["n_groups"] == 1
    sides = {r["side"] for r in out["mixed_label_groups"]["groups"][0]["rows"]}
    assert sides == {"train", "test"}
    for cat, ids in out["examples"].items():
        assert len(ids) <= ea.MAX_EXAMPLES, cat
    assert set(out["c1"]["by_model"]) == set(ea.MODELS)
    assert "char_lr_c1clean" not in out["confusion"] and "char_lr_c1clean" not in out["pairs"]


def test_examples_deterministic():
    _, a = _analyze()
    _, b = _analyze()
    assert a["examples"] == b["examples"]


def test_analyze_rejects_different_weights():
    with pytest.raises(ValueError, match="ağırlıkları"):
        _analyze(seed=7)


def test_analyze_rejects_misaligned_text():
    test, preds, T, comps, labels, oof, groups, ref = _world()
    test = test.assign(label=np.roll(test["label"].to_numpy(), 1))
    with pytest.raises(ValueError, match="aynı değil"):
        ea.analyze(test, preds, T, comps, labels, oof, NAMES, groups, ref, B=50, seed=42)


def test_excerpt_length_and_c1_marker():
    t = "a" * 300 + "\u0091"
    e = ea.excerpt(t)
    assert len(e) <= ea.EXCERPT_CHARS and e.endswith("…")
    assert ea.excerpt("x\u0091y  z") == "x¤y z"


def test_run_requires_clean_tree(monkeypatch):
    monkeypatch.setattr(ea.report, "git_commit", lambda: "abc1234-dirty")
    with pytest.raises(RuntimeError, match="temiz"):
        ea.run()


# --------------------------------------------------------------------------- güven bantları
def test_confidence_bands_denominator():
    y = np.array([0] * 30 + [1] * 30)
    pred = {"berturk_ft": y.copy(), "char_lr": y.copy()}
    pred["berturk_ft"][:5] = 1  # 5 hata, hepsi yüksek güvenli
    conf = {"berturk_ft": np.r_[np.full(40, 0.97), np.full(20, 0.5)],
            "char_lr": np.full(60, 0.6)}  # fmt: skip
    rates = ea.Rates(np.arange(60), B=30, seed=0)
    out = ea.confidence_bands(y, pred, conf, rates)
    b = out["berturk_ft"]["conf_ge_0.90"]
    assert b["share_of_predictions"]["k"] == 40 and b["share_of_predictions"]["n"] == 60
    assert b["accuracy_in_band"]["k"] == 35 and b["accuracy_in_band"]["n"] == 40
    assert b["accuracy_below_band"]["n"] == 20 and b["accuracy_below_band"]["ci95"] is not None
    c = out["char_lr"]["conf_ge_0.95"]
    assert c["share_of_predictions"]["k"] == 0 and c["accuracy_in_band"]["n"] == 0
    assert c["accuracy_in_band"]["ci95"] is None and c["mean_conf_in_band"] is None


def test_change_log_and_bands_run_gate(monkeypatch):
    assert "sonradan, betimsel" in ea.ERROR_RULES["change_log"]
    assert "kategori başına en çok 1" in ea.ERROR_RULES["change_log"]
    monkeypatch.setattr(ea.report, "git_commit", lambda: "abc1234-dirty")
    with pytest.raises(RuntimeError, match="temiz"):
        ea.bands_run()

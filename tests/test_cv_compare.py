import numpy as np
import pytest
from scipy import stats

from ttc4900.cv_compare import (
    c1_sensitivity,
    check_same_folds,
    grid_edges,
    holm,
    latest_metrics,
    nadeau_bengio,
    nb_mean_ci,
    summarize,
)


def test_holm_hand_computed():
    # sıralı: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotonluk: 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == [1.0, 1.0]  # 0.8*2=1.6 -> 1'e kırpılır, monotonluk


def test_nadeau_bengio_reduces_to_ttest_and_widens():
    d = np.array([0.01, 0.02, -0.005, 0.015, 0.008, 0.012])
    naive = nadeau_bengio(d, val_train_ratio=0.0)
    ref = stats.ttest_1samp(d, 0.0)
    assert np.isclose(naive["p"], ref.pvalue) and np.isclose(naive["t"], ref.statistic)
    nb = nadeau_bengio(d, val_train_ratio=0.25)
    assert nb["se_corrected"] > naive["se_corrected"] and nb["p"] > naive["p"]
    lo, hi = nb["ci95"]
    assert lo < d.mean() < hi


def test_nadeau_bengio_hand_value_ratio_quarter():
    d = np.array([0.01, 0.02, -0.005, 0.015, 0.008, 0.012])
    J, s = len(d), d.std(ddof=1)
    se = np.sqrt(1 / J + 0.25) * s
    r = nadeau_bengio(d, 0.25)
    assert np.isclose(r["se_corrected"], se) and np.isclose(r["t"], d.mean() / se)
    assert np.isclose(r["p"], 2 * stats.t.sf(d.mean() / se, J - 1))
    lo, hi = nb_mean_ci(d, 0.25)
    assert np.isclose(hi - lo, 2 * stats.t.ppf(0.975, J - 1) * se)


def _fake(name, folds_f1, rows, fold_id, tmp_path):
    np.savez(tmp_path / f"{name}.npz", row_id=rows, fold_id=fold_id)
    f1 = np.array(folds_f1)
    return {
        "split": "split_v2",
        "git_commit": "abc",
        "_file": f"{name}.json",
        "preds_file": f"{name}.npz",
        "cv": {
            "macro_f1_folds": f1.tolist(),
            "macro_f1_mean": f1.mean(),
            "macro_f1_std": f1.std(ddof=1),
            "macro_f1_repeat_means": [f1.mean()],
            "accuracy_mean": f1.mean(),
            "selection_bias": 0.001,
            "folds": [{"n_val": 25, "n_train": 100, "best_params": {"clf__C": 1.0}}] * len(f1),
        },
        "params": {"grid": {"clf__C": [1.0, 10.0]}},
        "final": {"best_params": {"clf__C": 10.0}},
    }


def test_summarize_and_check_same_folds(tmp_path):
    rows = np.arange(10)
    fid = np.array([[0, 1, 2, 3, 4] * 2])
    base = [0.90, 0.91, 0.89, 0.92, 0.90]
    metrics = {
        "char_lr": _fake("char_lr", base, rows, fid, tmp_path),
        "word_lr": _fake("word_lr", [x - 0.01 for x in base], rows[::-1], fid[:, ::-1], tmp_path),
        "word_cnb": _fake("word_cnb", [0.80, 0.95, 0.70, 0.99, 0.85], rows, fid, tmp_path),
    }
    check_same_folds(metrics, tmp_path)  # satır sırası farklı ama atama aynı
    s = summarize(metrics)
    assert s["selected_for_test"] == "char_lr" and s["split"] == "split_v2"
    assert np.isclose(s["val_train_ratio"], 0.25)
    by_b = {c["b"]: c for c in s["paired_vs_best"]}
    # sabit fark: varyans 0, t sonsuz, CI noktaya çöker ve 0'ı içermez
    assert (
        by_b["word_lr"]["std_diff"] < 1e-12 and "word_lr" not in s["not_distinguishable_from_best"]
    )
    assert "word_cnb" in s["not_distinguishable_from_best"]  # gürültülü, CI 0'ı içerir
    assert "selection_rule" in s and s["test"] is None
    # p tablosu: ham / N-B / N-B+Holm, paired_vs_best ile aynı sayılar
    for r in s["p_table"]:
        c = by_b[r["b"]]
        assert (r["p_naive"], r["p_nb"], r["p_nb_holm"]) == (c["p_naive"], c["p"], c["p_holm"])
        assert r["p_naive"] <= r["p_nb"] <= r["p_nb_holm"]
    e = s["grid_edges"]["char_lr"]["clf__C"]
    assert e["final_edge"] == "high" and e["n_folds_low_edge"] == 5

    metrics["word_cnb"] = _fake("word_cnb", base, rows, fid[:, [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]],
                                tmp_path)  # fmt: skip
    with pytest.raises(ValueError, match="farklı dış fold"):
        check_same_folds(metrics, tmp_path)
    metrics["word_cnb"] = _fake("word_cnb", base, rows, fid, tmp_path) | {"split": "split_v1"}
    with pytest.raises(ValueError, match="farklı split"):
        summarize(metrics)


def test_p_naive_equals_paired_ttest():
    d = np.array([0.01, 0.02, -0.005, 0.015, 0.008, 0.012])
    r = nadeau_bengio(d, 0.25)
    assert np.isclose(r["p_naive"], stats.ttest_1samp(d, 0.0).pvalue)
    assert r["p_naive"] < r["p"]  # düzeltme p'yi büyütür


def _m(grid, fold_choices, final):
    return {
        "params": {"grid": grid},
        "cv": {"folds": [{"best_params": c} for c in fold_choices]},
        "final": {"best_params": final},
    }


def test_grid_edges_marks_edges():
    grid = {"clf__C": [10.0, 100.0, 1000.0], "ng": [[1, 3], [2, 4]]}
    folds = [{"clf__C": 100.0, "ng": [2, 4]}] * 3 + [{"clf__C": 1000.0, "ng": [1, 3]}]
    e = grid_edges({"m": _m(grid, folds, {"clf__C": 100.0, "ng": [2, 4]})})["m"]
    assert e["clf__C"]["final_edge"] is None  # iç nokta
    assert e["clf__C"]["n_folds_high_edge"] == 1 and e["clf__C"]["n_folds_low_edge"] == 0
    assert e["clf__C"]["fold_counts"] == {"10.0": 0, "100.0": 3, "1000.0": 1}
    assert e["ng"]["final_edge"] == "high" and e["ng"]["grid_size"] == 2
    assert e["ng"]["n_folds_high_edge"] == 3 and e["ng"]["n_folds_low_edge"] == 1


def _write(m, name, ts, tmp_path):
    import json

    body = {k: v for k, v in m.items() if k != "_file"}
    body["cv"] = body["cv"] | {"per_class_f1_mean": {"a": body["cv"]["macro_f1_mean"], "b": 0.5}}
    body["preds_file"] = f"{name}.npz"
    (tmp_path / f"phase2_{name}_{ts}.json").write_text(json.dumps(body), encoding="utf-8")


def test_latest_metrics_exact_name(tmp_path):
    """char_lr aranırken char_lr_c1clean dosyası seçilmemeli (glob tuzağı)."""
    rows, fid = np.arange(10), np.array([[0, 1, 2, 3, 4] * 2])
    _write(_fake("char_lr", [0.9] * 5, rows, fid, tmp_path), "char_lr", "20260101-000000", tmp_path)
    _write(_fake("char_lr_c1clean", [0.8] * 5, rows, fid, tmp_path), "char_lr_c1clean",
           "20260102-000000", tmp_path)  # fmt: skip
    assert latest_metrics("char_lr", tmp_path)["_file"] == "phase2_char_lr_20260101-000000.json"
    assert latest_metrics("char_lr_c1clean", tmp_path)["cv"]["macro_f1_mean"] == 0.8
    assert latest_metrics("word_lr", tmp_path) is None


def test_c1_sensitivity_single_nb_comparison(tmp_path):
    rows, fid = np.arange(10), np.array([[0, 1, 2, 3, 4] * 2])
    raw = [0.90, 0.91, 0.89, 0.92, 0.90]
    clean = [0.89, 0.91, 0.885, 0.90, 0.895]
    _write(_fake("char_lr", raw, rows, fid, tmp_path), "char_lr", "20260101-000000", tmp_path)
    _write(_fake("char_lr_c1clean", clean, rows, fid, tmp_path), "char_lr_c1clean",
           "20260101-000001", tmp_path)  # fmt: skip
    out = c1_sensitivity(tmp_path)
    ref = nadeau_bengio(np.array(raw) - np.array(clean), 0.25)
    assert np.isclose(out["nadeau_bengio"]["p"], ref["p"])
    assert np.isclose(out["nadeau_bengio"]["mean_diff"], np.mean(raw) - np.mean(clean))
    assert out["split"] == "split_v2" and out["test"] is None
    assert "p_holm" not in out["nadeau_bengio"]  # tek karşılaştırma
    assert "SEÇİM" in out["rule"].upper() or "seçim" in out["rule"]

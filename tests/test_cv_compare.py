import numpy as np
import pytest
from scipy import stats

from ttc4900.cv_compare import (
    check_same_folds,
    holm,
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
            "folds": [{"n_val": 25, "n_train": 100}] * len(f1),
        },
        "final": {"best_params": {}},
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

    metrics["word_cnb"] = _fake("word_cnb", base, rows, fid[:, [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]],
                                tmp_path)  # fmt: skip
    with pytest.raises(ValueError, match="farklı dış fold"):
        check_same_folds(metrics, tmp_path)
    metrics["word_cnb"] = _fake("word_cnb", base, rows, fid, tmp_path) | {"split": "split_v1"}
    with pytest.raises(ValueError, match="farklı split"):
        summarize(metrics)

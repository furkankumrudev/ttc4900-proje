"""Faz 6 (calibration) testleri: yalnız sahte veriyle; gerçek OOF ve test dosyaları okunmaz."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from ttc4900 import calibration as cal
from ttc4900.final_eval import sha256_file

K = 7


def _synthetic(n: int, T_true: float, seed: int = 0):
    """Etiketler softmax(z0)'dan çekilir; model logit'i z = T_true · z0 (aşırı emin)."""
    rng = np.random.default_rng(seed)
    z0 = rng.normal(0, 1.5, (n, K))
    y = np.array([rng.choice(K, p=p) for p in softmax(z0, axis=1)])
    return z0 * T_true, y


# --------------------------------------------------------------------------- temel
def test_fit_temperature_recovers_known_T():
    z, y = _synthetic(20000, 2.5)
    r = cal.fit_temperature(z, y)
    assert abs(r["T"] - 2.5) < 0.1 and not r["at_bound"]


def test_fit_temperature_flags_bound():
    # her zaman doğru, küçük marj: NLL 1/T ile azalmayı sürdürür (sayısal olarak düzleşmez)
    z = np.tile(np.array([[0.2] + [0.0] * (K - 1)]), (50, 1))
    r = cal.fit_temperature(z, np.zeros(50, dtype=int))
    assert r["at_bound"] and r["T"] == pytest.approx(cal.T_BOUNDS[0], rel=1e-2)


def test_argmax_preserved_scalar_and_per_row_T():
    z, _ = _synthetic(500, 1.0)
    for T in (0.3, 1.0, 4.0, np.linspace(0.2, 5, 500)):
        assert np.array_equal(cal.log_probs(z, T).argmax(1), z.argmax(1))


def test_char_lr_log_proba_equivalent_to_decision_function():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 3, (100, K))
    z, clipped = cal.test_logits("char_lr", softmax(d, axis=1))
    for T in (0.5, 2.0):
        assert np.allclose(cal.log_probs(z, T), cal.log_probs(d, T))
    assert clipped == 0
    _, clipped = cal.test_logits("char_lr", np.array([[1.0] + [0.0] * (K - 1)]))
    assert clipped == K - 1


def test_cross_fit_never_uses_held_out_fold():
    z, y = _synthetic(3000, 2.0)
    fold = np.arange(3000) % 5
    t1, per = cal.cross_fit(z, y, fold)
    z2 = z.copy()
    z2[fold == 0] *= 10  # yalnız fold 0 bozulur: fold 0'ın kendi T'si değişmemeli
    t2, _ = cal.cross_fit(z2, y, fold)
    assert np.allclose(t1[fold == 0], t2[fold == 0])
    assert not np.allclose(t1[fold == 1], t2[fold == 1])
    assert [p["fold"] for p in per] == list(range(5))


def test_ece_hand_example():
    # conf: 0.9 (doğru), 0.9 (yanlış), 0.6 (doğru), 0.6 (doğru); 15 bin: 0.9 -> bin 13, 0.6 -> bin 8
    p = np.full((4, K), 0.0)
    for i, (c, cls) in enumerate([(0.9, 0), (0.9, 0), (0.6, 1), (0.6, 1)]):
        p[i, cls] = c
        p[i, [k for k in range(K) if k != cls]] = (1 - c) / (K - 1)
    y = np.array([0, 3, 1, 1])
    lp = np.log(p)
    point, _ = cal.bootstrap_metrics({"m": lp}, y, np.arange(4), B=10, seed=0)
    # bin 0.9: acc 0.5, conf 0.9 -> |0.4| · 2/4; bin 0.6: acc 1, conf 0.6 -> |0.4| · 2/4
    assert point["m"]["ece15"] == pytest.approx(0.4)
    assert point["m"]["acc"] == pytest.approx(0.75)
    assert point["m"]["nll"] == pytest.approx(-np.mean(lp[np.arange(4), y]))
    brier = ((p - np.eye(K)[y]) ** 2).sum(1).mean()
    assert point["m"]["brier"] == pytest.approx(brier)


def test_bin_edges_right_closed():
    m = 15
    assert cal._bin_index(np.array([1 / m, 1 / m + 1e-9, 1.0]), m).tolist() == [0, 1, m - 1]


def test_delta_sign_positive_means_improvement():
    z, y = _synthetic(4000, 3.0)
    T = cal.fit_temperature(z, y)["T"]
    lps = {"x:before": cal.log_probs(z), "x:after": cal.log_probs(z, T)}
    point, reps = cal.bootstrap_metrics(lps, y, np.arange(len(y)), B=200, seed=0)
    r = cal.before_after(point, reps, "x")
    assert r["nll"]["delta"] == pytest.approx(r["nll"]["before"] - r["nll"]["after"])
    assert r["nll"]["delta"] > 0 and r["nll"]["delta_ci95"][0] > 0


def test_decide():
    assert cal.decide([0.01, 0.02], False, "train") == "gösterildi"
    assert cal.decide([-0.01, 0.02], False, "train") == "gösterilemedi"
    assert cal.decide([0.0, 0.02], False, "train") == "gösterilemedi"  # tamamen üstünde değil
    assert cal.decide([0.01, 0.02], False, "test") == "teyit edildi"
    assert cal.decide([-0.03, -0.01], False, "test") == "teyit edilemedi"
    assert cal.decide([0.01, 0.02], True, "train") == "değerlendirilemez"


def test_same_weights_across_inputs():
    z, y = _synthetic(300, 1.5)
    g = np.arange(300) // 3
    lps = {"a": cal.log_probs(z), "b": cal.log_probs(z)}
    _, reps = cal.bootstrap_metrics(lps, y, g, B=100, seed=42)
    assert np.array_equal(reps["a"]["nll"], reps["b"]["nll"])


# --------------------------------------------------------------------------- kurallar
def test_rules_state_definitions_and_scope():
    R = cal.CALIB_RULES
    assert "ΔNLL = NLL(önce, T = 1) - NLL(sonra); pozitif = iyileşme" in R["metrics"]
    assert "tamamen 0'ın üstünde" in R["train_eval"] and "tamamen 0'ın üstünde" in R["test_eval"]
    assert "Faz 5 ile aynı grup bootstrap ağırlıkları" in R["test_eval"]
    assert "'X, Y'den daha iyi kalibre' iddiası yok" in R["between_models"]
    assert cal.PRIMARY == ("berturk_ft", "char_lr")
    assert "word_cnb" not in cal.CALIB_MODELS
    assert cal.ECE_BINS == 15 and cal.ECE_BINS_SENS == (10, 20)


def test_train_path_never_touches_test():
    src = Path(inspect.getsourcefile(cal)).read_text(encoding="utf-8")
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    assert "load_train_test" not in names and "load_train_test" not in src
    forbidden = {"test_groups", "preds_path", "test_logits", "calibrate_test"}
    for fn in (cal.calibrate_train, cal.calibrate_oof, cal.bert_oof, cal.char_lr_oof,
               cal.per_run_table, cal.t_bootstrap, cal.fold_breakdown,
               cal.fold_breakdown_train):  # fmt: skip
        tree = ast.parse(inspect.getsource(fn).lstrip())
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str)]  # fmt: skip
        assert not used & forbidden, fn.__name__
        assert not any("test_" in s for s in strings), fn.__name__


# --------------------------------------------------------------------------- 6b çekirdek
def _oofs(n: int = 700):
    out = {}
    row_id = np.arange(n)
    for i, name in enumerate(cal.CALIB_MODELS):
        z, y = _synthetic(n, 1.5 + 0.5 * i, seed=0)  # aynı etiketler, farklı aşırı güven
        out[name] = {"row_id": row_id, "y": y, "z": z, "fold": row_id % 5}
    return out


def test_calibrate_oof_end_to_end():
    oofs = _oofs()
    g = np.arange(700) // 2
    core = cal.calibrate_oof(oofs, g, B=30, seed=0)
    for n, role in cal.CALIB_MODELS.items():
        m = core["models"][n]
        assert m["role"] == role and len(m["cross_fit_folds"]) == 5
        assert set(m["metrics_cross_fit"]) == set(cal.METRICS)
        assert (m["verdict"] is None) == (role == "betimsel")
    assert all("p" not in r for r in core["between_models"])
    assert core["between_models"][-1]["metric"] == "T̂_final"
    assert len(core["sentences"]) == len(cal.CALIB_MODELS)


def test_calibrate_oof_rejects_misaligned_rows():
    oofs = _oofs()
    oofs["char_lr"]["y"] = np.roll(oofs["char_lr"]["y"], 1)
    with pytest.raises(ValueError, match="uyuşmuyor"):
        cal.calibrate_oof(oofs, np.arange(700), B=5)


def test_per_run_table_shapes():
    oofs = _oofs()
    runs = {n: [{"fold": k, "best_epoch": k + 1} for k in range(5)] for n in cal.BERT_SEEDS}
    t = cal.per_run_table(oofs, runs)
    assert len(t["bert_runs"]) == 15 and set(t["bert_by_seed"]) == {42, 43, 44}
    assert set(t["bert_by_fold"]) == set(range(5)) and len(t["char_lr_folds"]) == 5
    assert {"T", "best_epoch", "n"} <= set(t["bert_runs"][0])


def test_calibrate_train_requires_clean_tree(monkeypatch):
    monkeypatch.setattr(cal.report, "git_commit", lambda: "abc1234-dirty")
    with pytest.raises(RuntimeError, match="temiz"):
        cal.calibrate_train()


# --------------------------------------------------------------------------- 6c kapıları
@pytest.fixture
def test_world(tmp_path, monkeypatch):
    monkeypatch.setattr(cal.report, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(cal, "_untracked_inputs", lambda: [])
    monkeypatch.setattr(cal, "_is_tracked", lambda p: True)
    n = 60
    split = tmp_path / "split_v2.json"
    split.write_text(json.dumps({"train_ids": list(range(100, 110)), "test_ids": list(range(n)),
                                 "components": [[0, 1], [5, 6, 7]]}), encoding="utf-8")  # fmt: skip
    md, preds = tmp_path / "metrics", tmp_path / "metrics" / "preds"
    preds.mkdir(parents=True)
    z0, y = _synthetic(n, 2.0, seed=3)
    g = cal.test_groups(__import__("pandas").DataFrame({"row_id": np.arange(n)}), split)
    main = {"split": "split_v2", "preds": {}, "models": {}}
    for name in cal.CALIB_MODELS:
        scores = softmax(z0, axis=1) if name == "char_lr" else z0
        p = preds / f"test_{name}.npz"
        np.savez(p, row_id=np.arange(n), y_true=y, y_pred=z0.argmax(1), scores=scores)
        lp = cal.log_probs(cal.test_logits(name, scores)[0])
        _, reps = cal.bootstrap_metrics({"x": lp}, y, g, B=40, seed=42)
        main["preds"][name] = {"sha256": sha256_file(p)}
        main["models"][name] = {"accuracy_ci95": cal._pct_ci(reps["x"]["acc"])}
    (md / "phase5_test_20260101-000000.json").write_text(json.dumps(main), encoding="utf-8")
    frozen_t = {"T_final": 1.7, "T_final_at_bound": False}
    six_b = {"split": "split_v2", "models": {m: frozen_t for m in cal.CALIB_MODELS}}
    kw = dict(metrics_dir=md, preds_dir=preds, out_dir=md, fig_dir=tmp_path / "fig", path=split,
              B=40, seed=42)  # fmt: skip
    return {"md": md, "preds": preds, "kw": kw, "six_b": six_b}


def _write_6b(w):
    p = w["md"] / "phase6_train_calibration_20260102-000000.json"
    p.write_text(json.dumps(w["six_b"]), encoding="utf-8")
    return p


def test_calibrate_test_requires_6b(test_world):
    with pytest.raises(RuntimeError, match="önce 6b"):
        cal.calibrate_test(**test_world["kw"])


def test_calibrate_test_requires_committed_6b(test_world, monkeypatch):
    _write_6b(test_world)
    monkeypatch.setattr(cal, "_is_tracked", lambda p: False)
    with pytest.raises(RuntimeError, match="commit"):
        cal.calibrate_test(**test_world["kw"])


def test_calibrate_test_uses_frozen_T_and_runs_once(test_world):
    _write_6b(test_world)
    _, out = cal.calibrate_test(**test_world["kw"])
    for name, role in cal.CALIB_MODELS.items():
        m = out["models"][name]
        assert m["T_final"] == 1.7
        assert (m["verdict"] is None) == (role == "betimsel")
        assert m["verdict"] in (None, "teyit edildi", "teyit edilemedi")
    assert out["train_calibration"]["file"] == "phase6_train_calibration_20260102-000000.json"
    assert all("p" not in r for r in out["between_models"])
    with pytest.raises(RuntimeError, match="tek sefer"):
        cal.calibrate_test(**test_world["kw"])


def test_calibrate_test_rejects_changed_preds(test_world):
    _write_6b(test_world)
    p = test_world["preds"] / "test_char_lr.npz"
    with np.load(p) as z:
        d = dict(z)
    d["scores"] = d["scores"] * 0.999
    np.savez(p, **d)
    with pytest.raises(ValueError, match="sha256"):
        cal.calibrate_test(**test_world["kw"])


def test_calibrate_test_rejects_different_bootstrap_weights(test_world):
    _write_6b(test_world)
    with pytest.raises(ValueError, match="ağırlıkları"):
        cal.calibrate_test(**(test_world["kw"] | {"seed": 7}))


# --------------------------------------------------------------------------- sonradan döküm
def test_not_confirmed_sentence_lists_possible_causes_and_decision_unchanged():
    s = cal.SENTENCES["teyit edilemedi"]
    assert "olası nedenler: model uyuşmazlığı, düşük güç" in s
    assert "etki olmadığı anlamına gelmez" in s and "T farklı modellerden öğrenildi" not in s
    assert cal.decide([-0.001, 0.02], False, "test") == "teyit edilemedi"
    assert cal.decide([0.001, 0.02], False, "test") == "teyit edildi"
    assert (
        "decide" in cal.CALIB_RULES["change_log"] and "DEĞİŞMEDİ" in cal.CALIB_RULES["change_log"]
    )
    assert "etki olmadığı anlamına gelmez" in cal.CALIB_RULES["interpretation"]


def test_fold_breakdown_matches_6b_totals():
    oofs = _oofs()
    g = np.arange(700) // 2
    core = cal.calibrate_oof(oofs, g, B=20, seed=0)
    runs = {n: [{"fold": k, "best_epoch": 1} for k in range(5)] for n in cal.BERT_SEEDS}
    six_b = {"models": core["models"], "per_run_T": cal.per_run_table(oofs, runs)}
    out = cal.fold_breakdown(oofs, six_b, g, B=20, seed=0)
    for n, m in out.items():
        assert len(m["folds"]) == 5 and "p" not in m["folds"][0]
        assert m["delta_total_6b"] == pytest.approx(
            core["models"][n]["metrics_cross_fit"]["nll"]["delta"]
        )
    six_b["models"]["char_lr"]["cross_fit_folds"][0]["T"] += 0.5
    with pytest.raises(ValueError, match="6b json"):
        cal.fold_breakdown(oofs, six_b, g, B=20, seed=0)

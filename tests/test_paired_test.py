"""paired_test: exact McNemar, grup bootstrap, karar kuralı ve güç (yalnız sentetik veri)."""

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import f1_score

from ttc4900 import cv_compare, paired_test, test_plan
from ttc4900.paired_test import (
    cluster_bootstrap,
    cluster_bootstrap_diff,
    compare_family,
    decide,
    describe_pair,
    macro_f1,
    mcnemar_exact,
    mcnemar_power,
    mde,
    mde_macro_f1,
)


def test_test_plan_single_source():
    assert cv_compare.TEST_PLAN is test_plan.TEST_PLAN
    fam = test_plan.TEST_PLAN["families"]
    assert fam["primary"]["m"] == 1 and fam["secondary"]["m"] == 4
    assert "bge_m3 - char_lr" in fam["secondary"]["comparisons"]


def test_mcnemar_exact_hand_case():
    # 10 satır: A doğru B yanlış (n10), 3 satır: tersi (n01), 20 satır: ikisi de doğru
    a = np.array([1] * 10 + [0] * 3 + [1] * 20, dtype=bool)
    b = np.array([0] * 10 + [1] * 3 + [1] * 20, dtype=bool)
    r = mcnemar_exact(a, b)
    assert (r["n10"], r["n01"], r["n_discordant"]) == (10, 3, 13)
    # elle: p = 2 · P(X <= 3), X ~ Binom(13, 1/2) = 2 · (1 + 13 + 78 + 286) / 8192
    assert r["p"] == pytest.approx(2 * 378 / 8192)
    assert mcnemar_exact(a, a)["p"] == 1.0


def test_macro_f1_matches_sklearn_with_missing_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 200)  # sınıf 6 hiç yok
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 7, 200))
    ref = f1_score(y, p, average="macro", labels=list(range(7)), zero_division=0)
    assert macro_f1(y, p) == pytest.approx(ref)
    p2 = np.where(p == 6, 0, p)  # sınıf 6 ne gerçek ne tahmin
    ref2 = f1_score(y, p2, average="macro", labels=list(range(7)), zero_division=0)
    assert macro_f1(y, p2) == pytest.approx(ref2)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 7, n)
    pa = np.where(rng.random(n) < 0.85, y, rng.integers(0, 7, n))
    pb = np.where(rng.random(n) < 0.75, y, rng.integers(0, 7, n))
    groups = rng.integers(0, n // 2, n)  # ~2'şerli gruplar
    return y, pa, pb, groups


def test_bootstrap_identical_models_zero_and_deterministic():
    y, pa, _, g = _toy()
    r = cluster_bootstrap_diff(y, pa, pa, g, B=500, seed=1)
    assert r["diff"] == 0 and r["diff_ci95"] == [0.0, 0.0]
    y, pa, pb, g = _toy()
    r1 = cluster_bootstrap_diff(y, pa, pb, g, B=500, seed=1)
    r2 = cluster_bootstrap_diff(y, pa, pb, g, B=500, seed=1)
    assert r1 == r2
    assert r1["diff_ci95"][0] <= r1["diff"] <= r1["diff_ci95"][1]


def test_bootstrap_resamples_whole_groups():
    """Tek grup: her replikat o grubu G=1 kez alır -> bütün veri, varyans sıfır."""
    y, pa, pb, _ = _toy(n=50)
    reps = cluster_bootstrap(y, {"a": pa}, np.zeros(50), B=200, seed=0)
    assert np.allclose(reps["a"], macro_f1(y, pa))


def test_bootstrap_weights_independent_of_models():
    """Bir çiftin replikatları aileye başka model eklenince değişmez (aynı w)."""
    y, pa, pb, g = _toy()
    r1 = cluster_bootstrap(y, {"a": pa}, g, B=300, seed=3)
    r2 = cluster_bootstrap(y, {"a": pa, "b": pb}, g, B=300, seed=3)
    assert np.array_equal(r1["a"], r2["a"])


def test_decide_requires_all_conditions():
    d = decide(0.01, [0.01, 0.05], n10=30, n01=10)
    assert d["distinguished"] and d["direction"] == "a > b" and d["verdict"] == "ayırt edildi"
    assert decide(0.01, [-0.04, -0.01], n10=10, n01=30)["direction"] == "a < b"
    d = decide(0.01, [-0.01, 0.05], n10=30, n01=10)
    assert not d["distinguished"] and d["p_holm_below_alpha"] and not d["ci_excludes_zero"]
    d = decide(0.2, [0.01, 0.05], n10=30, n01=10)
    assert not d["distinguished"] and d["ci_excludes_zero"]
    assert not decide(0.05, [0.01, 0.05], n10=30, n01=10)["distinguished"]  # p < α katı


def test_decide_direction_conflict():
    """McNemar A lehine değil (n10 < n01) ama macro-F1 CI > 0: ayırt edilemedi."""
    d = decide(0.01, [0.01, 0.05], n10=10, n01=30)
    assert not d["distinguished"] and not d["directions_consistent"]
    assert d["verdict"] == "ayırt edilemedi (yön çelişkisi)" and d["direction"] is None


def test_compare_family_holm_and_fields():
    y, pa, pb, g = _toy(n=400)
    pc = pb.copy()
    rows = compare_family(y, {"a": pa, "b": pb, "c": pc}, [("a", "b"), ("a", "c")], g, B=300)
    assert [r["p_holm"] for r in rows] == pytest.approx(cv_compare.holm([r["p"] for r in rows]))
    for r in rows:
        assert {"n10", "n01", "diff_ci95", "distinguished", "direction"} <= set(r)


def test_power_size_and_monotone():
    n, psi = 300, 0.1
    assert mcnemar_power(n, psi, 0.0, 0.05) <= 0.05  # exact test: boyut <= α
    pw = [mcnemar_power(n, psi, d, 0.05) for d in (0.0, 0.01, 0.02, 0.04, 0.06)]
    assert all(b >= a for a, b in zip(pw, pw[1:], strict=False))
    assert mde(400, psi) > mde(1600, psi)
    assert mde(908, 0.1, 0.0125) >= mde(908, 0.1, 0.05)


@pytest.mark.parametrize("alpha", [0.05, 0.05 / 4])
def test_power_rejection_region_matches_binomtest(alpha):
    """Red bölgesi c_k, binomtest p < α ile aynı olmalı (n=908'de kullanılan k aralığı dahil)."""
    for k in range(0, 201):
        c = stats.binom.ppf(alpha / 2, k, 0.5) - 1
        # sınırın iki yanı yeterli: min(b, k-b) = c ve c + 1
        for m in {max(int(c), 0), int(c) + 1}:
            if m > k // 2:
                continue
            rej = m <= c
            p = stats.binomtest(m, k, 0.5).pvalue if k > 0 else 1.0
            assert rej == (p < alpha), (k, m)


def test_power_matches_monte_carlo():
    n, psi, delta, alpha = 120, 0.2, 0.08, 0.05
    exact = mcnemar_power(n, psi, delta, alpha)
    rng = np.random.default_rng(0)
    probs = [(psi + delta) / 2, (psi - delta) / 2, 1 - psi]  # n10, n01, uyuşma
    hits = 0
    sims = 4000
    for cnt in rng.multinomial(n, probs, size=sims):
        k = cnt[0] + cnt[1]
        hits += k > 0 and stats.binomtest(int(cnt[0]), int(k), 0.5).pvalue < alpha
    assert abs(hits / sims - exact) < 0.025


def test_mde_macro_f1_multipliers():
    assert mde_macro_f1(1.0, 0.05) == pytest.approx(2.8016, abs=1e-3)
    assert mde_macro_f1(1.0, 0.05 / 4) == pytest.approx(3.3393, abs=1e-4)


def test_power_pairs_cover_families():
    pairs = {(a, b) for _, a, b in paired_test.POWER_PAIRS}
    for f in test_plan.FAMILIES.values():
        assert set(f["comparisons"]) <= pairs
    assert set(test_plan.DESCRIPTIVE) <= pairs


def test_describe_pair_has_no_p():
    y, pa, pb, g = _toy()
    r = describe_pair(y, pa, pb, g, B=200)
    assert "p" not in r and "p_holm" not in r and "diff_ci95" in r

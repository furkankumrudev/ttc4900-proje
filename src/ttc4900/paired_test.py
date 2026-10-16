"""Final test yönteminin kodu (TEST_PLAN) ve test öncesi güç analizi.

Yöntem `ttc4900.test_plan.TEST_PLAN`'dadır; bu modül onu uygular:

    p     : exact binom McNemar. n10 = A doğru & B yanlış, n01 = tersi.
            H0: P(A doğru | uyuşmazlık) = 1/2  =>  n10 ~ Binom(n01 + n10, 1/2), iki yönlü.
    etki  : macro-F1(A) - macro-F1(B), eşleştirilmiş grup (cluster) bootstrap'iyle %95 CI.
    karar : Holm p < 0.05 VE CI 0'ı dışlıyor  =>  "ayırt edildi".

Grup bootstrap'inin hilesi: karışıklık matrisi satırlar üzerinde toplanabilir. Her grubun 7x7
matrisi C_g bir kez hesaplanır. Bir bootstrap replikatı w ~ Multinomial(G, 1/G) ağırlıklarıyla
C* = Σ_g w_g C_g olur. Tüm modeller AYNI w'leri kullanır (eşleştirme). macro-F1 C*'tan hesaplanır.

Güç (yalnızca train OOF'tan): uyuşmazlık oranı psi = P(A ve B'den yalnız biri doğru), accuracy
farkı delta = p10 - p01. n test satırında uyuşmazlık sayısı k ~ Binom(n, psi), uyuşmazlık
içinde n10 ~ Binom(k, q), q = (psi + delta) / (2 psi).
Güç = Σ_k P(k) · P(exact testi reddeder | k, q).

Kullanım: python -m ttc4900.paired_test --power
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ttc4900 import report
from ttc4900.cv_compare import holm, latest_metrics
from ttc4900.data import SPLIT_PATH, cv_groups, load_split, test_groups
from ttc4900.paths import METRICS_DIR
from ttc4900.test_plan import (
    ALPHA,
    BOOTSTRAP_B,
    BOOTSTRAP_CHUNK,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    DESCRIPTIVE,
    FAMILIES,
    POWER_TARGET,
    TEST_PLAN,
)

N_CLASSES = 7


# --------------------------------------------------------------------------- McNemar
def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact binom McNemar (süreklilik düzeltmesi yok). n01 + n10 = 0 ise p = 1."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "n_discordant": n, "p": p}


# --------------------------------------------------------------------------- macro-F1
def confusion(y, pred, k: int = N_CLASSES) -> np.ndarray:
    """k x k karışıklık matrisi (satır = gerçek, sütun = tahmin)."""
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def macro_f1_from_confusion(C: np.ndarray) -> np.ndarray:
    """F1_c = 2 TP / (2 TP + FP + FN); payda 0 ise 0 (sklearn zero_division=0). C: (..., k, k)."""
    C = np.asarray(C, dtype=float)
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    denom = C.sum(axis=-1) + C.sum(axis=-2)  # (TP + FN) + (TP + FP)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(axis=-1)


def macro_f1(y, pred, k: int = N_CLASSES) -> float:
    return float(macro_f1_from_confusion(confusion(y, pred, k)))


# --------------------------------------------------------------------------- bootstrap
def _group_confusions(y, pred, inv: np.ndarray, G: int, k: int) -> np.ndarray:
    """(G, k*k): her grubun düzleştirilmiş karışıklık matrisi."""
    flat = np.zeros((G, k * k), dtype=np.int64)
    cell = np.asarray(y, dtype=np.int64) * k + np.asarray(pred, dtype=np.int64)
    np.add.at(flat, (inv, cell), 1)
    return flat


def cluster_bootstrap(
    y,
    preds: dict[str, np.ndarray],
    groups,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    k: int = N_CLASSES,
    chunk: int = BOOTSTRAP_CHUNK,
) -> dict[str, np.ndarray]:
    """Her model için B bootstrap macro-F1 değeri; tüm modeller aynı grup örneklemlerini görür.

    Ağırlıklar modellerden bağımsız çekilir: aynı (G, B, seed) aynı w dizisini verir, yani bir
    çiftin CI'ı aileye başka model eklenince değişmez.
    """
    inv, G = group_index(groups)
    flats = {m: _group_confusions(y, p, inv, G, k) for m, p in preds.items()}
    out = {m: np.empty(B) for m in preds}
    for start, w in bootstrap_weights(G, B, seed, chunk):
        n = len(w)
        for m, F in flats.items():
            out[m][start : start + n] = macro_f1_from_confusion((w @ F).reshape(n, k, k))
    return out


def group_index(groups) -> tuple[np.ndarray, int]:
    """Satır -> grup sırası (np.unique sıralı id düzeni) ve grup sayısı G."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return inv, int(inv.max()) + 1


def bootstrap_weights(G: int, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED,
                      chunk: int = BOOTSTRAP_CHUNK):  # fmt: skip
    """(başlangıç, w) parçaları; w ~ Multinomial(G, 1/G), (n, G). Tek kaynak: aynı (G, B, seed,
    chunk) her çağrıda aynı diziyi verir (cluster_bootstrap ve Faz 5 accuracy CI'ı paylaşır)."""
    rng = np.random.default_rng(seed)
    for start in range(0, B, chunk):
        yield start, rng.multinomial(G, np.full(G, 1 / G), size=min(chunk, B - start))


def _pct_ci(x: np.ndarray, level: float = CI_LEVEL) -> list[float]:
    lo = (1 - level) / 2 * 100
    return [float(v) for v in np.percentile(x, [lo, 100 - lo])]


def cluster_bootstrap_diff(
    y, pred_a, pred_b, groups, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> dict:
    """macro-F1(A) - macro-F1(B): nokta tahmini, yüzdelik %95 CI, bootstrap SE."""
    reps = cluster_bootstrap(y, {"a": pred_a, "b": pred_b}, groups, B, seed)
    return _effect(y, pred_a, pred_b, reps["a"], reps["b"])


def _effect(y, pred_a, pred_b, ra: np.ndarray, rb: np.ndarray) -> dict:
    d = ra - rb
    fa, fb = macro_f1(y, pred_a), macro_f1(y, pred_b)
    return {
        "macro_f1_a": fa,
        "macro_f1_b": fb,
        "macro_f1_a_ci95": _pct_ci(ra),
        "macro_f1_b_ci95": _pct_ci(rb),
        "diff": fa - fb,
        "diff_ci95": _pct_ci(d),
        "diff_se_boot": float(d.std(ddof=1)),
    }


# --------------------------------------------------------------------------- karar
def decide(p_holm: float, ci: list[float], n10: int, n01: int, alpha: float = ALPHA) -> dict:
    """TEST_PLAN['decision_rule']: Holm p < α, CI 0'ı dışlıyor ve yönler tutarlı.

    McNemar yönü sign(n10 - n01) (accuracy), CI yönü macro-F1; ayrışırsa 'yön çelişkisi'.
    """
    p_ok = p_holm < alpha
    ci_sign = 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)
    mc_sign = int(np.sign(n10 - n01))
    consistent = ci_sign != 0 and ci_sign == mc_sign
    ok = p_ok and ci_sign != 0 and consistent
    if ok:
        verdict = "ayırt edildi"
    elif p_ok and ci_sign != 0:
        verdict = "ayırt edilemedi (yön çelişkisi)"
    else:
        verdict = "ayırt edilemedi"
    return {
        "p_holm_below_alpha": bool(p_ok),
        "ci_excludes_zero": ci_sign != 0,
        "directions_consistent": bool(consistent),
        "distinguished": bool(ok),
        "verdict": verdict,
        "direction": ("a > b" if ci_sign > 0 else "a < b") if ok else None,
    }


def compare_family(
    y,
    preds: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
    groups,
    alpha: float = ALPHA,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict]:
    """Bir Holm ailesi: exact McNemar -> Holm -> grup bootstrap -> karar kuralı."""
    y = np.asarray(y)
    used = sorted({m for pair in comparisons for m in pair})
    reps = cluster_bootstrap(y, {m: np.asarray(preds[m]) for m in used}, groups, B, seed)
    rows = []
    for a, b in comparisons:
        pa, pb = np.asarray(preds[a]), np.asarray(preds[b])
        rows.append(
            {
                "a": a,
                "b": b,
                **mcnemar_exact(pa == y, pb == y),
                **_effect(y, pa, pb, reps[a], reps[b]),
            }
        )
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        r |= decide(ph, r["diff_ci95"], r["n10"], r["n01"], alpha)
    return rows


def describe_pair(
    y, pred_a, pred_b, groups, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Betimsel çift (TEST_PLAN['descriptive']): yalnız skorlar ve fark CI'ı; p ve Holm YOK."""
    return {"inference": "betimsel; p hesaplanmaz"} | cluster_bootstrap_diff(
        y, pred_a, pred_b, groups, B, seed
    )


# --------------------------------------------------------------------------- güç
def mcnemar_power(n: int, psi: float, delta: float, alpha: float = ALPHA) -> float:
    """Exact iki yönlü McNemar'ın gücü (n satır, uyuşmazlık oranı psi, accuracy farkı delta).

    k uyuşmazlıkta red bölgesi: min(n10, n01) <= c_k, c_k = F^{-1}(alpha/2) - 1 (Binom(k, 1/2));
    p = min(1, 2 F(min)) < alpha  <=>  F(min) < alpha/2.
    """
    if not 0 < psi <= 1 or abs(delta) > psi:
        raise ValueError("0 < psi <= 1 ve |delta| <= psi olmalı")
    k = np.arange(n + 1)
    pk = stats.binom.pmf(k, n, psi)
    c = stats.binom.ppf(alpha / 2, k, 0.5) - 1  # -1: o k'da red yok
    q = 0.5 + delta / (2 * psi)
    rej = stats.binom.cdf(c, k, q) + stats.binom.sf(k - c - 1, k, q)
    rej = np.where(c >= 0, rej, 0.0)
    return float(np.sum(pk * rej))


def mde(n: int, psi: float, alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """Gücün `power`a ulaştığı en küçük |delta| (accuracy farkı); ulaşılamazsa nan."""
    if mcnemar_power(n, psi, psi, alpha) < power:
        return float("nan")
    lo, hi = 0.0, psi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mcnemar_power(n, psi, mid, alpha) < power else (lo, mid)
    return hi


def mde_macro_f1(se_test: float, alpha: float, power: float = POWER_TARGET) -> float:
    """Kaba normal yaklaşım: (z_{1-α/2} + z_power) · SE."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se_test)


# --------------------------------------------------------------------------- güç CLI
PHASE_OF = {
    "char_lr": 2,
    "word_lr": 2,
    "word_svc": 2,
    "word_cnb": 2,
    "char_lr_c1clean": 2,
    "berturk_mean": 3,
    "e5_base": 3,
    "bge_m3": 3,
    "berturk_ft": 4,
}
M_SECONDARY = len(FAMILIES["secondary"]["comparisons"])
SECONDARY_ALPHAS = (ALPHA / M_SECONDARY, ALPHA)
POWER_PAIRS = [
    *(("primary", a, b) for a, b in FAMILIES["primary"]["comparisons"]),
    *(("secondary", a, b) for a, b in FAMILIES["secondary"]["comparisons"]),
    # seed 43/44 final modellerinin birleşik OOF'u yok (PHASE_OF dışı): güçleri hesaplanmaz
    *(("descriptive", a, b) for a, b in DESCRIPTIVE if a in PHASE_OF and b in PHASE_OF),
    ("primary_proxy", "berturk_mean", "char_lr"),
    ("primary_proxy", "e5_base", "char_lr"),
]


def load_oof(name: str, metrics_dir: Path = METRICS_DIR) -> dict | None:
    """Modelin son koşusunun dış-fold (OOF) tahminleri, row_id'ye göre sıralı."""
    m = latest_metrics(name, metrics_dir, phase=PHASE_OF[name])
    if m is None:
        return None
    z = np.load(Path(metrics_dir) / m["preds_file"])
    o = np.argsort(z["row_id"])
    return {
        "source": m["_file"],
        "preds_file": m["preds_file"],
        "git_commit": m["git_commit"],
        "split": m["split"],
        "row_id": z["row_id"][o],
        "y_true": z["y_true"][o],
        "y_pred": z["y_pred"][:, o],
        "fold_id": z["fold_id"][:, o],
    }


def _check_pair(A: dict, B: dict, R: int) -> None:
    if A["split"] != B["split"]:
        raise ValueError("Farklı split'ler.")
    if not np.array_equal(A["row_id"], B["row_id"]) or not np.array_equal(A["y_true"], B["y_true"]):
        raise ValueError("OOF satırları/etiketleri uyuşmuyor.")
    if not np.array_equal(A["fold_id"][:R], B["fold_id"][:R]):
        raise ValueError("Dış fold'lar farklı; eşleştirilmiş karşılaştırma geçersiz.")


def pair_power(A: dict, B: dict, n_test: int, alphas, B_boot: int, seed: int, path: Path) -> dict:
    """Bir çift için OOF uyuşmazlığı, n_test'te accuracy MDE'si ve kaba macro-F1 MDE'si."""
    R = min(len(A["y_pred"]), len(B["y_pred"]))
    _check_pair(A, B, R)
    y = A["y_true"]
    per = []
    for r in range(R):
        ca, cb = A["y_pred"][r] == y, B["y_pred"][r] == y
        mc = mcnemar_exact(ca, cb)
        per.append(
            {
                "repeat": r,
                "psi": mc["n_discordant"] / len(y),
                "n10": mc["n10"],
                "n01": mc["n01"],
                "acc_diff": float(ca.mean() - cb.mean()),
                "macro_f1_diff": macro_f1(y, A["y_pred"][r]) - macro_f1(y, B["y_pred"][r]),
            }
        )
    psi = float(np.mean([p["psi"] for p in per]))
    acc_diff = float(np.mean([p["acc_diff"] for p in per]))
    g = cv_groups(pd.DataFrame({"row_id": A["row_id"]}), path)
    eff = cluster_bootstrap_diff(y, A["y_pred"][0], B["y_pred"][0], g, B_boot, seed)
    iid = cluster_bootstrap_diff(y, A["y_pred"][0], B["y_pred"][0], np.arange(len(y)), B_boot, seed)
    se_test = eff["diff_se_boot"] * np.sqrt(len(y) / n_test)
    by_alpha = {}
    for a in alphas:
        by_alpha[f"{a:g}"] = {
            "mde_acc": mde(n_test, psi, a),
            "power_at_oof_acc_diff": mcnemar_power(n_test, psi, min(abs(acc_diff), psi), a),
            "mde_macro_f1_rough": mde_macro_f1(se_test, a),
            "z_multiplier": float(stats.norm.ppf(1 - a / 2) + stats.norm.ppf(POWER_TARGET)),
        }
    return {
        "n_repeats_used": R,
        "psi_mean": psi,
        "psi_by_repeat": [p["psi"] for p in per],
        "oof_acc_diff_mean": acc_diff,
        "oof_macro_f1_diff_mean": float(np.mean([p["macro_f1_diff"] for p in per])),
        "per_repeat": per,
        "oof_macro_f1_diff_se_boot_r0": eff["diff_se_boot"],
        # grup bootstrap SE / satır (iid) bootstrap SE: train'deki yakın-tekrar bağımlılığının
        # SE'ye etkisi (tasarım etkisinin karekökü); McNemar'ın bağımsızlık varsayımı için gösterge
        "se_ratio_group_vs_iid_r0": float(eff["diff_se_boot"] / iid["diff_se_boot"]),
        "macro_f1_diff_se_test_rough": float(se_test),
        "by_alpha": by_alpha,
        "sources": {"a": A["source"], "b": B["source"]},
        "source_commits": {"a": A["git_commit"], "b": B["git_commit"]},
    }


def power_report(
    metrics_dir: Path = METRICS_DIR,
    path: Path = SPLIT_PATH,
    B_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    split = load_split(path)
    n_test = len(split["test_ids"])
    tg = test_groups(pd.DataFrame({"row_id": split["test_ids"]}), path)  # yalnız id'ler
    sizes = pd.Series(tg).value_counts()
    oof = {n: load_oof(n, metrics_dir) for n in {m for _, a, b in POWER_PAIRS for m in (a, b)}}
    rows = []
    for role, a, b in POWER_PAIRS:
        row = {"role": role, "a": a, "b": b}
        # birincil (m=1) ve betimsel: yalnız α; ikincil ve vekil: α/m ve α
        alphas = (ALPHA,) if role in ("primary", "descriptive") else SECONDARY_ALPHAS
        if oof[a] is None or oof[b] is None:
            row["status"] = "OOF yok (Faz 4 sonrası aynı CLI ile hesaplanır)"
        else:
            row |= pair_power(oof[a], oof[b], n_test, alphas, B_boot, seed, path)
        if role == "descriptive":
            row["note"] = "testte p yok: yalnız mde_macro_f1_rough (CI koşulu) anlamlı"
        rows.append(row)
    return {
        "experiment": "test_power",
        "phase": "test öncesi (yalnız train OOF; test seti açılmadı)",
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": Path(path).stem,
        "n_test": n_test,
        "test_group_structure": {
            "n_groups": int(len(sizes)),
            "group_size_hist": {
                str(int(k)): int(v) for k, v in sizes.value_counts().sort_index().items()
            },
            "note": "yalnızca split dosyasındaki test row_id'leri ve components; metin/etiket yok",
        },
        "settings": {
            "power_target": POWER_TARGET,
            "alpha_primary": ALPHA,
            "alphas_secondary": list(SECONDARY_ALPHAS),
            "bootstrap_B": B_boot,
            "bootstrap_seed": seed,
        },
        "pairs": rows,
        "notes": {
            "mde_acc": (
                "Exact iki yönlü McNemar'ın %80 güce ulaştığı en küçük accuracy farkı "
                "(p10 - p01), n = n_test, psi = OOF uyuşmazlık oranının tekrar ortalaması; psi "
                "alternatif altında sabit tutulur (yalnız uyuşmazlığın yönü değişir). "
                "nan: psi bu güce izin vermiyor."
            ),
            "alphas": (
                f"Birincil aile m=1: α={ALPHA}. İkincil aile m={M_SECONDARY}: Holm'un ilk adımı "
                f"α/{M_SECONDARY} (en kötü), son adımı α (en iyi); p < α/m her zaman reddedilir, "
                "p >= α hiç reddedilmez, bu yüzden iki α'daki güç Holm altındaki marjinal "
                "gücün alt ve üst sınırıdır. Vekil çiftlerde iki değer bilgi amaçlıdır; "
                "betimsel çiftte yalnız α (CI)."
            ),
            "oof_differences": (
                "OOF farkları kazananın lanetiyle yanlı olabilir (char_lr 4 klasik adayın, bge_m3 "
                "3 encoder'ın CV maksimumu): seçilen model a tarafındaysa fark yukarı, b "
                "tarafındaysa aşağı yanlıdır. Birincil çiftte (berturk_ft - char_lr) BERT "
                "önceden belirlenmiş tek konfigürasyon, kazanan char_lr b tarafında: OOF farkı "
                "aşağı yanlı. Gerçek test farkı daha küçükse güç daha da düşüktür."
            ),
            "se_ratio_group_vs_iid_r0": (
                "Train OOF'ta grup bootstrap SE'si / satır bootstrap SE'si (macro-F1 farkı). "
                "1'e yakınsa yakın-tekrar bağımlılığının SE'ye etkisi küçüktür; testte çok satırlı "
                "gruplardaki satır oranı ayrıca test_group_structure'da."
            ),
            "mde_macro_f1_rough": (
                "Kaba normal yaklaşım: (z_{1-α/2} + z_0.8) · SE_test; SE_test = OOF tekrar 0 "
                "grup bootstrap SE'si · sqrt(n_oof / n_test). Karar kuralının CI koşulu içindir."
            ),
            "power_at_oof_acc_diff": (
                "Gerçek test farkı OOF accuracy farkına eşitse beklenen güç (yalnız train'den)."
            ),
            "approximation": (
                "OOF modelleri dış fold'un train'iyle (~2904 satır; BERT fine-tuning'de validation "
                "ayrıldığı için ~2323) eğitildi, final modeller 3630 satırla; psi ve SE "
                "yaklaşıktır. McNemar gücü test içi grup bağımlılığını yok "
                "sayar (TEST_PLAN['limitations']). Karar kuralı iki koşulu birlikte istediği için "
                "ortak güç her birinin gücünden yüksek olamaz."
            ),
            "primary_proxy": (
                "Birincil satır (berturk_ft - char_lr) Faz 4'ün seed 42 CV OOF'undan hesaplanır "
                "(TEST_PLAN['power']; yoksa 'OOF yok'). Donmuş BERTurk/e5 + LR vekilleri yalnız "
                "karşılaştırma için kalır; fine-tuned modelin char_lr ile uyuşmazlığını temsil "
                "etmeyebilir. Birincil OOF tek tekrardır (R=1): psi ve SE tek bölümlemeden."
            ),
        },
        "test_plan_power": TEST_PLAN["power"],
        "test": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--power", action="store_true", help="train OOF'tan test gücü / MDE")
    args = ap.parse_args()
    if not args.power:
        ap.error("şimdilik yalnızca --power (test değerlendirmesi Faz 5'te)")
    out = power_report()
    p = report.write_metrics(out, "test_power")
    print(p)
    print(f"n_test={out['n_test']}  test grupları={out['test_group_structure']['n_groups']}")
    print(
        f"{'rol':15s} {'a - b':28s} {'psi':>6s} {'OOF Δacc':>9s} {'OOF ΔF1':>8s}  MDE (α: acc / F1)"
    )
    for r in out["pairs"]:
        name = f"{r['a']} - {r['b']}"
        if "status" in r:
            print(f"{r['role']:15s} {name:28s} {r['status']}")
            continue
        mdes = "  ".join(
            f"{a}: {v['mde_acc']:.4f} / {v['mde_macro_f1_rough']:.4f}"
            for a, v in r["by_alpha"].items()
        )
        print(f"{r['role']:15s} {name:28s} {r['psi_mean']:6.3f} {r['oof_acc_diff_mean']:+9.4f} "
              f"{r['oof_macro_f1_diff_mean']:+8.4f}  {mdes}")  # fmt: skip


if __name__ == "__main__":
    main()

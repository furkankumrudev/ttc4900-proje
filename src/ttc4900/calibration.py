"""Faz 6: olasılık kalibrasyonu (temperature scaling). Kurallar `CALIB_RULES`'tadır; 6a commit'inde
hiçbir kalibrasyon sayısı hesaplanmadan donduruldu. İnsan okuması: docs/faz6_protokol.md.

Sezgi: aşırı emin bir model büyük logit üretir; tüm logit'leri tek bir T > 0'a bölmek olasılıkları
yumuşatır (T > 1) ya da sertleştirir (T < 1). Sıralama değişmez, tahmin edilen sınıf değişmez.

    p̂(k | x) = softmax(z(x) / T)_k,   T̂ = argmin_T  -(1/n) Σ_i log p̂(y_i | x_i)
    ΔNLL = NLL(önce, T = 1) - NLL(sonra);  pozitif = iyileşme

Akış (sıra zorunlu, her adım temiz commit'te):
    python -m ttc4900.calibration train   # 6b: yalnız train OOF; ana kanıt
    python -m ttc4900.calibration folds   # sonradan, betimsel: fold başına cross-fit ΔNLL
    python -m ttc4900.calibration test    # 6c: test npz skorları, TEK SEFER; teyit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from ttc4900 import report
from ttc4900.cv_compare import latest_metrics
from ttc4900.data import SPLIT_PATH, cv_groups, load_split, load_train, test_groups
from ttc4900.final_eval import (
    PREDS_DIR,
    _is_tracked,
    _untracked_inputs,
    fit_frozen,
    make_frozen,
    preds_path,
    sha256_file,
    write_once,
)
from ttc4900.paired_test import _pct_ci, bootstrap_weights, group_index
from ttc4900.paths import FIGURES_DIR, METRICS_DIR
from ttc4900.test_plan import BOOTSTRAP_B, BOOTSTRAP_SEED

SPLIT_NAME = Path(SPLIT_PATH).stem
CALIB_MODELS = {
    "berturk_ft": "birincil",
    "char_lr": "birincil",
    "berturk_ft_s43": "betimsel",
    "berturk_ft_s44": "betimsel",
}
PRIMARY = tuple(n for n, r in CALIB_MODELS.items() if r == "birincil")
BERT_SEEDS = {"berturk_ft": 42, "berturk_ft_s43": 43, "berturk_ft_s44": 44}
CHAR_LR = "char_lr"
BETWEEN = ("berturk_ft", "char_lr")  # yalnız betimsel fark
T_BOUNDS = (0.05, 20.0)
T_XATOL = 1e-6
BOUND_TOL = 1e-3  # log T sınıra bu kadar yakınsa "sınırda"
ECE_BINS = 15
ECE_BINS_SENS = (10, 20)
METRICS = ("nll", "brier", *(f"ece{m}" for m in (ECE_BINS, *ECE_BINS_SENS)))

CALIB_RULES = {
    "status": (
        "6a commit'inde, hiçbir kalibrasyon sayısı hesaplanmadan donduruldu; sonuçlara göre "
        "değiştirilmez. Faz 5 sınıflandırma sonuçları değişmez (temperature scaling argmax'ı "
        "korur; kod her modelde denetler)."
    ),
    "scope": (
        "İki model, iki ayrı soru: berturk_ft (seed 42) ve char_lr birincil; berturk_ft_s43 ve "
        "berturk_ft_s44 betimsel (karar yok). word_lr, word_svc, word_cnb, char_lr_c1clean, "
        "bge_m3 kapsam dışı."
    ),
    "method": (
        "Temperature scaling: p̂ = softmax(z / T). T, kalibrasyon kümesinde ortalama NLL'yi "
        "minimize eder: scipy.optimize.minimize_scalar, log T üzerinde, sınırlar "
        f"[log {T_BOUNDS[0]}, log {T_BOUNDS[1]}], 'bounded', xatol {T_XATOL}. Optimum bir sınıra "
        f"(log T'de {BOUND_TOL} içinde) düşerse kaydedilir ve o modelin kararı 'değerlendirilemez' "
        "olur. Logit: BERT'te model logit'leri; char_lr'de OOF'ta decision_function, testte "
        "log(predict_proba) (log p_k = z_k - logsumexp(z), sabit kayma softmax'ı değiştirmez); "
        "p = 0 alt taşması np.finfo(float).tiny'ye kırpılır, kırpılan hücre sayısı raporlanır."
    ),
    "metrics": (
        "Birincil: NLL = -(1/n) Σ log p̂(y_i) (T'nin optimize ettiği uygun skor kuralı); karar "
        "yalnız NLL'ye dayanır. ΔNLL = NLL(önce, T = 1) - NLL(sonra); pozitif = iyileşme. "
        "Betimsel: Brier = (1/n) Σ_i Σ_k (p̂_k - 1[y_i = k])²; top-label ECE = Σ_m (|B_m|/n) "
        f"|acc(B_m) - conf(B_m)|, conf = max_k p̂_k, {ECE_BINS} eşit genişlikte bin "
        f"((m-1)/M, m/M]; bin duyarlılığı için {ECE_BINS_SENS} bin; reliability diyagramı "
        f"({ECE_BINS} bin). ECE yanlı ve bin seçimine duyarlıdır. Bütün metriklerde Δ = önce - "
        "sonra (hepsi küçük = iyi)."
    ),
    "train_eval": (
        "6b (ana kanıt), yalnız train OOF; test npz'leri okunmaz. BERT: Faz 4 CV koşularının "
        "dış fold logit'leri (early stopping'in iç validation'ı değil; seçim sızıntısı yok); "
        "seed başına 5 koşu birleştirilir, seed 42 phase4 birincil OOF npz'siyle birebir "
        "denetlenir. char_lr: dış fold'lar char_lr npz fold_id[0]'dan (BERT ile aynı fold'lar; "
        "make_cv çalıştırılmaz); her fold'da donmuş final.best_params ile load_train() verisinde "
        "fit, decision_function kaydedilir (write-once npz); ayar yok. Cross-fit: fold j'nin T'si "
        "diğer 4 fold'un OOF'unda fit edilir ve fold j'ye uygulanır. Birincil 6b büyüklüğü: "
        "cross-fit ΔNLL. CI: train grupları (split_v2 components, cv_groups) üzerinde grup "
        f"bootstrap, B = {BOOTSTRAP_B}, np.random.default_rng({BOOTSTRAP_SEED}) "
        "(paired_test.bootstrap_weights), %95 yüzdelik; T'ler sabit tutulur (CI T kestirim "
        "belirsizliğini içermez). Karar (her birincil model için ayrı): 'train içinde iyileşme "
        "gösterildi' ⇔ ΔNLL CI'ı tamamen 0'ın üstünde; değilse 'gösterilemedi'; p yok."
    ),
    "final_T": (
        "T̂_final modelin tüm 3630 OOF satırında fit edilir. Belirsizliği: train grupları "
        "üzerinde grup bootstrap'i, her replikatta T yeniden fit edilir (aynı B, seed), %95 "
        "yüzdelik. T̂_final 6b json'una yazılır; 6c yalnız bu değeri okur."
    ),
    "per_run_T": (
        "Betimsel (karar yok): 15 BERT CV koşusunun (5 fold × 3 seed) her birinde T yalnız o "
        "koşunun kendi 726 OOF satırında ayrı fit edilir. Rapor: 5 × 3 tablo, genel ortalama, "
        "std (ddof=1), min, max; seed başına (fold'lar üzerinden) ve fold başına (seed'ler "
        "üzerinden) ortalama ve std; her koşunun best epoch'u yan yana (Faz 4 kayıtları). Bu "
        "T'ler in-sample'dır, performans iddiası için kullanılmaz. char_lr için 5 fold'un T'si "
        "aynı biçimde."
    ),
    "test_eval": (
        "6c (teyit), tek sefer. Ön koşullar: ağaç temiz; 6b json'u izleniyor (sha256 "
        "kaydedilir); daha önce bir phase6_test_calibration json'u yok; izlenmeyen girdi yok. "
        "Girdi yalnız outputs/metrics/preds/test_<model>.npz 'scores'; sha256'ları Faz 5 sonuç "
        "json'undakiyle aynı olmalı; metin okunmaz, model koşulmaz, testte hiçbir şey fit "
        "edilmez. T̂_final değiştirilmeden uygulanır. ΔNLL_test = NLL(T = 1) - NLL(T̂_final). CI: "
        f"Faz 5 ile aynı grup bootstrap ağırlıkları (test_groups, B = {BOOTSTRAP_B}, seed "
        f"{BOOTSTRAP_SEED}); kod, accuracy CI'larının Faz 5 json'uyla birebir aynı çıktığını "
        "denetler. Karar: 'testte teyit edildi' ⇔ ΔNLL_test %95 CI'ı tamamen 0'ın üstünde; "
        "değilse 'teyit edilemedi'. Ana kanıt 6b'dir; test sonucu 6b kararını değiştirmez."
    ),
    "between_models": (
        "İki model iki ayrı sorudur. berturk_ft ile char_lr arasındaki kalibrasyon farkları "
        "(kalibre NLL/Brier/ECE, ΔNLL farkı, T̂_final farkı) yalnız betimsel verilir: aynı "
        "w'lerden fark ve CI; p yok, karar yok, 'X, Y'den daha iyi kalibre' iddiası yok."
    ),
    "limitations": (
        "Model uyuşmazlığı (BERT): T, ~2323 satırla early stopping ile eğitilen CV modellerinin "
        "OOF logit'lerinden öğrenilir, 3630 satırla sabit 4 epoch eğitilen final modele "
        "uygulanır; final modelin güven düzeyi farklı olabilir, yönü bilinmiyor; yeniden eğitim "
        "yok. 15 koşunun T dağılımı bu duyarlılığın bağlamıdır. char_lr: OOF modelleri 2904, "
        "final 3630 satırla; parametreler tüm train'de seçildi. OOF logit'leri Colab fp16, test "
        "logit'leri CPU fp32. CI'lar sabit T'ye koşulludur. Kalibrasyon yalnız bu veri dağılımı "
        "için geçerlidir (tek kaynak ve dönem, kısayol artefaktları; docs/shortcut_learning.md)."
    ),
    "demo_language": (
        "Demoda BERTurk bir dağıtım tercihidir (Faz 5'te önceden belirlenmiş birincil model); "
        "testte char_lr'den ayırt edilemedi, 'char_lr'den iyi' iddiası yapılmaz. Olasılıklar "
        "için 'bu veri dağılımında kalibre edildi' yazılır; kısayol notları (bozuk tırnaklar, "
        "ajans künyeleri, dönem adları) demo açıklamasında yer alır."
    ),
    "commits": (
        "6a: kurallar + kod + testler (sayı yok). 6b: temiz commit'te koşu, json + npz + figür, "
        "yorumsuz. 6c: temiz commit'te test koşusu, json + figür, yorumsuz. Yorum ve belgeler "
        "sonraki commit'te."
    ),
    "interpretation": (
        "'Teyit edilemedi' etki olmadığı anlamına gelmez (test 908 satır, train'in dörtte biri; "
        "CI'lar daha geniş); yorumda açıkça yazılır. Ana kanıt 6b'dir."
    ),
    "change_log": (
        "6b sonuçları görüldükten sonra, 6c'den ÖNCE, kullanıcı onayıyla: (1) 'teyit edilemedi' "
        "şablonu nedeni önceden atfetmek yerine 'olası nedenler: model uyuşmazlığı, düşük güç' "
        "der; karar mantığı (decide) DEĞİŞMEDİ. (2) 'interpretation' kuralı eklendi. (3) Fold "
        "başına cross-fit ΔNLL dökümü (fold_breakdown) sonradan ve betimsel olarak eklendi; "
        "hiçbir karara girmez."
    ),
}

SENTENCES = {
    "gösterildi": (
        "{model}: train içinde cross-fit ΔNLL {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (NLL {b:.4f} → "
        "{a:.4f}); iyileşme gösterildi. T̂_final = {t:.3f} [{tlo:.3f}, {thi:.3f}]."
    ),
    "gösterilemedi": (
        "{model}: train içinde cross-fit ΔNLL {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (NLL {b:.4f} → "
        "{a:.4f}); CI tamamen 0'ın üstünde değil, iyileşme gösterilemedi. Bu, modelin kalibre "
        "olduğunu kanıtlamaz. T̂_final = {t:.3f} [{tlo:.3f}, {thi:.3f}]."
    ),
    "teyit edildi": (
        "{model}: testte ΔNLL {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (NLL {b:.4f} → {a:.4f}, "
        "T̂_final = {t:.3f}); teyit edildi (ana kanıt 6b)."
    ),
    "teyit edilemedi": (
        "{model}: testte ΔNLL {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (NLL {b:.4f} → {a:.4f}, "
        "T̂_final = {t:.3f}); teyit edilemedi; olası nedenler: model uyuşmazlığı, düşük güç. "
        "Bu, testte etki olmadığı anlamına gelmez. 6b kararı değişmez (ana kanıt 6b)."
    ),
    "değerlendirilemez": ("{model}: T optimizasyonu sınıra düştü (T = {t:.3f}); karar verilmedi."),
    "betimsel": (
        "{model} (betimsel, karar yok): ΔNLL {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (NLL {b:.4f} → "
        "{a:.4f}), T̂ = {t:.3f}."
    ),
    "modeller_arasi": (
        "{a} − {b} (betimsel, p yok; 'daha iyi kalibre' iddiası değildir): {metric} farkı "
        "{d:+.4f} [{lo:+.4f}, {hi:+.4f}]."
    ),
}


# --------------------------------------------------------------------------- temel
def log_probs(z, T=1.0) -> np.ndarray:
    """log softmax(z / T). T skaler ya da satır başına (n,) dizi."""
    z = np.asarray(z, dtype=float)
    T = np.asarray(T, dtype=float)
    return log_softmax(z / (T[:, None] if T.ndim == 1 else T), axis=1)


def nll_rows(z, y, T=1.0) -> np.ndarray:
    lp = log_probs(z, T)
    return -lp[np.arange(len(lp)), np.asarray(y)]


def fit_temperature(z, y, w=None) -> dict:
    """CALIB_RULES['method']: (ağırlıklı) ortalama NLL'yi log T üzerinde minimize eder."""
    z, y = np.asarray(z, dtype=float), np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    lo, hi = np.log(T_BOUNDS)

    def f(log_t: float) -> float:
        return float(np.average(nll_rows(z, y, np.exp(log_t)), weights=w))

    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": T_XATOL})
    at_bound = bool(r.x - lo < BOUND_TOL or hi - r.x < BOUND_TOL)
    return {"T": float(np.exp(r.x)), "at_bound": at_bound, "nll": float(r.fun)}


def cross_fit(z, y, fold) -> tuple[np.ndarray, list[dict]]:
    """Fold j'nin T'si diğer fold'larda fit edilir; satır başına T döner (held-out görülmez)."""
    z, y, fold = np.asarray(z, dtype=float), np.asarray(y), np.asarray(fold)
    t_row, per = np.empty(len(y)), []
    for k in np.unique(fold):
        m = fold == k
        r = fit_temperature(z[~m], y[~m])
        t_row[m] = r["T"]
        per.append({"fold": int(k), **r})
    return t_row, per


def t_bootstrap(z, y, groups, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """CALIB_RULES['final_T']: her grup bootstrap replikatında T yeniden fit edilir."""
    inv, G = group_index(groups)
    out = np.empty(B)
    for start, w in bootstrap_weights(G, B, seed):
        for i, wr in enumerate(w):
            out[start + i] = fit_temperature(z, y, wr[inv])["T"]
    return out


# --------------------------------------------------------------------------- metrikler
def _bin_index(conf: np.ndarray, m: int) -> np.ndarray:
    """((b-1)/M, b/M] binleri -> 0..M-1."""
    return np.clip(np.ceil(conf * m).astype(int) - 1, 0, m - 1)


def group_stats(lp: np.ndarray, y, inv: np.ndarray, G: int) -> dict:
    """Satır katkılarının grup toplamları: replikat metrikleri Σ_g w_g (·)_g'den hesaplanır."""
    y = np.asarray(y)
    n, k = lp.shape
    p = np.exp(lp)
    onehot = np.eye(k)[y]
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
    agg = lambda v: np.bincount(inv, weights=v, minlength=G)  # noqa: E731
    out = {
        "n": agg(np.ones(n)),
        "nll": agg(-lp[np.arange(n), y]),
        "brier": agg(((p - onehot) ** 2).sum(1)),
        "acc": agg(correct),
    }
    for m in (ECE_BINS, *ECE_BINS_SENS):
        idx = inv * m + _bin_index(conf, m)
        cell = lambda v: np.bincount(idx, weights=v, minlength=G * m).reshape(G, m)  # noqa: E731,B023
        out[f"ece{m}"] = (cell(conf), cell(correct))
    return out


def _metrics_from(stats: dict, w: np.ndarray) -> dict[str, np.ndarray]:
    """w: (R, G). Döner: metrik -> (R,)."""
    tot = w @ stats["n"]
    out = {k: (w @ stats[k]) / tot for k in ("nll", "brier", "acc")}
    for m in (ECE_BINS, *ECE_BINS_SENS):
        c, a = stats[f"ece{m}"]
        out[f"ece{m}"] = np.abs(w @ a - w @ c).sum(1) / tot
    return out


def bootstrap_metrics(
    lps: dict[str, np.ndarray], y, groups, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> tuple[dict, dict]:
    """Tüm girdiler AYNI w'lerle. Döner: (nokta, replikat) — ad -> metrik -> değer/(B,)."""
    inv, G = group_index(groups)
    st = {k: group_stats(lp, y, inv, G) for k, lp in lps.items()}
    point = {k: {m: float(v[0]) for m, v in _metrics_from(s, np.ones((1, G))).items()}
             for k, s in st.items()}  # fmt: skip
    reps = {k: {m: np.empty(B) for m in (*METRICS, "acc")} for k in lps}
    for start, w in bootstrap_weights(G, B, seed):
        for k, s in st.items():
            for m, v in _metrics_from(s, w).items():
                reps[k][m][start : start + len(w)] = v
    return point, reps


def before_after(point: dict, reps: dict, model: str) -> dict:
    """Her metrik için önce, sonra ve Δ = önce - sonra (pozitif = iyileşme), CI'larıyla."""
    b, a = f"{model}:before", f"{model}:after"
    out = {}
    for m in METRICS:
        out[m] = {
            "before": point[b][m],
            "before_ci95": _pct_ci(reps[b][m]),
            "after": point[a][m],
            "after_ci95": _pct_ci(reps[a][m]),
            "delta": point[b][m] - point[a][m],
            "delta_ci95": _pct_ci(reps[b][m] - reps[a][m]),
        }
    return out


def decide(delta_ci, at_bound: bool, stage: str) -> str:
    """6b: gösterildi/gösterilemedi; 6c: teyit edildi/edilemedi; sınırda T: değerlendirilemez."""
    if at_bound:
        return "değerlendirilemez"
    ok = delta_ci[0] > 0
    if stage == "train":
        return "gösterildi" if ok else "gösterilemedi"
    return "teyit edildi" if ok else "teyit edilemedi"


def _sentence(verdict: str, model: str, r: dict, t: float, t_ci=None) -> str:
    n = r["nll"]
    lo, hi = n["delta_ci95"]
    tlo, thi = t_ci if t_ci is not None else (float("nan"), float("nan"))
    return SENTENCES[verdict].format(model=model, d=n["delta"], lo=lo, hi=hi, b=n["before"],
                                     a=n["after"], t=t, tlo=tlo, thi=thi)  # fmt: skip


def between_models(point: dict, reps: dict, a: str, b: str) -> list[dict]:
    """CALIB_RULES['between_models']: kalibre metrik farkları ve ΔNLL farkı; betimsel."""
    rows = []
    for m in METRICS:
        d = point[f"{a}:after"][m] - point[f"{b}:after"][m]
        r = reps[f"{a}:after"][m] - reps[f"{b}:after"][m]
        rows.append({"metric": f"{m} (sonra)", "diff": d, "diff_ci95": _pct_ci(r)})
    da = reps[f"{a}:before"]["nll"] - reps[f"{a}:after"]["nll"]
    db = reps[f"{b}:before"]["nll"] - reps[f"{b}:after"]["nll"]
    d = (point[f"{a}:before"]["nll"] - point[f"{a}:after"]["nll"]) - (
        point[f"{b}:before"]["nll"] - point[f"{b}:after"]["nll"]
    )
    rows.append({"metric": "ΔNLL", "diff": d, "diff_ci95": _pct_ci(da - db)})
    for r in rows:
        lo, hi = r["diff_ci95"]
        r["inference"] = "betimsel; p yok, karar yok"
        r["sentence"] = SENTENCES["modeller_arasi"].format(a=a, b=b, metric=r["metric"],
                                                           d=r["diff"], lo=lo, hi=hi)  # fmt: skip
    return rows


def reliability(lp: np.ndarray, y, m: int = ECE_BINS) -> list[dict]:
    p = np.exp(lp)
    conf, correct = p.max(1), p.argmax(1) == np.asarray(y)
    b = _bin_index(conf, m)
    return [
        {"bin": i, "lo": i / m, "hi": (i + 1) / m, "n": int((b == i).sum()),
         "conf": float(conf[b == i].mean()) if (b == i).any() else None,
         "acc": float(correct[b == i].mean()) if (b == i).any() else None}
        for i in range(m)
    ]  # fmt: skip


def plot_reliability(rel: dict[str, dict], path: Path, title: str) -> None:
    """Birincil modeller: önce (gri) ve sonra (mavi) reliability eğrileri; nokta alanı ~ n."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, surf, hue, ref_c = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#8a8984"
    fig, axes = plt.subplots(1, len(rel), figsize=(5.2 * len(rel), 4.6), facecolor=surf)
    for ax, (name, r) in zip(np.atleast_1d(axes), rel.items(), strict=True):
        ax.plot([0, 1], [0, 1], color=ink2, lw=1, ls="--")
        for key, col, lab in (("before", ref_c, "önce (T = 1)"), ("after", hue, "sonra")):
            pts = [b for b in r[key] if b["n"] > 0]
            ax.plot([b["conf"] for b in pts], [b["acc"] for b in pts], color=col, lw=1.6,
                    label=lab)  # fmt: skip
            ax.scatter([b["conf"] for b in pts], [b["acc"] for b in pts], color=col,
                       s=[8 + 0.4 * b["n"] for b in pts], alpha=0.8, linewidths=0)  # fmt: skip
        ax.set_title(name, loc="left", fontsize=10, color=ink)
        ax.set_xlabel("Ortalama güven (bin)")
        ax.set_ylabel("Accuracy (bin)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_facecolor(surf)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, color=ink)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


# --------------------------------------------------------------------------- 6b: OOF girdileri
def _sorted(row_id, y, z, fold) -> dict:
    o = np.argsort(row_id)
    return {"row_id": np.asarray(row_id)[o], "y": np.asarray(y)[o],
            "z": np.asarray(z, dtype=float)[o], "fold": np.asarray(fold)[o]}  # fmt: skip


def bert_oof(seed: int, metrics_dir: Path = METRICS_DIR) -> tuple[dict, list[dict]]:
    """Faz 4 CV koşularının dış fold logit'leri (seed başına 5 koşu). Döner: (oof, koşular)."""
    summ = latest_metrics("berturk_ft", metrics_dir, phase=4)
    s = json.loads(Path(metrics_dir, _summary_file(metrics_dir)).read_text(encoding="utf-8"))
    runs = [r for r in s["runs"] if r["seed"] == seed]
    if len(runs) != 5:
        raise ValueError(f"seed {seed}: {len(runs)} CV koşusu (beklenen 5)")
    parts = []
    for r in runs:
        rj = json.loads(Path(metrics_dir, r["file"]).read_text(encoding="utf-8"))
        with np.load(Path(metrics_dir, rj["preds_file"]), allow_pickle=False) as z:
            parts.append({k: z[k] for k in ("row_id", "y_true", "logits", "fold_id")})
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    oof = _sorted(cat["row_id"], cat["y_true"], cat["logits"], cat["fold_id"])
    if seed == s["finals"]["primary_seed"]:  # birleşik birincil OOF ile birebir aynı olmalı
        with np.load(Path(metrics_dir, summ["preds_file"]), allow_pickle=False) as z:
            ref = _sorted(z["row_id"], z["y_true"], z["logits"], z["fold_id"][0])
        for k in ref:
            if not np.array_equal(ref[k], oof[k]):
                raise ValueError(f"seed {seed}: koşu npz'leri birincil OOF ile uyuşmuyor ({k})")
    return oof, runs


def _summary_file(metrics_dir: Path) -> str:
    s = latest_metrics("summary", metrics_dir, phase=4)
    if s is None:
        raise FileNotFoundError("phase4_summary json'u yok")
    return s["_file"]


def char_lr_oof(metrics_dir: Path = METRICS_DIR, load_data=load_train) -> tuple[dict, dict]:
    """Donmuş char_lr, char_lr npz fold_id[0] dış fold'larında: decision_function OOF'u."""
    m = latest_metrics(CHAR_LR, metrics_dir, phase=2)
    with np.load(Path(metrics_dir, m["preds_file"]), allow_pickle=False) as z:
        rid, fold, y_npz = z["row_id"], z["fold_id"][0], z["y_true"]
    train = load_data().set_index("row_id").loc[rid].reset_index()
    if not np.array_equal(train["label"].to_numpy(), y_npz):
        raise ValueError("char_lr npz etiketleri load_train ile uyuşmuyor")
    scores = np.empty((len(rid), 7))
    info = None
    for k in np.unique(fold):
        te = fold == k
        pipe, info = make_frozen(CHAR_LR, metrics_dir)
        fit_frozen(pipe, train["text"][~te], train["label"][~te])
        if not np.array_equal(pipe.classes_, np.arange(7)):
            raise ValueError("char_lr sınıf sırası 0..6 değil")
        scores[te] = pipe.decision_function(train["text"][te])
    return _sorted(rid, y_npz, scores, fold), info | {"preds_source": m["preds_file"]}


# --------------------------------------------------------------------------- 6b: çekirdek
def _t_summary(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}  # fmt: skip


def per_run_table(oofs: dict[str, dict], runs: dict[str, list[dict]]) -> dict:
    """CALIB_RULES['per_run_T']: BERT 15 koşu (her biri kendi 726 OOF satırında) ve char_lr'nin
    5 fold'u."""
    rows = []
    for name, seed in BERT_SEEDS.items():
        o = oofs[name]
        for r in runs[name]:
            m = o["fold"] == r["fold"]
            fit = fit_temperature(o["z"][m], o["y"][m])
            rows.append({"seed": seed, "fold": r["fold"], "n": int(m.sum()), "T": fit["T"],
                         "at_bound": fit["at_bound"], "best_epoch": r["best_epoch"]})  # fmt: skip
    df = pd.DataFrame(rows)
    o = oofs[CHAR_LR]
    lr_rows = []
    for k in np.unique(o["fold"]):
        fit = fit_temperature(o["z"][o["fold"] == k], o["y"][o["fold"] == k])
        lr_rows.append({"fold": int(k), "T": fit["T"], "at_bound": fit["at_bound"]})
    return {
        "note": "in-sample (her koşu kendi OOF satırında); betimsel, performans iddiası değil",
        "bert_runs": rows,
        "bert_overall": _t_summary(df["T"]),
        "bert_by_seed": {int(s): _t_summary(g["T"]) for s, g in df.groupby("seed")},
        "bert_by_fold": {int(f): _t_summary(g["T"]) for f, g in df.groupby("fold")},
        "char_lr_folds": lr_rows,
        "char_lr_overall": _t_summary([r["T"] for r in lr_rows]),
    }


def calibrate_oof(oofs: dict[str, dict], groups, B: int = BOOTSTRAP_B,
                  seed: int = BOOTSTRAP_SEED) -> dict:  # fmt: skip
    """6b çekirdeği: cross-fit, final T ve CI'ı, metrikler (aynı w'ler), kararlar."""
    ref = oofs[PRIMARY[0]]
    for n, o in oofs.items():
        if not (np.array_equal(o["row_id"], ref["row_id"]) and np.array_equal(o["y"], ref["y"])):
            raise ValueError(f"{n}: OOF satırları/etiketleri birincil BERT ile uyuşmuyor")
    lps, fits = {}, {}
    for n, o in oofs.items():
        t_row, per_fold = cross_fit(o["z"], o["y"], o["fold"])
        final = fit_temperature(o["z"], o["y"])
        t_reps = t_bootstrap(o["z"], o["y"], groups, B, seed)
        lps[f"{n}:before"], lps[f"{n}:after"] = log_probs(o["z"]), log_probs(o["z"], t_row)
        if not np.array_equal(lps[f"{n}:before"].argmax(1), lps[f"{n}:after"].argmax(1)):
            raise AssertionError(f"{n}: kalibrasyon argmax'ı değiştirdi")
        fits[n] = {"final": final, "t_reps": t_reps, "cross_fit_folds": per_fold}
    point, reps = bootstrap_metrics(lps, ref["y"], groups, B, seed)
    models, sentences = {}, []
    for n in oofs:
        f = fits[n]
        at_bound = f["final"]["at_bound"] or any(p["at_bound"] for p in f["cross_fit_folds"])
        r = before_after(point, reps, n)
        t_ci = _pct_ci(f["t_reps"])
        verdict = decide(r["nll"]["delta_ci95"], at_bound, "train")
        role = CALIB_MODELS[n]
        models[n] = {
            "role": role,
            "T_final": f["final"]["T"],
            "T_final_ci95": t_ci,
            "T_final_at_bound": f["final"]["at_bound"],
            "cross_fit_folds": f["cross_fit_folds"],
            "metrics_cross_fit": r,
            "accuracy": point[f"{n}:before"]["acc"],
            "verdict": verdict if role == "birincil" else None,
        }
        key = verdict if role == "birincil" or verdict == "değerlendirilemez" else "betimsel"
        sentences.append(_sentence(key, n, r, f["final"]["T"], t_ci))
    a, b = BETWEEN
    t_diff = fits[a]["t_reps"] - fits[b]["t_reps"]
    between = between_models(point, reps, a, b) + [{
        "metric": "T̂_final", "diff": fits[a]["final"]["T"] - fits[b]["final"]["T"],
        "diff_ci95": _pct_ci(t_diff), "inference": "betimsel; p yok, karar yok",
    }]  # fmt: skip
    rel = {n: {"before": reliability(lps[f"{n}:before"], ref["y"]),
               "after": reliability(lps[f"{n}:after"], ref["y"])} for n in oofs}  # fmt: skip
    return {"models": models, "between_models": between, "sentences": sentences,
            "reliability": rel}  # fmt: skip


def _check_clean(what: str) -> str:
    head = report.git_commit()
    if head == "unknown" or head.endswith("-dirty"):
        raise RuntimeError(f"çalışma ağacı temiz değil ({head}); {what} temiz commit'te koşar")
    untracked = _untracked_inputs()
    if untracked:
        raise RuntimeError(f"izlenmeyen girdi dosyaları var: {untracked}; {what} koşmaz")
    return head


def calibrate_train(
    metrics_dir: Path = METRICS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    fig_dir: Path = FIGURES_DIR,
    load_data=load_train,
    path: Path = SPLIT_PATH,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Path, dict]:
    """6b (CALIB_RULES['train_eval']): yalnız train OOF; test npz'lerine dokunmaz."""
    head = _check_clean("6b")
    started = report.now()
    oofs, runs = {}, {}
    for n, s in BERT_SEEDS.items():
        oofs[n], runs[n] = bert_oof(s, metrics_dir)
    oofs[CHAR_LR], lr_info = char_lr_oof(metrics_dir, load_data)
    train_ids = set(load_split(path)["train_ids"])
    if set(oofs[PRIMARY[0]]["row_id"].tolist()) != train_ids:
        raise ValueError("OOF satırları split train'iyle aynı değil")
    groups = cv_groups(pd.DataFrame({"row_id": oofs[PRIMARY[0]]["row_id"]}), path)
    core = calibrate_oof(oofs, groups, B, seed)
    o = oofs[CHAR_LR]
    p_npz = Path(preds_dir) / f"cv_phase6_char_lr_scores_{started:%Y%m%d-%H%M%S}.npz"
    sha = write_once(p_npz, row_id=o["row_id"], y_true=o["y"], scores=o["z"], fold_id=o["fold"],
                     meta=np.array(json.dumps({"split": SPLIT_NAME, "score": "decision_function",
                                               "git_commit": head})))  # fmt: skip
    fig = Path(fig_dir) / "phase6_train_reliability.png"
    title = "Train OOF (cross-fit T): reliability, 15 bin; nokta alanı ~ satır sayısı"
    plot_reliability({n: core["reliability"][n] for n in PRIMARY}, fig, title)
    out = {
        "experiment": "phase6_train_calibration",
        "phase": 6,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": head,
        "split": SPLIT_NAME,
        "n": len(groups),
        "n_groups": int(len(np.unique(groups))),
        "bootstrap": {"B": B, "seed": seed, "unit": "split_v2 components (cv_groups)"},
        **{k: core[k] for k in ("models", "between_models", "sentences", "reliability")},
        "per_run_T": per_run_table(oofs, runs),
        "char_lr_oof": {"frozen": lr_info, "file": p_npz.name, "sha256": sha},
        "figure": fig.name,
        "rules": CALIB_RULES,
    }
    return report.write_metrics(out, "phase6_train_calibration", out_dir, when=started), out


# --------------------------------------------------------------------------- sonradan: fold dökümü
POSTHOC_NOTE = (
    "Sonradan, betimsel: 6b sonuçları görüldükten sonra kullanıcı onayıyla eklendi; "
    "CALIB_RULES kararlarına girmez, p yok. Her fold'da 6b'nin cross-fit T'si (diğer 4 fold'da "
    "fit) o fold'a uygulanır; CI o fold'un grupları üzerinde grup bootstrap'iyle, T sabit. "
    "T_own_in_sample: koşunun (char_lr'de fold'un) kendi satırlarında fit edilen T (6b per_run_T)."
)


def fold_breakdown(oofs: dict[str, dict], six_b: dict, groups, B: int = BOOTSTRAP_B,
                   seed: int = BOOTSTRAP_SEED) -> dict:  # fmt: skip
    """Fold başına cross-fit ΔNLL. 6b'nin T'leri yeniden hesaplanıp json'la denetlenir; fold
    ΔNLL'lerinin n ağırlıklı ortalaması 6b'nin toplam ΔNLL'sine eşit olmalıdır."""
    runs = six_b["per_run_T"]
    out = {}
    for n, o in oofs.items():
        m6 = six_b["models"][n]
        t_cf = {p["fold"]: p["T"] for p in m6["cross_fit_folds"]}
        _, per = cross_fit(o["z"], o["y"], o["fold"])
        if not np.allclose([p["T"] for p in per], [t_cf[p["fold"]] for p in per]):
            raise ValueError(f"{n}: cross-fit T'leri 6b json'uyla aynı değil")
        if n == CHAR_LR:
            own = {r["fold"]: r["T"] for r in runs["char_lr_folds"]}
        else:
            own = {r["fold"]: r["T"] for r in runs["bert_runs"] if r["seed"] == BERT_SEEDS[n]}
        rows = []
        for k in sorted(t_cf):
            m = o["fold"] == k
            lps = {"x:before": log_probs(o["z"][m]), "x:after": log_probs(o["z"][m], t_cf[k])}
            point, reps = bootstrap_metrics(lps, o["y"][m], np.asarray(groups)[m], B, seed)
            r = before_after(point, reps, "x")["nll"]
            rows.append({"fold": int(k), "n": int(m.sum()), "T_cross_fit": t_cf[k],
                         "T_own_in_sample": own[k], "nll_before": r["before"],
                         "nll_after": r["after"], "delta": r["delta"],
                         "delta_ci95": r["delta_ci95"]})  # fmt: skip
        total = sum(r["delta"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
        if not np.isclose(total, m6["metrics_cross_fit"]["nll"]["delta"]):
            raise ValueError(f"{n}: fold ΔNLL'leri 6b toplamıyla tutmuyor")
        out[n] = {"role": CALIB_MODELS[n], "folds": rows, "delta_total_6b": total}
    return out


def fold_breakdown_train(
    metrics_dir: Path = METRICS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    path: Path = SPLIT_PATH,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Path, dict]:
    """Sonradan, betimsel (POSTHOC_NOTE): yalnız train OOF ve 6b json'u; test npz'i okunmaz."""
    head = _check_clean("fold dökümü")
    six_b = latest_metrics("train_calibration", out_dir, phase=6)
    if six_b is None or not _is_tracked(Path(out_dir, six_b["_file"])):
        raise RuntimeError("commit'lenmiş phase6_train_calibration json'u yok: önce 6b")
    oofs = {n: bert_oof(s, metrics_dir)[0] for n, s in BERT_SEEDS.items()}
    lr = six_b["char_lr_oof"]
    p_npz = Path(preds_dir) / lr["file"]
    if sha256_file(p_npz) != lr["sha256"]:
        raise ValueError(f"{p_npz.name}: sha256 6b json'undakiyle aynı değil")
    with np.load(p_npz, allow_pickle=False) as z:
        oofs[CHAR_LR] = _sorted(z["row_id"], z["y_true"], z["scores"], z["fold_id"])
    groups = cv_groups(pd.DataFrame({"row_id": oofs[PRIMARY[0]]["row_id"]}), path)
    out = {
        "experiment": "phase6_train_folds",
        "phase": 6,
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": head,
        "split": SPLIT_NAME,
        "label": "sonradan, betimsel",
        "note": POSTHOC_NOTE,
        "train_calibration": {
            "file": six_b["_file"],
            "sha256": sha256_file(Path(out_dir, six_b["_file"])),
        },  # fmt: skip
        "bootstrap": {"B": B, "seed": seed, "unit": "split_v2 components (cv_groups), fold içi"},
        "models": fold_breakdown(oofs, six_b, groups, B, seed),
    }
    return report.write_metrics(out, "phase6_train_folds", out_dir), out


# --------------------------------------------------------------------------- 6c: test
def test_logits(name: str, scores: np.ndarray) -> tuple[np.ndarray, int]:
    """BERT: logit; char_lr: log(predict_proba), p = 0 tiny'ye kırpılır. Döner: (z, kırpılan)."""
    if name == CHAR_LR:
        tiny = np.finfo(float).tiny
        return np.log(np.maximum(scores, tiny)), int((scores < tiny).sum())
    return np.asarray(scores, dtype=float), 0


test_logits.__test__ = False  # pytest bunu test fonksiyonu sanmasın


def calibrate_test(
    metrics_dir: Path = METRICS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    fig_dir: Path = FIGURES_DIR,
    path: Path = SPLIT_PATH,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Path, dict]:
    """6c (CALIB_RULES['test_eval']): T̂_final test skorlarına bir kez uygulanır; fit yok."""
    head = _check_clean("6c")
    if list(Path(out_dir).glob("phase6_test_calibration_*.json")):
        raise RuntimeError("phase6_test_calibration json'u zaten var: 6c tek seferdir")
    tr = latest_metrics("train_calibration", out_dir, phase=6)
    if tr is None:
        raise RuntimeError("phase6_train_calibration json'u yok: önce 6b")
    tr_path = Path(out_dir, tr["_file"])
    if not _is_tracked(tr_path):
        raise RuntimeError(f"{tr['_file']} commit'lenmemiş: önce 6b sonuç commit'i")
    main_res = latest_metrics("test", out_dir, phase=5)
    if main_res is None:
        raise RuntimeError("phase5_test json'u yok")
    started = report.now()
    z, clipped, ref = {}, {}, None
    for n in CALIB_MODELS:
        p = preds_path(n, preds_dir)
        if sha256_file(p) != main_res["preds"][n]["sha256"]:
            raise ValueError(f"{p.name}: sha256 Faz 5 sonuç json'undakiyle aynı değil")
        with np.load(p, allow_pickle=False) as f:
            d = {k: f[k] for k in ("row_id", "y_true", "y_pred", "scores")}
        if ref is None:
            ref = d
        elif not (np.array_equal(d["row_id"], ref["row_id"])
                  and np.array_equal(d["y_true"], ref["y_true"])):  # fmt: skip
            raise ValueError(f"{n}: test satırları/etiketleri diğer modellerle aynı değil")
        z[n], clipped[n] = test_logits(n, d["scores"])
        if not np.array_equal(z[n].argmax(1), d["y_pred"]):
            raise ValueError(f"{n}: skorların argmax'ı kayıtlı y_pred ile aynı değil")
    y = ref["y_true"]
    groups = test_groups(pd.DataFrame({"row_id": ref["row_id"]}), path)
    lps = {}
    for n in CALIB_MODELS:
        T = tr["models"][n]["T_final"]
        lps[f"{n}:before"], lps[f"{n}:after"] = log_probs(z[n]), log_probs(z[n], T)
        if not np.array_equal(lps[f"{n}:before"].argmax(1), lps[f"{n}:after"].argmax(1)):
            raise AssertionError(f"{n}: kalibrasyon argmax'ı değiştirdi")
    point, reps = bootstrap_metrics(lps, y, groups, B, seed)
    for n in CALIB_MODELS:  # aynı w'ler: Faz 5 accuracy CI'ları birebir çıkmalı
        if not np.allclose(_pct_ci(reps[f"{n}:before"]["acc"]),
                           main_res["models"][n]["accuracy_ci95"]):  # fmt: skip
            raise ValueError(f"{n}: bootstrap ağırlıkları Faz 5 ile aynı değil")
    models, sentences = {}, []
    for n, role in CALIB_MODELS.items():
        r = before_after(point, reps, n)
        t = tr["models"][n]["T_final"]
        verdict = decide(r["nll"]["delta_ci95"], tr["models"][n]["T_final_at_bound"], "test")
        models[n] = {"role": role, "T_final": t, "metrics": r, "clipped_cells": clipped[n],
                     "verdict": verdict if role == "birincil" else None}  # fmt: skip
        key = verdict if role == "birincil" or verdict == "değerlendirilemez" else "betimsel"
        sentences.append(_sentence(key, n, r, t))
    rel = {n: {"before": reliability(lps[f"{n}:before"], y),
               "after": reliability(lps[f"{n}:after"], y)} for n in CALIB_MODELS}  # fmt: skip
    fig = Path(fig_dir) / "phase6_test_reliability.png"
    title = "Test (T̂_final, 6b'den): reliability, 15 bin; nokta alanı ~ satır sayısı"
    plot_reliability({n: rel[n] for n in PRIMARY}, fig, title)
    out = {
        "experiment": "phase6_test_calibration",
        "phase": 6,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": head,
        "split": SPLIT_NAME,
        "n_test": int(len(y)),
        "train_calibration": {"file": tr["_file"], "sha256": sha256_file(tr_path)},
        "main_results": main_res["_file"],
        "bootstrap": {"B": B, "seed": seed, "unit": "split_v2 components (test_groups)"},
        "models": models,
        "between_models": between_models(point, reps, *BETWEEN),
        "sentences": sentences,
        "reliability": rel,
        "figure": fig.name,
        "rules": CALIB_RULES,
    }
    return report.write_metrics(out, "phase6_test_calibration", out_dir, when=started), out


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["train", "folds", "test"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp1254 konsol: T̂, Δ, ≤
    if args.cmd == "folds":
        print(fold_breakdown_train()[0])
        return
    p, out = calibrate_train() if args.cmd == "train" else calibrate_test()
    print(p, *out["sentences"], sep="\n")


if __name__ == "__main__":
    main()

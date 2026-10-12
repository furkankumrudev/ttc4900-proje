"""Faz 2 karşılaştırması: CV skorları, eşleştirilmiş fold farkları ve grafik (test seti yok).

Bütün modeller aynı dış fold'ları kullanır (aynı make_cv random_state'leri; npz'lerden
doğrulanır). Bu yüzden fold bazında fark d_j = skor_A,j - skor_B,j eşleştirilmiştir: iki
modelin fold skorları güçlü korelasyonlu olduğundan Var(d) << Var(A) + Var(B).

Fold skorları bağımsız değildir (tekrar içinde train parçaları %75, tekrarlar arasında ~%80
örtüşür); düz t-testi varyansı küçük gösterir. Düzeltilmiş tekrarlı k-fold testi
(Nadeau & Bengio 2003; tekrarlı k-fold uyarlaması Bouckaert & Frank 2004):

    Var(d̄) ≈ (1/J + n_val/n_train) · s_d²,   t = d̄ / sqrt(Var(d̄)),   sd = J - 1

Varsayımlar ve sınırlar:
- Düzeltme sezgiseldir (fold'lar arası korelasyon ≈ n_val/n varsayımı; rastgele alt örnekleme
  için türetilmiştir, k-fold'a uygulanması yaygın ama yaklaşıktır).
- Kaybın örnekler üzerinde toplanabilir olduğunu varsayar; macro-F1 toplanabilir değildir.
- Karşılaştırılan, ayarlama DAHİL öğrenme prosedürüdür; donmuş final konfigürasyonlar değil.
- "En iyi" aynı CV'den argmax ile seçilip ona karşı test edilir (en iyiyle çoklu karşılaştırma,
  Hsu MCB): p-değerleri ilke olarak biraz iyimserdir.
- ci95 aralıkları marjinaldir (çoklu karşılaştırma düzeltmesiz); p_holm ise Holm düzeltmelidir.

Bu fazdaki p-değerleri keşifseldir; model seçimine yardım eder, "A, B'den iyidir" iddiası
final test fazındaki McNemar / paired bootstrap ile kurulur.

Kullanım: python -m ttc4900.cv_compare
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from ttc4900 import report
from ttc4900.paths import FIGURES_DIR, METRICS_DIR

DISPLAY = {
    "word_lr": "Kelime TF-IDF + LR",
    "char_lr": "Karakter TF-IDF + LR",
    "word_svc": "Kelime TF-IDF + LinearSVC",
    "word_cnb": "Kelime TF-IDF + ComplementNB",
}

SELECTION_RULE = (
    "Birincil klasik baseline = dış CV macro-F1 ortalaması en yüksek model. Farklar CV'de "
    "ayırt edilemese de CV'si en güçlü olan seçilir (BERTurk karşılaştırmasında 'saman adam' "
    "eleştirisini önlemek için tutucu tercih). Kural testten ÖNCE donar; test sonucuna göre "
    "'en iyi klasik' değiştirilmez."
)
TEST_PLAN = {
    "primary": "BERTurk vs birincil klasik baseline (McNemar + paired bootstrap, ayrı aile)",
    "secondary": (
        "Dört klasik modelin donmuş final.best_params konfigürasyonları betimsel raporlanır; "
        "birincil modele karşı 3 karşılaştırma ayrı bir Holm ailesi"
    ),
    "power_warning": (
        "908 test örneğiyle klasik modeller arası 0.004-0.01'lik farkların McNemar ile "
        "yakalanması beklenmez; anlamsız sonuç eşdeğerlik kanıtı sayılmaz"
    ),
    "sensitivity_c1": (
        "C1 duyarlılık analizi (kural CV'den ÖNCE donduruldu): birincil klasik baseline ham "
        "char_lr olarak KALIR. U+0091-U+0094 silinmiş char_lr_c1clean yalnızca duyarlılık "
        "analizidir; final.best_params ile teste ikincil/betimsel olarak taşınır ve ham "
        "char_lr ile birlikte raporlanır. İkisi arasında test (ya da CV) sonucuna göre seçim "
        "YAPILMAZ; BERTurk karşılaştırmasındaki referans değişmez. Testte bu çift için p "
        "hesaplanmaz: yalnızca iki modelin skorları ve farkın paired bootstrap %95 CI'ı "
        "betimsel raporlanır (hiçbir Holm ailesine girmez)."
    ),
    "expectation_note": (
        "CV skoru 2904 satırla eğitimi ölçer, final model 3630 satırla eğitilir; ayrıca seçilen "
        "modelin CV skoru 4 adayın maksimumudur (kazananın laneti). Test skorunun CV'den "
        "farklı çıkması tek başına anomali değildir."
    ),
}


def latest_metrics(name: str, metrics_dir: Path = METRICS_DIR, phase: int = 2) -> dict | None:
    """`phase<N>_<name>_<YYYYMMDD-HHMMSS>.json` dosyalarının en sonu.

    Ad tam eşleşmeli: `char_lr_*` glob'u `char_lr_c1clean_*`'ı da yakalardı.
    """
    pat = re.compile(rf"phase{phase}_{re.escape(name)}_\d{{8}}-\d{{6}}\.json")
    files = sorted(
        f for f in Path(metrics_dir).glob(f"phase{phase}_*.json") if pat.fullmatch(f.name)
    )
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8")) | {"_file": files[-1].name}


def latest_phase2(metrics_dir: Path = METRICS_DIR) -> dict[str, dict]:
    """Ana karşılaştırmadaki her model için en son phase2_<model>_<ts>.json."""
    found = {name: latest_metrics(name, metrics_dir) for name in DISPLAY}
    return {k: v for k, v in found.items() if v is not None}


def check_same_folds(metrics: dict[str, dict], metrics_dir: Path = METRICS_DIR) -> None:
    """Eşleştirme varsayımı: her modelde her satır aynı (tekrar, fold)'da doğrulanmış olmalı.

    npz, json'daki `preds_file` alanından okunur (aynı koşunun tahminleri).
    """
    ref = None
    for name, m in metrics.items():
        z = np.load(Path(metrics_dir) / m["preds_file"])
        order = np.argsort(z["row_id"])
        cur = (z["row_id"][order], z["fold_id"][:, order])
        if ref is None:
            ref = cur
        elif not (np.array_equal(ref[0], cur[0]) and np.array_equal(ref[1], cur[1])):
            raise ValueError(f"{name} farklı dış fold'lar kullanmış; eşleştirilmiş fark geçersiz.")


def _nb_scale(J: int, val_train_ratio: float) -> float:
    return float(np.sqrt(1 / J + val_train_ratio))


def nadeau_bengio(d: np.ndarray, val_train_ratio: float, alpha: float = 0.05) -> dict:
    """Düzeltilmiş tekrarlı-CV t aralığı ve iki yönlü p-değeri (d: J fold farkı)."""
    d = np.asarray(d, dtype=float)
    J = len(d)
    mean, s2 = d.mean(), d.var(ddof=1)
    se = _nb_scale(J, val_train_ratio) * np.sqrt(s2)
    tcrit = stats.t.ppf(1 - alpha / 2, J - 1)
    if se > 0:
        t = mean / se
    else:  # tüm farklar aynı: fark 0 ise t=0, değilse sonsuz
        t = 0.0 if mean == 0 else np.inf * np.sign(mean)
    return {
        "mean_diff": float(mean),
        "std_diff": float(np.sqrt(s2)),
        "se_corrected": float(se),
        "se_naive": float(np.sqrt(s2 / J)),
        "ci95": [float(mean - tcrit * se), float(mean + tcrit * se)],
        "t": float(t),
        "df": J - 1,
        "p": float(2 * stats.t.sf(abs(t), J - 1)),
        # düzeltmesiz eşleştirilmiş t (fold'ları bağımsız sayar); yalnızca karşılaştırma için
        "p_naive": _naive_p(d),
        "n_folds_positive": int((d > 0).sum()),  # betimsel; fold'lar bağımsız değil
        "J": J,
    }


def _naive_p(d: np.ndarray) -> float:
    """Düz eşleştirilmiş t-testi p'si: se = s/sqrt(J). Fold bağımlılığını yok sayar, iyimserdir."""
    J, mean, s = len(d), d.mean(), d.std(ddof=1)
    if s == 0:
        return 1.0 if mean == 0 else 0.0
    return float(2 * stats.t.sf(abs(mean / (s / np.sqrt(J))), J - 1))


def nb_mean_ci(scores: np.ndarray, val_train_ratio: float, alpha: float = 0.05) -> list[float]:
    """Tek model için N-B mantığıyla düzeltilmiş ortalama CI: x̄ ± t·sqrt(1/J + oran)·s."""
    x = np.asarray(scores, dtype=float)
    J = len(x)
    half = stats.t.ppf(1 - alpha / 2, J - 1) * _nb_scale(J, val_train_ratio) * x.std(ddof=1)
    return [float(x.mean() - half), float(x.mean() + half)]


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni düzeltilmiş p-değerleri (giriş sırasında)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _edge(grid: list, value) -> str | None:
    """Seçilen değerin grid'deki konumu: 'low' / 'high' (uç) ya da None (iç nokta).

    Grid listedeki sırayla okunur (C ve ngram_range artan yazılmıştır). İki değerli bir
    grid'de her seçim bir uçtadır; bu yüzden `grid_size` ile birlikte yorumlanmalı.
    """
    i = grid.index(value)
    if i == len(grid) - 1:
        return "high"
    return "low" if i == 0 else None


def grid_edges(metrics: dict[str, dict]) -> dict[str, dict]:
    """Her model x hiperparametre: dış fold seçim sayıları, final seçim ve uç durumu.

    Dış fold seçimleri iç CV'den gelir; final seçim tüm train'deki GridSearchCV'den.
    Seçimler grid ucunda yığılıyorsa optimum grid'in dışında olabilir.
    """
    out = {}
    for name, m in metrics.items():
        grid = m["params"]["grid"]
        folds = [f["best_params"] for f in m["cv"]["folds"]]
        final = m["final"]["best_params"]
        out[name] = {}
        for param, values in grid.items():
            chosen = [f[param] for f in folds]
            edges = [_edge(values, v) for v in chosen]
            out[name][param] = {
                "grid": values,
                "grid_size": len(values),
                "fold_counts": {json.dumps(v): chosen.count(v) for v in values},
                "n_folds": len(chosen),
                "n_folds_low_edge": edges.count("low"),
                "n_folds_high_edge": edges.count("high"),
                "final": final[param],
                "final_edge": _edge(values, final[param]),
            }
    return out


def summarize(metrics: dict[str, dict]) -> dict:
    splits = {m["split"] for m in metrics.values()}
    if len(splits) != 1:
        raise ValueError(f"Modeller farklı split'lerde: {splits}")
    any_m = next(iter(metrics.values()))
    ratio = float(np.mean([f["n_val"] / f["n_train"] for f in any_m["cv"]["folds"]]))
    rows = sorted(
        (
            {
                "model": name,
                "display": DISPLAY[name],
                "macro_f1_mean": m["cv"]["macro_f1_mean"],
                "macro_f1_std": m["cv"]["macro_f1_std"],
                "macro_f1_nb_ci95": nb_mean_ci(m["cv"]["macro_f1_folds"], ratio),
                "macro_f1_repeat_means": m["cv"]["macro_f1_repeat_means"],
                "accuracy_mean": m["cv"]["accuracy_mean"],
                "selection_bias": m["cv"]["selection_bias"],
                "final_params": m["final"]["best_params"],
                "source": m["_file"],
                "git_commit": m["git_commit"],
            }
            for name, m in metrics.items()
        ),
        key=lambda r: -r["macro_f1_mean"],
    )
    best = rows[0]["model"]
    fb = np.array(metrics[best]["cv"]["macro_f1_folds"])
    comps = []
    for r in rows[1:]:
        fo = np.array(metrics[r["model"]]["cv"]["macro_f1_folds"])
        c = {"a": best, "b": r["model"], **nadeau_bengio(fb - fo, ratio)}
        c["fold_corr"] = float(np.corrcoef(fb, fo)[0, 1])  # eşleştirmenin kazancı
        c["std_diff_if_unpaired"] = float(np.sqrt(fb.var(ddof=1) + fo.var(ddof=1)))
        comps.append(c)
    for c, p in zip(comps, holm([c["p"] for c in comps]), strict=True):
        c["p_holm"] = p
    p_table = [
        {
            "a": c["a"],
            "b": c["b"],
            "mean_diff": c["mean_diff"],
            "p_naive": c["p_naive"],
            "p_nb": c["p"],
            "p_nb_holm": c["p_holm"],
        }
        for c in comps
    ]
    tied = [c["b"] for c in comps if c["ci95"][0] <= 0]
    return {
        "experiment": "phase2_summary",
        "phase": 2,
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": splits.pop(),
        "ranking": rows,
        "paired_vs_best": comps,
        "p_table": p_table,
        "grid_edges": grid_edges(metrics),
        "val_train_ratio": ratio,
        "notes": {
            "macro_f1_std": "fold skorlarının dağılımı (ddof=1); ortalamanın belirsizliği DEĞİL",
            "macro_f1_nb_ci95": "N-B düzeltmeli ortalama CI: x̄ ± t14·sqrt(1/15 + oran)·s",
            "repeat_means": (
                "tekrar ortalamalarının yakınlığı yalnızca bölümleme gürültüsünün küçük olduğunu "
                "gösterir; CI olarak kullanılmaz"
            ),
            "ci95": "marjinal %95 (çoklu karşılaştırma düzeltmesiz); p_holm Holm düzeltmeli",
            "p_table": (
                "p_naive: düz eşleştirilmiş t (se = s/sqrt(J)); fold'lar bağımsız sayıldığı için "
                "iyimser, karar için KULLANILMAZ. p_nb: Nadeau-Bengio düzeltmeli t "
                "(se = sqrt(1/J + n_val/n_train)·s), çoklu karşılaştırma düzeltmesiz. "
                "p_nb_holm: p_nb'lere Holm (aile = en iyiye karşı m karşılaştırma); karar "
                "sütunu budur. 'p_holm' alanı p_nb_holm ile aynıdır."
            ),
            "p_007_erratum": (
                "Raporda geçen 'düzeltilmiş p≈0.07' doğru tanımların hiçbirine karşılık gelmez: "
                "char_lr-word_lr için p_naive 0.0025, p_nb 0.113, tek yönlü p_nb 0.057, "
                "p_nb_holm 0.324. Muhtemel kaynak: N-B'de yanlış oran (n_val/n=0.2) ile t yerine "
                "normal yaklaşım (≈0.066; t ile 0.087). Geçerli değerler p_table'dadır."
            ),
            "grid_edges": (
                "n_folds_*_edge: 15 dış fold'da iç CV seçiminin grid ucunda olduğu sayı; "
                "final_edge: tüm train seçimi. grid_size=2 ise her seçim bir uçtadır."
            ),
            "not_distinguishable_from_best": (
                "'ayırt edilemedi' demektir, 'eşdeğer' değil; ci95[1] dışlanamayan en büyük açık"
            ),
            "grid_history": (
                "LR grid'leri iç CV seçimleri kenarda yığıldığı için bir kez genişletildi "
                "(superseded/); dış skorlar değişmedi. char_lr'de C=100 güncel grid'de "
                "({10,100,1000}) İÇ noktadır; üst uçta olduğu grid ilk grid'di ({1,10,100}). "
                "char_lr'de kenarda kalan ngram_range=(2,4)'tür (15/15 ve final; 2 değerli grid) "
                "ama (3,5) ilk grid'de denenip 4/15'te kalmıştı. Test kullanılmadı."
            ),
        },
        "method": (
            "Eşleştirilmiş fold farkları (aynı dış fold'lar, npz'den doğrulandı), Nadeau-Bengio / "
            "Bouckaert-Frank düzeltmeli t, Holm; keşifsel, test seti kullanılmadı"
        ),
        "selection_rule": SELECTION_RULE,
        "selected_for_test": best,
        "not_distinguishable_from_best": tied,
        "test_plan": TEST_PLAN,
        "test": None,
    }


C1_PAIR = ("char_lr", "char_lr_c1clean")


def c1_sensitivity(metrics_dir: Path = METRICS_DIR) -> dict:
    """Tek karşılaştırma: ham char_lr - temiz char_lr, Nadeau-Bengio düzeltmeli (Holm yok, m=1).

    d_j > 0: C1 artefaktı fold j'de skoru artırıyor (model kısmen kısayol kullanıyor).
    """
    raw, clean = (latest_metrics(n, metrics_dir) for n in C1_PAIR)
    if raw is None or clean is None:
        raise SystemExit(f"Eksik sonuç: {C1_PAIR}")
    pair = dict(zip(C1_PAIR, (raw, clean), strict=True))
    if raw["split"] != clean["split"]:
        raise ValueError("Ham ve temiz char_lr farklı split'lerde.")
    check_same_folds(pair, metrics_dir)
    commits = {m["git_commit"] for m in pair.values()}
    if any(c.endswith("-dirty") for c in commits):
        print(f"UYARI: kirli commit'ten üretilmiş sonuç: {sorted(commits)}")
    ratio = float(np.mean([f["n_val"] / f["n_train"] for f in raw["cv"]["folds"]]))
    fr, fc = (np.array(m["cv"]["macro_f1_folds"]) for m in (raw, clean))
    res = nadeau_bengio(fr - fc, ratio)
    pc = {
        k: raw["cv"]["per_class_f1_mean"][k] - clean["cv"]["per_class_f1_mean"][k]
        for k in raw["cv"]["per_class_f1_mean"]
    }
    return {
        "experiment": "phase2_c1_sensitivity",
        "phase": 2,
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": raw["split"],
        "comparison": "char_lr (ham) - char_lr_c1clean (U+0091-U+0094 silinmiş)",
        "sources": {n: m["_file"] for n, m in pair.items()},
        "source_commits": {n: m["git_commit"] for n, m in pair.items()},
        "macro_f1_mean": {n: m["cv"]["macro_f1_mean"] for n, m in pair.items()},
        "macro_f1_std": {n: m["cv"]["macro_f1_std"] for n, m in pair.items()},
        "final_params": {n: m["final"]["best_params"] for n, m in pair.items()},
        "nadeau_bengio": res | {"fold_corr": float(np.corrcoef(fr, fc)[0, 1])},
        "per_class_f1_diff": pc,  # betimsel: belirsizlik ve çoklu karşılaştırma düzeltmesi yok
        "val_train_ratio": ratio,
        "rule": TEST_PLAN["sensitivity_c1"],
        "notes": {
            "family": "tek karşılaştırma, çoklu karşılaştırma düzeltmesi yok",
            "interpretation": (
                "CI 0'ı içeriyorsa artefaktın CV skoruna etkisi ayırt edilemedi demektir, "
                "'etkisi yok' değil",
            ),
            "per_class_f1_diff": (
                "betimsel (15 fold ortalamaları farkı); CI ve düzeltme yok, sınıf bazında "
                "'etkilendi' iddiası için kullanılmaz"
            ),
        },
        "test": None,
    }


def plot(summary: dict, metrics: dict[str, dict], path: Path) -> None:
    """Sol: 15 fold skoru + ortalama ± std. Sağ: en iyiye göre fark ve düzeltilmiş %95 CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf, hue = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    rows = summary["ranking"]
    names = [r["model"] for r in rows][::-1]  # en iyi en üstte
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1]}, facecolor=surf
    )
    rng = np.random.default_rng(0)
    for y, name in enumerate(names):
        f = np.array(metrics[name]["cv"]["macro_f1_folds"])
        jitter = rng.uniform(-0.12, 0.12, len(f))
        ax1.scatter(f, y + jitter, s=22, color=hue, alpha=0.45, linewidths=0, zorder=2)
        m, s = f.mean(), f.std(ddof=1)
        ax1.plot([m - s, m + s], [y, y], color=ink, lw=2, zorder=3, solid_capstyle="round")
        ax1.scatter([m], [y], s=64, color=ink, zorder=4, edgecolors=surf, linewidths=2)
        ax1.annotate(f"{m:.3f} ± {s:.3f}", (m, y), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8.5, color=ink2)  # fmt: skip
    ax1.set_yticks(range(len(names)), [DISPLAY[n] for n in names])
    ax1.set_xlabel("Dış fold macro-F1 (5 fold × 3 tekrar)")
    ax1.set_title(
        "CV macro-F1: nokta = fold; çizgi = ortalama ± std (fold dağılımı, CI değil)",
        loc="left", fontsize=10,
    )  # fmt: skip

    comps = {c["b"]: c for c in summary["paired_vs_best"]}
    best = summary["selected_for_test"]
    others = [n for n in names if n != best]
    for y, name in enumerate(others):
        c = comps[name]
        ax2.plot(c["ci95"], [y, y], color=hue, lw=2, solid_capstyle="round")
        ax2.scatter([c["mean_diff"]], [y], s=64, color=hue, zorder=3, edgecolors=surf,
                    linewidths=2)  # fmt: skip
        ax2.annotate(f"p_Holm={c['p_holm']:.2f}", (c["ci95"][1], y), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=ink2)  # fmt: skip
    ax2.axvline(0, color=ink2, lw=1, ls="--")
    ax2.set_yticks(range(len(others)), [DISPLAY[n] for n in others])
    ax2.set_xlabel(f"macro-F1 farkı: {DISPLAY[best]} − model")
    ax2.set_title(
        "Eşleştirilmiş fark, N–B %95 CI\n(çoklu karşılaştırma düzeltmesiz)", loc="left", fontsize=10
    )
    lo = min(c["ci95"][0] for c in comps.values())
    hi = max(c["ci95"][1] for c in comps.values())
    ax2.set_xlim(min(lo, 0) - 0.005, hi + (hi - min(lo, 0)) * 0.6)

    for ax in (ax1, ax2):
        ax.set_facecolor(surf)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=ink2, length=0)
        ax.xaxis.label.set_color(ink2)
        ax.title.set_color(ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main_c1() -> None:
    out = c1_sensitivity()
    p = report.write_metrics(out, "phase2_c1_sensitivity")
    nb = out["nadeau_bengio"]
    lo, hi = nb["ci95"]
    print(p)
    for n, v in out["macro_f1_mean"].items():
        print(f"  {n:16s} {v:.4f} ± {out['macro_f1_std'][n]:.4f}")
    print(f"  ham - temiz: {nb['mean_diff']:+.4f} N-B CI [{lo:+.4f}, {hi:+.4f}] "
          f"p_NB={nb['p']:.3g} (düz p={nb['p_naive']:.3g}) corr={nb['fold_corr']:.2f}")  # fmt: skip
    for k, v in out["per_class_f1_diff"].items():
        print(f"    {k:10s} {v:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c1-sensitivity", action="store_true", help="ham vs temiz char_lr")
    if ap.parse_args().c1_sensitivity:
        return main_c1()
    metrics = latest_phase2()
    missing = set(DISPLAY) - set(metrics)
    if missing:
        raise SystemExit(f"Eksik model sonuçları: {sorted(missing)}")
    commits = {m["git_commit"] for m in metrics.values()}
    if len(commits) != 1 or any(c.endswith("-dirty") for c in commits):
        print(f"UYARI: model json'ları farklı/kirli commit'lerden: {sorted(commits)}")
    check_same_folds(metrics)
    summary = summarize(metrics)
    p = report.write_metrics(summary, "phase2_summary")
    plot(summary, metrics, FIGURES_DIR / "phase2_cv_macro_f1.png")
    print(p)
    for r in summary["ranking"]:
        lo, hi = r["macro_f1_nb_ci95"]
        print(f"{r['display']:30s} {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} "
              f"N-B CI [{lo:.4f}, {hi:.4f}]  bias {r['selection_bias']:+.4f}")  # fmt: skip
    for c in summary["paired_vs_best"]:
        lo, hi = c["ci95"]
        print(f"  {c['a']} - {c['b']}: {c['mean_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"p_Holm={c['p_holm']:.3g} corr={c['fold_corr']:.2f}")  # fmt: skip
    print()
    print(f"{'karşılaştırma':22s} {'fark':>8s} {'düz p':>8s} {'N-B p':>8s} {'N-B+Holm':>9s}")
    for r in summary["p_table"]:
        print(f"{r['a'] + ' - ' + r['b']:22s} {r['mean_diff']:+8.4f} {r['p_naive']:8.4f} "
              f"{r['p_nb']:8.4f} {r['p_nb_holm']:9.4f}")  # fmt: skip
    print()
    print("Grid kenarları (dış fold seçimleri / final):")
    for name, params in summary["grid_edges"].items():
        for param, e in params.items():
            flag = f"final {e['final_edge']} UÇTA" if e["final_edge"] else "final iç nokta"
            print(f"  {name:9s} {param:20s} {e['fold_counts']}  final={e['final']}  {flag}"
                  f"  (grid_size={e['grid_size']})")  # fmt: skip


if __name__ == "__main__":
    main()

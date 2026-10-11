"""Faz 2 karşılaştırması: CV skorları, eşleştirilmiş fold farkları ve grafik (test seti yok).

Bütün modeller aynı dış fold'ları kullanır (aynı make_cv random_state'leri). Bu yüzden fold
bazında fark d_j = skor_A,j - skor_B,j eşleştirilmiştir; iki ayrı ortalama ± std aralığının
örtüşmesine bakmaktan daha güçlüdür.

Fold skorları bağımsız değildir (train parçaları ~%75 örtüşür), düz t-testi varyansı küçük
gösterir. Nadeau & Bengio (2003) düzeltmesi:

    Var(d̄) ≈ (1/J + n_val/n_train) · s_d²,   t = d̄ / sqrt(Var(d̄)),   sd = J - 1

Bu fazdaki p-değerleri keşifseldir; model seçimine yardım eder, "A, B'den iyidir" iddiası
final test fazındaki McNemar / paired bootstrap ile kurulur.

Kullanım: python -m ttc4900.cv_compare
"""

from __future__ import annotations

import json
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


def latest_phase2(metrics_dir: Path = METRICS_DIR) -> dict[str, dict]:
    """Her model için en son phase2_<model>_<ts>.json (zaman damgası ada gömülü)."""
    out = {}
    for name in DISPLAY:
        files = sorted(Path(metrics_dir).glob(f"phase2_{name}_*.json"))
        if files:
            out[name] = json.loads(files[-1].read_text(encoding="utf-8")) | {
                "_file": files[-1].name
            }
    return out


def check_same_folds(names: list[str], metrics_dir: Path = METRICS_DIR) -> None:
    """Eşleştirme varsayımı: her modelde her satır aynı (tekrar, fold)'da doğrulanmış olmalı."""
    ref = None
    for name in names:
        z = np.load(Path(metrics_dir) / "preds" / f"cv_{name}.npz")
        order = np.argsort(z["row_id"])
        cur = (z["row_id"][order], z["fold_id"][:, order])
        if ref is None:
            ref = cur
        elif not (np.array_equal(ref[0], cur[0]) and np.array_equal(ref[1], cur[1])):
            raise ValueError(f"{name} farklı dış fold'lar kullanmış; eşleştirilmiş fark geçersiz.")


def nadeau_bengio(d: np.ndarray, val_train_ratio: float, alpha: float = 0.05) -> dict:
    """Düzeltilmiş tekrarlı-CV t aralığı ve iki yönlü p-değeri (d: J fold farkı)."""
    d = np.asarray(d, dtype=float)
    J = len(d)
    mean, s2 = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + val_train_ratio) * s2)
    se_naive = np.sqrt(s2 / J)
    tcrit = stats.t.ppf(1 - alpha / 2, J - 1)
    t = mean / se if se > 0 else np.inf * np.sign(mean)
    return {
        "mean_diff": float(mean),
        "std_diff": float(np.sqrt(s2)),
        "se_corrected": float(se),
        "se_naive": float(se_naive),
        "ci95": [float(mean - tcrit * se), float(mean + tcrit * se)],
        "t": float(t),
        "df": J - 1,
        "p": float(2 * stats.t.sf(abs(t), J - 1)),
        "n_folds_positive": int((d > 0).sum()),
        "J": J,
    }


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


def summarize(metrics: dict[str, dict]) -> dict:
    rows = sorted(
        (
            {
                "model": name,
                "display": DISPLAY[name],
                "macro_f1_mean": m["cv"]["macro_f1_mean"],
                "macro_f1_std": m["cv"]["macro_f1_std"],
                "accuracy_mean": m["cv"]["accuracy_mean"],
                # aynı eğitim boyutu (5 fold, train'in %80'i): seçim yanlılığının tahmini
                "selection_bias": m["final"]["cv_best_score_optimistic"] - m["cv"]["macro_f1_mean"],
                "final_params": m["final"]["best_params"],
                "source": m["_file"],
            }
            for name, m in metrics.items()
        ),
        key=lambda r: -r["macro_f1_mean"],
    )
    best = rows[0]["model"]
    fb = np.array(metrics[best]["cv"]["macro_f1_folds"])
    folds = metrics[best]["cv"]["folds"]
    ratio = float(np.mean([f["n_val"] / f["n_train"] for f in folds]))
    comps = []
    for r in rows[1:]:
        d = fb - np.array(metrics[r["model"]]["cv"]["macro_f1_folds"])
        comps.append({"a": best, "b": r["model"], **nadeau_bengio(d, ratio)})
    for c, p in zip(comps, holm([c["p"] for c in comps]), strict=True):
        c["p_holm"] = p
    tied = [c["b"] for c in comps if c["ci95"][0] <= 0]
    splits = {m["split"] for m in metrics.values()}
    if len(splits) != 1:
        raise ValueError(f"Modeller farklı split'lerde: {splits}")
    return {
        "experiment": "phase2_summary",
        "phase": 2,
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": splits.pop(),
        "ranking": rows,
        "paired_vs_best": comps,
        "val_train_ratio": ratio,
        "method": (
            "Eşleştirilmiş fold farkları (aynı dış fold'lar, doğrulandı), Nadeau-Bengio düzeltmeli "
            "t aralığı, Holm düzeltmesi; keşifsel, test seti kullanılmadı"
        ),
        "selected_for_test": best,
        "not_distinguishable_from_best": tied,
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
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]}, facecolor=surf
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
    ax1.set_title("CV macro-F1: noktalar = fold, çizgi = ortalama ± std", loc="left", fontsize=10)

    comps = {c["b"]: c for c in summary["paired_vs_best"]}
    best = summary["selected_for_test"]
    others = [n for n in names if n != best]
    for y, name in enumerate(others):
        c = comps[name]
        ax2.plot(c["ci95"], [y, y], color=hue, lw=2, solid_capstyle="round")
        ax2.scatter([c["mean_diff"]], [y], s=64, color=hue, zorder=3, edgecolors=surf,
                    linewidths=2)  # fmt: skip
        ax2.annotate(f"p_Holm={c['p_holm']:.3f}", (c["ci95"][1], y), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=ink2)  # fmt: skip
    ax2.axvline(0, color=ink2, lw=1, ls="--")
    ax2.set_yticks(range(len(others)), [DISPLAY[n] for n in others])
    ax2.set_xlabel(f"{DISPLAY[best]} − model (macro-F1 farkı)")
    ax2.set_title("Eşleştirilmiş fark, Nadeau–Bengio %95 CI", loc="left", fontsize=10)
    lo = min(c["ci95"][0] for c in comps.values())
    hi = max(c["ci95"][1] for c in comps.values())
    ax2.set_xlim(min(lo, 0) - 0.005, hi + (hi - min(lo, 0)) * 0.45)

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


def main() -> None:
    metrics = latest_phase2()
    missing = set(DISPLAY) - set(metrics)
    if missing:
        raise SystemExit(f"Eksik model sonuçları: {sorted(missing)}")
    check_same_folds(list(metrics))
    summary = summarize(metrics)
    p = report.write_metrics(summary, "phase2_summary")
    plot(summary, metrics, FIGURES_DIR / "phase2_cv_macro_f1.png")
    print(p)
    for r in summary["ranking"]:
        print(f"{r['display']:32s} {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}")
    for c in summary["paired_vs_best"]:
        lo, hi = c["ci95"]
        print(f"  {c['a']} - {c['b']}: {c['mean_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"p_Holm={c['p_holm']:.3g}")  # fmt: skip


if __name__ == "__main__":
    main()

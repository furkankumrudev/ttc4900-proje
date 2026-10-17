"""Faz 7: betimsel hata analizi. Kurallar `ERROR_RULES`'tadır; 7a commit'inde, analiz koşulmadan
donduruldu. İnsan okuması: docs/faz7_protokol.md.

Bu modül model FIT ETMEZ ve tahmin ÜRETMEZ. Yalnız Faz 5 test npz'lerini (sha256'ları Faz 5
manifest'iyle doğrulanır; tutmazsa durur), 6b'nin T̂_final değerlerini, split_v2 bileşenlerini ve
test metnini okur. Test metni yalnız sayısal özellikler (C1 varlığı, uzunluk) için kullanılır;
sonuç json'una metin yazılmaz, yalnız row_id yazılır. p hesaplanmaz.

    python -m ttc4900.error_analysis   # 7b: temiz commit'te, tek koşu
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ttc4900 import report
from ttc4900.baselines import C1_QUOTES
from ttc4900.calibration import log_probs, test_logits
from ttc4900.cv_compare import latest_metrics
from ttc4900.data import SPLIT_PATH, load_split, load_train_test, test_groups
from ttc4900.final_eval import (
    PREDS_DIR,
    _is_tracked,
    _untracked_inputs,
    preds_path,
    sha256_file,
)
from ttc4900.paired_test import _pct_ci, bootstrap_weights, confusion, group_index
from ttc4900.paths import FIGURES_DIR, METRICS_DIR
from ttc4900.test_plan import BOOTSTRAP_B, BOOTSTRAP_SEED

SPLIT_NAME = Path(SPLIT_PATH).stem
PRIMARY = ("berturk_ft", "char_lr")
C1_ONLY = "char_lr_c1clean"  # yalnız C1 maddesinde
MODELS = (*PRIMARY, C1_ONLY)
HIGH_CONF = 0.90
HIGH_CONF_SENS = 0.95
MIN_N_CI = 20
MAX_EXAMPLES = 2
EXCERPT_CHARS = 160
NAMED_PAIRS = (("ekonomi", "teknoloji"), ("siyaset", "dunya"))
TRAIN_OOF = {
    "berturk_ft": ("berturk_ft", 4),
    "char_lr": ("char_lr", 2),
}  # (latest_metrics adı, faz)

ERROR_RULES = {
    "status": (
        "7a commit'inde, analiz koşulmadan donduruldu. Tamamı betimseldir: p yok, karar yok; "
        "hiçbir model, ayar ya da Faz 1-6 kuralı değişmez."
    ),
    "access": (
        "Test metnine erişim yalnız bu modüle açıldı (load_train_test AST kilidine tek istisna). "
        "Modül model fit etmez, tahmin üretmez: yalnız Faz 5 test npz'lerini okur ve sha256'larını "
        "Faz 5 manifest'iyle (phase5_test_preds_<ts>.json) doğrular; tutmazsa durur. Skorların "
        "argmax'ı kayıtlı y_pred ile, etiketler test metninin etiketiyle aynı olmalıdır."
    ),
    "models": ("Birincil: berturk_ft (seed 42) ve char_lr. char_lr_c1clean yalnız C1 maddesinde."),
    "confidence": (
        "Kalibre güven = max_k softmax(z / T)_k; T = 6b json'undaki T̂_final (berturk_ft 1.283, "
        "char_lr 0.916; 6c'de kullanılanla aynı). char_lr'de z = log(predict_proba). Yüksek "
        f"güvenli hata: yanlış tahmin ve kalibre güven ≥ {HIGH_CONF}; duyarlılık için "
        f"≥ {HIGH_CONF_SENS} (betimsel)."
    ),
    "uncertainty": (
        "Oranlar için Faz 5 ile aynı grup bootstrap'i: split_v2 test bileşenleri, "
        f"B = {BOOTSTRAP_B}, seed {BOOTSTRAP_SEED} (paired_test.bootstrap_weights); %95 yüzdelik. "
        "Kod, accuracy CI'larının Faz 5 json'uyla birebir aynı çıktığını denetler. Oran = "
        "Σ_g w_g k_g / Σ_g w_g n_g; paydası 0 olan replikatlar düşer ve sayısı yazılır. "
        f"n < {MIN_N_CI} olan hücrelerde CI verilmez, yalnız sayı yazılır; iki oranın farkında "
        f"iki hücrenin de n ≥ {MIN_N_CI} olması gerekir. Sayımlar (karışıklık matrisi, sınıf "
        "çiftleri) için CI yok. p yok."
    ),
    "items": (
        "(1) Test karışıklık matrisleri: sayı ve satır normalize. (2) Sınıf çiftleri: 21 sırasız "
        "çiftin tümü simetrik karışma sayısına göre sıralanır; ekonomi-teknoloji ve siyaset-dunya "
        "sıralamadaki yerleriyle raporlanır (seçici raporlama yok). (3) BERT ile char_lr'nin "
        "ayrıştığı satırlar (n10: BERT doğru/char_lr yanlış, n01: tersi): gerçek sınıf, yanlış "
        "modelin tahmini, iki modelin kalibre güveni, karakter uzunluğu, C1 varlığı. (4) Yüksek "
        "güvenli hatalar: model başına sayı, hatalar içindeki pay, sınıf çiftleri, çok satırlı "
        "yakın-tekrar grubu / karışık etiketli grup üyeliği, C1 varlığı; iki modelin birlikte "
        "yüksek güvenle yanıldığı satırlar. (5) Karışık etiketli gruplar (split_v2 bileşenlerinde "
        "etiketi farklı satır bulunan gruplar; train+test): satır başına taraf, etiket, tahmin "
        "(test: final modeller; train: OOF — BERT seed 42 CV, char_lr tekrar 0). Etiketlere hüküm "
        "verilmez, yeniden etiketleme yok. (6) C1 (U+0091-U+0094, baselines.C1_QUOTES): C1 içeren "
        "ve içermeyen test satırlarında model başına hata oranı ve farkı, sınıf içinde de "
        "(C1 sınıfla karışık); char_lr ile char_lr_c1clean'in ayrıştığı satırlarda C1 varlığı."
    ),
    "examples": (
        f"Kategori başına en çok {MAX_EXAMPLES} row_id, deterministik: yanılan modelin kalibre "
        "güveni azalan, eşitlikte row_id artan (karışık etiketli test satırlarında row_id artan). "
        "Elle seçim yok."
    ),
    "text": (
        "Sonuç json'una metin yazılmaz, yalnız row_id. Belgede kısaltılmış alıntılar "
        f"(≤ {EXCERPT_CHARS} karakter), kategori başına en çok {MAX_EXAMPLES}; veri seti kaynağı "
        "ve dataset card'daki lisans notu (license: unknown) belgede belirtilir."
    ),
    "commits": (
        "7a: kurallar + kod + testler + protokol (sayı yok). 7b: temiz commit'te koşu, json + "
        "figür, yorumsuz. 7c: yorum, README, denetçiler (kullanıcı onayıyla)."
    ),
}


# --------------------------------------------------------------------------- girdiler
def verify_preds(manifest: dict, preds_dir: Path = PREDS_DIR, models=MODELS) -> dict[str, str]:
    """ERROR_RULES['access']: npz sha256'ları Faz 5 manifest'iyle aynı olmalı; değilse durur."""
    out = {}
    for n in models:
        p = preds_path(n, preds_dir)
        sha = sha256_file(p)
        if sha != manifest["files"][n]["sha256"]:
            raise ValueError(f"{p.name}: sha256 Faz 5 manifest'iyle aynı değil; analiz durdu")
        out[n] = sha
    return out


def load_preds(preds_dir: Path = PREDS_DIR, models=MODELS) -> dict[str, dict]:
    out = {}
    for n in models:
        with np.load(preds_path(n, preds_dir), allow_pickle=False) as z:
            out[n] = {k: z[k] for k in ("row_id", "y_true", "y_pred", "scores")}
    return out


def calibrated_conf(name: str, scores: np.ndarray, T: float) -> np.ndarray:
    z, _ = test_logits(name, scores)
    return np.exp(log_probs(z, T)).max(1)


# --------------------------------------------------------------------------- oranlar
class Rates:
    """Faz 5 ile aynı w'lerle oran replikatları (Σ w k / Σ w n); n < MIN_N_CI ise CI yok."""

    def __init__(self, groups, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
        self.inv, self.G = group_index(groups)
        self.W = np.vstack([w for _, w in bootstrap_weights(self.G, B, seed)]).astype(float)

    def _reps(self, k_rows, mask) -> np.ndarray:
        m = np.asarray(mask, dtype=float)
        k = np.bincount(self.inv, weights=np.asarray(k_rows, dtype=float) * m, minlength=self.G)
        n = np.bincount(self.inv, weights=m, minlength=self.G)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.W @ k) / (self.W @ n)

    def rate(self, k_rows, mask) -> dict:
        mask = np.asarray(mask, dtype=bool)
        k_rows = np.asarray(k_rows, dtype=bool)
        n, k = int(mask.sum()), int((k_rows & mask).sum())
        out = {"k": k, "n": n, "rate": k / n if n else None, "ci95": None}
        if n >= MIN_N_CI:
            r = self._reps(k_rows, mask)
            ok = np.isfinite(r)
            out |= {"ci95": _pct_ci(r[ok]), "n_dropped_replicates": int((~ok).sum())}
        else:
            out["ci_note"] = f"n < {MIN_N_CI}: CI verilmez"
        return out

    def diff(self, k_rows, mask_a, mask_b) -> dict:
        a, b = self.rate(k_rows, mask_a), self.rate(k_rows, mask_b)
        out = {"a": a, "b": b, "diff": None, "ci95": None}
        if a["n"] and b["n"]:
            out["diff"] = a["rate"] - b["rate"]
        if a["n"] >= MIN_N_CI and b["n"] >= MIN_N_CI:
            r = self._reps(k_rows, mask_a) - self._reps(k_rows, mask_b)
            ok = np.isfinite(r)
            out |= {"ci95": _pct_ci(r[ok]), "n_dropped_replicates": int((~ok).sum())}
        else:
            out["ci_note"] = f"bir hücrede n < {MIN_N_CI}: CI verilmez"
        return out


# --------------------------------------------------------------------------- maddeler
def confusion_item(y, pred, names) -> dict:
    C = confusion(y, pred, len(names))
    rows = C.sum(1, keepdims=True)
    norm = np.divide(C, rows, out=np.zeros(C.shape), where=rows > 0)
    return {"labels": names, "counts": C.tolist(), "row_normalized": norm.tolist()}


def pairs_item(y, pred, names) -> dict:
    C = confusion(y, pred, len(names))
    errors = int(C.sum() - np.trace(C))
    rows = []
    for i, j in combinations(range(len(names)), 2):
        tot = int(C[i, j] + C[j, i])
        share = tot / errors if errors else None
        rows.append({"a": names[i], "b": names[j], "a_to_b": int(C[i, j]), "b_to_a": int(C[j, i]),
                     "total": tot, "share_of_errors": share})  # fmt: skip
    rows.sort(key=lambda r: (-r["total"], r["a"], r["b"]))
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    named = []
    for a, b in NAMED_PAIRS:
        r = next(r for r in rows if {r["a"], r["b"]} == {a, b})
        named.append({"pair": [a, b], "rank": r["rank"], "total": r["total"]})
    return {"n_errors": errors, "pairs": rows, "named_pairs": named}


def _summ(x) -> dict | None:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3)}


def _counts(idx, names) -> dict:
    c = np.bincount(np.asarray(idx, dtype=int), minlength=len(names))
    return dict(zip(names, c.tolist(), strict=True))


def disagreement_item(d: dict, rates: Rates, names) -> dict:
    y, pb, pc = d["y"], d["pred"]["berturk_ft"], d["pred"]["char_lr"]
    out = {}
    for key, mask, wrong in (
        ("bert_right_char_wrong", (pb == y) & (pc != y), "char_lr"),
        ("bert_wrong_char_right", (pb != y) & (pc == y), "berturk_ft"),
    ):
        out[key] = {
            "n": int(mask.sum()),
            "row_ids": d["row_id"][mask].tolist(),
            "true_class": _counts(y[mask], names),
            "wrong_model": wrong,
            "wrong_model_pred": _counts(d["pred"][wrong][mask], names),
            "conf_berturk_ft": _summ(d["conf"]["berturk_ft"][mask]),
            "conf_char_lr": _summ(d["conf"]["char_lr"][mask]),
            "length_chars": _summ(d["length"][mask]),
            "c1": rates.rate(d["c1"], mask),
        }
    out["all_rows_reference"] = {
        "length_chars": _summ(d["length"]),
        "c1": rates.rate(d["c1"], np.ones(len(y), dtype=bool)),
    }
    return out


def high_conf_item(d: dict, rates: Rates, names, multi: np.ndarray, mixed: np.ndarray) -> dict:
    y = d["y"]
    out = {}
    for n in PRIMARY:
        wrong = d["pred"][n] != y
        per = {"n_errors": int(wrong.sum())}
        for thr in (HIGH_CONF, HIGH_CONF_SENS):
            m = wrong & (d["conf"][n] >= thr)
            pairs = pd.Series([f"{names[a]}→{names[b]}" for a, b in zip(y[m], d["pred"][n][m],
                                                                         strict=True)])  # fmt: skip
            per[f"conf_ge_{thr:.2f}"] = {
                "n": int(m.sum()),
                "share_of_errors": rates.rate(m, wrong),
                "true_to_pred": {k: int(v) for k, v in pairs.value_counts().items()},
                "in_multi_row_group": int((m & multi).sum()),
                "in_mixed_label_group": int((m & mixed).sum()),
                "c1": int((m & d["c1"]).sum()),
                "row_ids": d["row_id"][m].tolist(),
            }
        out[n] = per
    both = np.ones(len(y), dtype=bool)
    for n in PRIMARY:
        both &= (d["pred"][n] != y) & (d["conf"][n] >= HIGH_CONF)
    out["both_wrong_high_conf"] = {
        "n": int(both.sum()),
        "same_wrong_label": int((both & (d["pred"]["berturk_ft"] == d["pred"]["char_lr"])).sum()),
        "row_ids": d["row_id"][both].tolist(),
    }
    return out


def mixed_groups(components, labels: dict[int, int]) -> list[list[int]]:
    """split_v2 bileşenlerinden etiketi farklı satır içerenler (row_id listeleri, sıralı)."""
    return [sorted(map(int, c)) for c in components if len({labels[int(i)] for i in c}) > 1]


def mixed_item(groups, test_ids: set, labels, test_pred: dict, oof_pred: dict, names) -> list:
    out = []
    for g in groups:
        rows = []
        for r in g:
            side = "test" if r in test_ids else "train"
            src = test_pred if side == "test" else oof_pred
            rows.append({"row_id": r, "side": side, "label": names[labels[r]],
                         "pred_source": "final model" if side == "test" else "OOF",
                         **{f"pred_{n}": names[src[n][r]] for n in PRIMARY}})  # fmt: skip
        out.append({"group_id": g[0], "n": len(g), "rows": rows})
    return out


def c1_item(d: dict, rates: Rates, names) -> dict:
    y, c1 = d["y"], d["c1"]
    out = {"n_c1": int(c1.sum()), "n_no_c1": int((~c1).sum()), "by_model": {}}
    for n in MODELS:
        err = d["pred"][n] != y
        by_class = {}
        for k, cls in enumerate(names):
            in_k = y == k
            by_class[cls] = rates.diff(err, in_k & c1, in_k & ~c1)
        out["by_model"][n] = {"error_c1_minus_no_c1": rates.diff(err, c1, ~c1),
                              "by_class": by_class}  # fmt: skip
    pc, pk = d["pred"]["char_lr"], d["pred"][C1_ONLY]
    dis = pc != pk
    out["char_lr_vs_c1clean"] = {
        "n_disagree": int(dis.sum()),
        "n_disagree_with_c1": int((dis & c1).sum()),
        "char_lr_right_c1clean_wrong": {"n": int((dis & (pc == y)).sum()),
                                        "n_c1": int((dis & (pc == y) & c1).sum())},
        "c1clean_right_char_lr_wrong": {"n": int((dis & (pk == y)).sum()),
                                        "n_c1": int((dis & (pk == y) & c1).sum())},
        "both_wrong": {"n": int((dis & (pc != y) & (pk != y)).sum()),
                       "n_c1": int((dis & (pc != y) & (pk != y) & c1).sum())},
        "row_ids": d["row_id"][dis].tolist(),
    }  # fmt: skip
    return out


def _top(row_ids, key, mask, k: int = MAX_EXAMPLES) -> list[int]:
    idx = np.flatnonzero(mask)
    order = sorted(idx, key=lambda i: (-key[i], row_ids[i]))
    return [int(row_ids[i]) for i in order[:k]]


def examples_item(d: dict, mixed: np.ndarray) -> dict:
    y, rid, conf, pred = d["y"], d["row_id"], d["conf"], d["pred"]
    pb, pc, pk = pred["berturk_ft"], pred["char_lr"], pred[C1_ONLY]
    zero = np.zeros(len(y))
    return {
        "bert_right_char_wrong": _top(rid, conf["char_lr"], (pb == y) & (pc != y)),
        "bert_wrong_char_right": _top(rid, conf["berturk_ft"], (pb != y) & (pc == y)),
        "high_conf_error_berturk_ft": _top(rid, conf["berturk_ft"],
                                           (pb != y) & (conf["berturk_ft"] >= HIGH_CONF)),
        "high_conf_error_char_lr": _top(rid, conf["char_lr"],
                                        (pc != y) & (conf["char_lr"] >= HIGH_CONF)),
        "both_wrong_high_conf": _top(rid, (conf["berturk_ft"] + conf["char_lr"]) / 2,
                                     (pb != y) & (pc != y) & (conf["berturk_ft"] >= HIGH_CONF)
                                     & (conf["char_lr"] >= HIGH_CONF)),
        "mixed_label_test": _top(rid, zero, mixed),
        "c1_char_lr_vs_c1clean": _top(rid, conf["char_lr"], (pc != pk) & d["c1"]),
    }  # fmt: skip


def analyze(
    test: pd.DataFrame,
    preds: dict[str, dict],
    T: dict[str, float],
    components,
    labels: dict[int, int],
    oof_pred: dict[str, dict[int, int]],
    names: list[str],
    groups,
    ref_accuracy_ci: dict[str, list[float]],
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Çekirdek: test (row_id, label, text) npz'lerle aynı sırada olmalı. Döner: json gövdesi."""
    rid = test["row_id"].to_numpy()
    y = test["label"].to_numpy()
    for n, p in preds.items():
        if not (np.array_equal(p["row_id"], rid) and np.array_equal(p["y_true"], y)):
            raise ValueError(f"{n}: npz satırları/etiketleri test metniyle aynı değil")
        if not np.array_equal(test_logits(n, p["scores"])[0].argmax(1), p["y_pred"]):
            raise ValueError(f"{n}: skorların argmax'ı kayıtlı y_pred ile aynı değil")
    rates = Rates(groups, B, seed)
    for n in PRIMARY:  # aynı w'ler: Faz 5 accuracy CI'ları birebir çıkmalı
        r = rates.rate(preds[n]["y_pred"] == y, np.ones(len(y), dtype=bool))
        if not np.allclose(r["ci95"], ref_accuracy_ci[n]):
            raise ValueError(f"{n}: bootstrap ağırlıkları Faz 5 ile aynı değil")
    d = {
        "row_id": rid,
        "y": y,
        "pred": {n: preds[n]["y_pred"] for n in preds},
        "conf": {n: calibrated_conf(n, preds[n]["scores"], T[n]) for n in PRIMARY},
        "c1": test["text"].map(lambda t: bool(C1_QUOTES.search(t))).to_numpy(),
        "length": test["text"].str.len().to_numpy(),
    }
    mg = mixed_groups(components, labels)
    in_mixed = {r for g in mg for r in g}
    in_multi = {int(r) for c in components for r in c}
    mixed = np.isin(rid, list(in_mixed))
    multi = np.isin(rid, list(in_multi))
    test_pred = {n: dict(zip(rid.tolist(), preds[n]["y_pred"].tolist(), strict=True))
                 for n in PRIMARY}  # fmt: skip
    return {
        "confusion": {n: confusion_item(y, d["pred"][n], names) for n in PRIMARY},
        "pairs": {n: pairs_item(y, d["pred"][n], names) for n in PRIMARY},
        "disagreement": disagreement_item(d, rates, names),
        "high_conf_errors": high_conf_item(d, rates, names, multi, mixed),
        "mixed_label_groups": {
            "n_groups": len(mg),
            "n_test_rows": int(mixed.sum()),
            "groups": mixed_item(mg, set(rid.tolist()), labels, test_pred, oof_pred, names),
        },
        "c1": c1_item(d, rates, names),
        "examples": examples_item(d, mixed),
    }


# --------------------------------------------------------------------------- belge yardımcısı
def excerpt(text: str, n: int = EXCERPT_CHARS) -> str:
    """7c belgesi için: boşlukları sadeleştirir, C1'i görünür işaretle değiştirir, ≤ n karakter."""
    t = " ".join(C1_QUOTES.sub("¤", text).split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


# --------------------------------------------------------------------------- figür
def plot_confusion(conf: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, surf = "#0b0b0b", "#fcfcfb"
    fig, axes = plt.subplots(1, len(conf), figsize=(6.2 * len(conf), 5.4), facecolor=surf)
    for ax, (name, c) in zip(np.atleast_1d(axes), conf.items(), strict=True):
        R, C, lab = np.array(c["row_normalized"]), np.array(c["counts"]), c["labels"]
        ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(lab)):
            for j in range(len(lab)):
                if C[i, j]:
                    ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=8.5,
                            color="white" if R[i, j] > 0.5 else ink)  # fmt: skip
        ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(lab)), lab, fontsize=8.5)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.set_title(f"{name}: sayı; renk = satır oranı", loc="left", fontsize=10, color=ink)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


# --------------------------------------------------------------------------- 7b
def _oof_preds(metrics_dir: Path) -> dict[str, dict[int, int]]:
    """Train satırları için OOF tahminleri: BERT seed 42 CV (Faz 4), char_lr tekrar 0 (Faz 2)."""
    out = {}
    for n, (name, phase) in TRAIN_OOF.items():
        m = latest_metrics(name, metrics_dir, phase=phase)
        with np.load(Path(metrics_dir, m["preds_file"]), allow_pickle=False) as z:
            out[n] = dict(zip(z["row_id"].tolist(), z["y_pred"][0].tolist(), strict=True))
    return out


def run(
    metrics_dir: Path = METRICS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    fig_dir: Path = FIGURES_DIR,
    path: Path = SPLIT_PATH,
    load_data=load_train_test,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Path, dict]:
    head = report.git_commit()
    if head == "unknown" or head.endswith("-dirty"):
        raise RuntimeError(f"çalışma ağacı temiz değil ({head}); 7b temiz commit'te koşar")
    untracked = _untracked_inputs()
    if untracked:
        raise RuntimeError(f"izlenmeyen girdi dosyaları var: {untracked}; 7b koşmaz")
    manifest = latest_metrics("test_preds", metrics_dir, phase=5)
    main_res = latest_metrics("test", metrics_dir, phase=5)
    six_b = latest_metrics("train_calibration", metrics_dir, phase=6)
    for m in (manifest, main_res, six_b):
        if m is None or not _is_tracked(Path(metrics_dir, m["_file"])):
            raise RuntimeError("Faz 5 manifest'i, Faz 5 sonuç json'u ve 6b json'u commit'li olmalı")
    shas = verify_preds(manifest, preds_dir)  # tutmazsa durur; metin okunmadan önce
    preds = load_preds(preds_dir)
    split = load_split(path)
    names = split["label_names"]
    started = report.now()
    train, test = load_data(path)
    test = test.set_index("row_id").loc[preds["berturk_ft"]["row_id"]].reset_index()
    labels = dict(zip(train["row_id"].tolist(), train["label"].tolist(), strict=True))
    labels |= dict(zip(test["row_id"].tolist(), test["label"].tolist(), strict=True))
    groups = test_groups(test[["row_id"]], path)
    T = {n: six_b["models"][n]["T_final"] for n in PRIMARY}
    body = analyze(test, preds, T, split["components"], labels, _oof_preds(metrics_dir), names,
                   groups, {n: main_res["models"][n]["accuracy_ci95"] for n in PRIMARY}, B,
                   seed)  # fmt: skip
    fig = Path(fig_dir) / "phase7_confusion.png"
    plot_confusion(body["confusion"], fig)
    out = {
        "experiment": "phase7_error_analysis",
        "phase": 7,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": head,
        "split": SPLIT_NAME,
        "n_test": len(test),
        "sources": {
            "manifest": manifest["_file"],
            "preds_sha256": shas,
            "main_results": main_res["_file"],
            "train_calibration": six_b["_file"],
            "T_final": T,
        },
        "bootstrap": {
            "B": B,
            "seed": seed,
            "unit": "split_v2 components (test_groups)",
            "min_n_ci": MIN_N_CI,
        },  # fmt: skip
        **body,
        "figure": fig.name,
        "rules": ERROR_RULES,
    }
    return report.write_metrics(out, "phase7_error_analysis", out_dir, when=started), out


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p, out = run()
    print(p)
    print(json.dumps({k: out[k] for k in ("n_test", "sources")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

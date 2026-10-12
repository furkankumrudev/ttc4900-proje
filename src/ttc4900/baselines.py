"""Faz 2: klasik baseline'lar, YALNIZCA train üzerinde, grup-bilinçli iç içe (nested) CV ile.

Neden nested: hiperparametreyi seçen CV'nin skoru, seçim yanlılığı yüzünden iyimserdir
(en iyi kombinasyon gürültüden de faydalanır). Bu yüzden:

    dış CV  : make_cv(train, random_state=42+r), 5 fold x n_repeats -> raporlanan skor
    iç CV   : her dış train fold'unda make_cv(..., n_splits=3) + GridSearchCV -> seçim
    final   : tüm train'de make_cv (5 fold) + GridSearchCV -> donan parametreler

TF-IDF Pipeline içinde olduğundan her fit yalnızca o anki train parçasında yapılır.
Test seti bu modülde hiç yüklenmez.

Kullanım: python -m ttc4900.baselines --model word_lr --n-jobs 4
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from ttc4900 import report
from ttc4900.data import RANDOM_STATE, SPLIT_PATH, load_split, load_train, make_cv
from ttc4900.paths import METRICS_DIR

OUTER_SPLITS = 5
INNER_SPLITS = 3  # iç CV yalnızca sıralama yapar; 3 fold yeter ve maliyeti ~%40 azaltır
N_REPEATS = 3

# Sabit TF-IDF ayarları (taranmaz). Veride büyük harf yok, Türkçe İ/I sorunu oluşmaz.
#   sublinear_tf: tf -> 1 + ln tf; uzun haberde tekrarlanan kelime ağırlığı şişmesin.
#   min_df=2: tek belgede geçen terimler gürültü; df fold'un train kısmından hesaplanır.
TFIDF_FIXED = {"sublinear_tf": True, "min_df": 2}


@dataclass(frozen=True)
class Spec:
    make: Callable[[], Pipeline]
    grid: dict[str, list]
    fixed: dict


def _pipe(clf, **tfidf) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_FIXED, **tfidf)), ("clf", clf)])


# cp1252 akıllı tırnakların (‘ ’ “ ”) yanlış çözülmüş hali: U+0091-U+0094 kontrol karakterleri.
# Sınıfla güçlü ilişkili bir kaynak artefaktı (outputs/leakage/README.md). Duyarlılık analizi
# için YALNIZCA bu dört karakter silinir; başka hiçbir normalizasyon yapılmaz.
C1_QUOTES = re.compile(r"[\u0091-\u0094]")


def strip_c1_quotes(texts) -> list[str]:
    """Durumsuz (veriden öğrenmeyen) temizlik: fold'un neresinde uygulandığı fark etmez."""
    return [C1_QUOTES.sub("", t) for t in texts]


def _clean_pipe(clf, **tfidf) -> Pipeline:
    return Pipeline([("clean", FunctionTransformer(strip_c1_quotes)), *_pipe(clf, **tfidf).steps])


SPECS: dict[str, Spec] = {
    "word_lr": Spec(
        make=lambda: _pipe(LogisticRegression(max_iter=3000)),
        # v1 grid {1,10,100}: C=100 14/15 fold'da üst kenardaydı -> yukarı kaydırıldı
        grid={"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [10.0, 100.0, 1000.0]},
        fixed={**TFIDF_FIXED, "analyzer": "word", "clf": "LogisticRegression(lbfgs, L2)"},
    ),
    "char_lr": Spec(
        make=lambda: _pipe(LogisticRegression(max_iter=3000), analyzer="char_wb"),
        # v1 grid: C=100 üst kenar (10/15), (2,4) alt kenar (11/15) -> ikisi de kaydırıldı
        grid={"tfidf__ngram_range": [(1, 3), (2, 4)], "clf__C": [10.0, 100.0, 1000.0]},
        fixed={**TFIDF_FIXED, "analyzer": "char_wb", "clf": "LogisticRegression(lbfgs, L2)"},
    ),
    # Duyarlılık analizi: char_lr ile AYNI grid ve protokol, tek fark C1 tırnaklarının silinmesi
    "char_lr_c1clean": Spec(
        make=lambda: _clean_pipe(LogisticRegression(max_iter=3000), analyzer="char_wb"),
        grid={"tfidf__ngram_range": [(1, 3), (2, 4)], "clf__C": [10.0, 100.0, 1000.0]},
        fixed={
            **TFIDF_FIXED,
            "analyzer": "char_wb",
            "clf": "LogisticRegression(lbfgs, L2)",
            "preprocess": "yalnızca U+0091-U+0094 silinir (durumsuz FunctionTransformer)",
        },
    ),
    "word_svc": Spec(
        make=lambda: _pipe(LinearSVC(max_iter=10_000, random_state=RANDOM_STATE)),
        grid={"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1.0, 10.0]},
        fixed={**TFIDF_FIXED, "analyzer": "word", "clf": "LinearSVC(squared hinge, OvR)"},
    ),
    "word_cnb": Spec(
        make=lambda: _pipe(ComplementNB()),
        grid={"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.01, 0.1, 1.0]},
        fixed={**TFIDF_FIXED, "analyzer": "word", "clf": "ComplementNB"},
    ),
}


def macro_f1_scorer(labels: list[int]):
    """Etiket sırası sabit: bir fold'da eksik sınıf olsa bile 7 sınıf üzerinden ortalama."""
    return make_scorer(f1_score, average="macro", labels=labels, zero_division=0)


def _features(train: pd.DataFrame, features: np.ndarray | None):
    if features is None:
        return train["text"]
    if len(features) != len(train):
        raise ValueError(f"features satır sayısı {len(features)} != train {len(train)}")
    return np.asarray(features)


def _rows(X, idx):
    return X.iloc[idx] if isinstance(X, pd.Series) else X[idx]


def _grid_search(spec: Spec, cv, labels: list[int], n_jobs: int) -> GridSearchCV:
    return GridSearchCV(
        spec.make(), spec.grid, scoring=macro_f1_scorer(labels), cv=cv, n_jobs=n_jobs
    )


def nested_cv(
    spec: Spec,
    train: pd.DataFrame,
    labels: list[int],
    path: Path = SPLIT_PATH,
    n_repeats: int = N_REPEATS,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
    n_jobs: int = 1,
    features: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """(fold kayıtları, dış-fold tahminleri). Her kayıt bir (tekrar, fold) çiftidir.

    `features` verilirse (train satırlarına hizalı, ör. donmuş embedding [n, H]) model metin
    yerine bunu görür; fold'lar ve gruplar yine `train`'den make_cv ile kurulur.
    """
    train = train.reset_index(drop=True)
    X, y = _features(train, features), train["label"].to_numpy()
    y_pred = np.full((n_repeats, len(train)), -1, dtype=np.int64)
    fold_id = np.full((n_repeats, len(train)), -1, dtype=np.int64)
    folds = []
    for r in range(n_repeats):
        seed = RANDOM_STATE + r
        cv, g = make_cv(train, path, n_splits=outer_splits, random_state=seed)
        for k, (tr, va) in enumerate(cv.split(X, y, g)):
            tr_df = train.iloc[tr].reset_index(drop=True)
            inner, g_in = make_cv(tr_df, path, n_splits=inner_splits, random_state=seed)
            gs = _grid_search(spec, inner, labels, n_jobs)
            gs.fit(_rows(X, tr), tr_df["label"].to_numpy(), groups=g_in)
            pred = gs.predict(_rows(X, va))
            y_pred[r, va], fold_id[r, va] = pred, k
            folds.append(
                {
                    "repeat": r,
                    "fold": k,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "macro_f1": f1_score(
                        y[va], pred, average="macro", labels=labels, zero_division=0
                    ),
                    "accuracy": accuracy_score(y[va], pred),
                    "per_class_f1": f1_score(
                        y[va], pred, average=None, labels=labels, zero_division=0
                    ).tolist(),
                    "best_params": gs.best_params_,
                    "inner_best_score": gs.best_score_,
                    "val_row_ids": train["row_id"].iloc[va].tolist(),
                }
            )
    assert (y_pred >= 0).all(), "her satır her tekrarda tam bir kez doğrulanmalı"
    preds = {"row_id": train["row_id"].to_numpy(), "y_true": y, "y_pred": y_pred}
    return folds, preds | {"fold_id": fold_id}


def final_params(
    spec: Spec,
    train: pd.DataFrame,
    labels: list[int],
    path: Path = SPLIT_PATH,
    n_jobs: int = 1,
    n_splits: int = OUTER_SPLITS,
    features: np.ndarray | None = None,
) -> tuple[dict, float]:
    """Tüm train'de seçim (test fazında kullanılacak parametreler) ve iyimser CV skoru.

    Bölümleme dış CV'nin 0. tekrarıyla AYNIDIR (aynı n_splits, random_state=RANDOM_STATE);
    böylece selection_bias bölümleme gürültüsünden arınır.
    """
    train = train.reset_index(drop=True)
    cv, g = make_cv(train, path, n_splits=n_splits, random_state=RANDOM_STATE)
    gs = _grid_search(spec, cv, labels, n_jobs)
    gs.fit(_features(train, features), train["label"].to_numpy(), groups=g)
    return gs.best_params_, float(gs.best_score_)


def _summary(folds: list[dict], label_names: list[str], n_repeats: int) -> dict:
    f1 = np.array([f["macro_f1"] for f in folds])
    acc = np.array([f["accuracy"] for f in folds])
    pc = np.array([f["per_class_f1"] for f in folds])
    inner = np.array([f["inner_best_score"] for f in folds])
    rep = f1.reshape(n_repeats, -1).mean(axis=1)
    return {
        "macro_f1_folds": f1.tolist(),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)),  # örneklem std (ddof=1), tüm fold'lar üzerinden
        "macro_f1_repeat_means": rep.tolist(),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "per_class_f1_mean": dict(zip(label_names, pc.mean(axis=0).tolist(), strict=True)),
        # iç modeller train'in ~%53'ünü görür: bu skor öğrenme eğrisi yüzünden KARAMSARDIR,
        # seçim yanlılığı için kullanılmaz (bkz. selection_bias)
        "inner_best_score_mean": float(inner.mean()),
    }


def selection_bias(final_best_score: float, repeat0_mean: float) -> float:
    """Seçim yanlılığının kaba tahmini: aynı bölümleme ve eğitim boyutunda (dış CV 0. tekrar)
    seçilmiş en iyi skor - nested dış skor. Belirsizliği verilmemiş tek bir sayıdır."""
    return final_best_score - repeat0_mean


def run_baseline(
    name: str,
    train: pd.DataFrame,
    path: Path = SPLIT_PATH,
    out_dir: Path = METRICS_DIR,
    n_repeats: int = N_REPEATS,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
    n_jobs: int = 1,
) -> Path:
    """Bir Faz 2 baseline'ını çalıştırır; metrik json'u ve dış-fold tahminlerini yazar."""
    return run_nested(
        name, SPECS[name], train, path, out_dir, n_repeats, outer_splits, inner_splits, n_jobs
    )


def run_nested(
    name: str,
    spec: Spec,
    train: pd.DataFrame,
    path: Path = SPLIT_PATH,
    out_dir: Path = METRICS_DIR,
    n_repeats: int = N_REPEATS,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
    n_jobs: int = 1,
    features: np.ndarray | None = None,
    phase: int = 2,
    extra: dict | None = None,
) -> Path:
    """Nested CV + final seçim; `phase<phase>_<name>_<ts>.json` ve tahmin npz'i yazar.

    Faz 2 (metin) ve Faz 3 (donmuş embedding, `features`) aynı protokolü paylaşır.
    """
    split = load_split(path)
    label_names = split["label_names"]
    labels = list(range(len(label_names)))
    t0, started = time.perf_counter(), report.now()
    folds, preds = nested_cv(
        spec, train, labels, path, n_repeats, outer_splits, inner_splits, n_jobs, features
    )
    best, best_score = final_params(
        spec, train, labels, path, n_jobs, n_splits=outer_splits, features=features
    )
    stamp = f"{started:%Y%m%d-%H%M%S}"
    prefix = "" if phase == 2 else f"phase{phase}_"
    preds_file = f"preds/cv_{prefix}{name}_{stamp}.npz"
    out = {
        "experiment": f"phase{phase}_{name}",
        "phase": phase,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": Path(path).stem,
        "seed": RANDOM_STATE,
        "label_names": label_names,
        "params": {"fixed": spec.fixed, "grid": spec.grid},
        "cv": {
            "scheme": (
                f"nested; dış: make_cv {outer_splits} fold x {n_repeats} tekrar "
                f"(random_state {RANDOM_STATE}..{RANDOM_STATE + n_repeats - 1}), "
                f"iç: make_cv {inner_splits} fold GridSearchCV; skor macro-F1 (7 sınıf)"
            ),
            "n_folds": outer_splits,
            "n_repeats": n_repeats,
            "inner_n_folds": inner_splits,
            **(summary := _summary(folds, label_names, n_repeats)),
            "macro_f1_std_note": "fold skorlarının dağılımı; ortalamanın belirsizliği DEĞİL",
            "selection_bias": selection_bias(best_score, summary["macro_f1_repeat_means"][0]),
            "selection_bias_note": "final seçim skoru - dış CV 0. tekrar (aynı bölümleme); kaba",
            "folds": [{k: v for k, v in f.items() if k != "val_row_ids"} for f in folds],
        },
        "final": {
            "best_params": best,
            "cv_best_score_optimistic": best_score,
            "note": "tüm train'de dış CV 0. tekrarla aynı bölümlemede seçildi; skor iyimser",
        },
        "preds_file": preds_file,
        "test": None,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "notes": f"Faz {phase}: test seti kullanılmadı.",
        **(extra or {}),
    }
    path_json = report.write_metrics(out, f"phase{phase}_{name}", out_dir, when=started)
    pred_dir = Path(out_dir) / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(Path(out_dir) / preds_file, **preds)  # json ile aynı zaman damgası
    return path_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=[*SPECS, "all"])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()
    train = load_train()  # test satırları belleğe hiç alınmaz
    for name in SPECS if args.model == "all" else [args.model]:
        p = run_baseline(name, train, n_repeats=args.repeats, n_jobs=args.n_jobs)
        print(f"{name}: {p}")


if __name__ == "__main__":
    main()

"""Faz 3: donmuş encoder embedding'i + StandardScaler + LogisticRegression, Faz 2 ile aynı protokol.

Embedding'ler `ttc4900.embeddings` önbelleğinden (yalnızca train) okunur. Veriden öğrenen her adım
(StandardScaler: sütun ortalaması/std'si; LR) Pipeline içindedir ve her fit yalnızca o anki train
parçasında yapılır (dış fold'un train'i, iç CV'de iç train). Test embedding'i bu fazda yoktur.

Bu dosyadaki aile, grid ve kurallar CV KOŞMADAN ÖNCE ayrı bir commit'le donduruldu.

Kullanım:
    python -m ttc4900.frozen --encoder all --n-jobs 8     # ailenin nested CV'si (ham metin)
    python -m ttc4900.frozen --expand-edges --n-jobs 8    # grid kenarı kuralı (tetiklenirse)
    python -m ttc4900.frozen --sensitivity --n-jobs 8     # en iyi encoder, C1 temiz metin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ttc4900.baselines import Spec, run_nested
from ttc4900.data import load_train
from ttc4900.embeddings import ENCODERS, cache_meta, embed_train, load_embeddings
from ttc4900.paths import METRICS_DIR

FAMILY = ("berturk_mean", "e5_base", "bge_m3")
REFERENCE = "char_lr"  # Faz 2 birincil klasik baseline (ham metin), sabit
C1_SUFFIX = "_c1clean"

# StandardScaler: her boyutu train parçasının ortalama/std'siyle z-skora çevirir; LR'nin L2 cezası
# böylece boyutlara eşit davranır. C ölçeği TF-IDF'tekinden farklı olduğundan grid 5 dekad.
SPEC = Spec(
    make=lambda: Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]
    ),
    grid={"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    fixed={
        "features": "donmuş encoder embedding'i (pooling + L2, ttc4900.embeddings)",
        "scale": "StandardScaler (Pipeline içinde, fold'un train kısmında fit)",
        "clf": "LogisticRegression(lbfgs, L2, multinomial, max_iter=5000)",
    },
)

RULES = {
    "family": (
        f"Aile = {list(FAMILY)} (m=3). Token uzunluğu ve 64 metinlik hız ölçümünden sonra, "
        "CV'den önce donduruldu; bge-m3 önceden konan <= 60 dk eşiğini (32 dk) geçtiği için "
        "dahil."
    ),
    "max_length": (
        "Üç encoder'da 512 token, baştan kesme (head truncation). Kesilen metin oranı %22-32, "
        "önceden konan %40 parçalama eşiğinin altında. bge-m3'ün kendi sınırı 8192 token'dır; "
        "512 karşılaştırılabilirlik (BERT ailesinin sınırı) ve CPU maliyeti için seçildi, bu "
        "yüzden bge-m3 uzun metin avantajını bu fazda kullanmaz."
    ),
    "input_text": (
        "Ham metin (Faz 2 birincil char_lr ile aynı girdi). ASİMETRİ: e5_base ve bge_m3 "
        "(XLM-R tokenizer) U+0091-U+0094 kontrol karakterlerini gerçek token id'leriyle görür; "
        "berturk_mean (BERT normalizer'ı Cc karakterlerini siler) görmez. Encoder'lar arası ve "
        "char_lr'ye karşı farklar bu artefakta erişim farkını içerebilir."
    ),
    "comparison": (
        f"Her encoder için d_j = encoder_j - {REFERENCE}_j (aynı 15 dış fold, npz'den "
        "doğrulanır). Sütunlar: düz eşleştirilmiş t p'si (karar için kullanılmaz), "
        "Nadeau-Bengio düzeltmeli t (df=14, oran n_val/n_train), N-B p'lerine Holm (aile = 3 "
        "encoder). Referans sabit char_lr'dir (argmax değil). Encoder'lar arası sıralama "
        "betimseldir."
    ),
    "grid_edge": (
        "Final seçim ya da dış fold seçimlerinin >= 8/15'i C grid'inin bir ucundaysa grid o "
        "yönde bir dekad genişletilip o encoder bir kez yeniden koşulur; iki koşu da raporlanır "
        "(Faz 2'deki tek genişletme ile aynı ilke)."
    ),
    "test_transfer": (
        "Teste taşınan tek Faz 3 konfigürasyonu: ham metinle dış CV macro-F1 ortalaması en "
        "yüksek encoder, final.best_params ile; char_lr'yi geçip geçmemesinden BAĞIMSIZ taşınır. "
        "Testte char_lr'ye karşı McNemar + paired bootstrap; Faz 2'nin ikincil Holm ailesine "
        "(birincil char_lr'ye karşı karşılaştırmalar) eklenir, aile 3'ten 4'e çıkar. Birincil "
        "aile (BERTurk fine-tuning vs char_lr) değişmez. Test embedding'i yalnızca final "
        "değerlendirmede, aynı revision/max_length/pooling ile üretilir; test sonucuna göre "
        "encoder, pooling ya da C değiştirilmez."
    ),
    "sensitivity_c1": (
        "Ham metinle CV'de en iyi çıkan encoder (test_transfer ile aynı seçim) bir kez de "
        "U+0091-U+0094 silinmiş metinle (baselines.strip_c1_quotes) embed edilip aynı protokolle "
        "koşulur. Tek karşılaştırma: ham - temiz, Nadeau-Bengio (Holm yok). İkincil duyarlılık "
        "analizi olarak raporlanır; model seçiminde KULLANILMAZ ve teste taşınmaz. En iyi "
        "berturk_mean çıkarsa tokenizer bu karakterleri zaten sildiğinden fark yapısal olarak "
        "~0 beklenir; yine de koşulur."
    ),
}


def expanded_spec(direction: str) -> Spec:
    """RULES["grid_edge"]: C grid'i bir dekad aşağı ("low") ya da yukarı ("high") genişletilir."""
    grid = SPEC.grid["clf__C"]
    new = [grid[0] / 10, *grid] if direction == "low" else [*grid, grid[-1] * 10]
    return Spec(make=SPEC.make, grid={"clf__C": new}, fixed=SPEC.fixed)


def edge_direction(m: dict) -> str | None:
    """Kuralın tetiklenip tetiklenmediği: final uçta ya da dış seçimlerin >= 8/15'i bir uçta."""
    from ttc4900.cv_compare import grid_edges

    e = grid_edges({"x": m})["x"]["clf__C"]
    if e["final_edge"]:
        return e["final_edge"]
    if e["n_folds_low_edge"] >= 8:
        return "low"
    if e["n_folds_high_edge"] >= 8:
        return "high"
    return None


def best_encoder(metrics_dir: Path = METRICS_DIR) -> str:
    """Ham metinle dış CV ortalaması en yüksek aile üyesi (eksik sonuç varsa hata)."""
    from ttc4900.cv_compare import latest_metrics

    res = {n: latest_metrics(n, metrics_dir, phase=3) for n in FAMILY}
    missing = [n for n, m in res.items() if m is None]
    if missing:
        raise SystemExit(f"Önce ailenin CV'si: eksik {missing}")
    return max(res, key=lambda n: res[n]["cv"]["macro_f1_mean"])


def run(
    name: str,
    train: pd.DataFrame,
    n_jobs: int = 1,
    c1clean: bool = False,
    spec: Spec = SPEC,
    extra: dict | None = None,
) -> Path:
    X = load_embeddings(name, train, c1clean)
    meta = cache_meta(name, c1clean)
    label = name + (C1_SUFFIX if c1clean else "")
    return run_nested(
        label,
        spec,
        train,
        n_jobs=n_jobs,
        features=X,
        phase=3,
        extra={"embedding": meta, "rules": RULES, "reference": REFERENCE, **(extra or {})},
    )


def expand_edges(train: pd.DataFrame, n_jobs: int = 1, metrics_dir: Path = METRICS_DIR) -> None:
    """RULES["grid_edge"]'i uygular: tetiklenen encoder'lar genişletilmiş grid'le BİR KEZ
    yeniden koşulur (zaten genişletilmiş bir koşu varsa tekrar edilmez)."""
    from ttc4900.cv_compare import latest_metrics

    for n in FAMILY:
        m = latest_metrics(n, metrics_dir, phase=3)
        if m.get("grid_expansion"):
            print(f"{n}: zaten genişletilmiş grid'le koşulmuş, atlandı", flush=True)
            continue
        d = edge_direction(m)
        if d is None:
            print(f"{n}: kural tetiklenmedi", flush=True)
            continue
        spec = expanded_spec(d)
        info = {
            "rule": RULES["grid_edge"],
            "direction": d,
            "previous_run": m["_file"],
            "previous_grid": m["params"]["grid"],
            "previous_macro_f1_mean": m["cv"]["macro_f1_mean"],
            "previous_final_params": m["final"]["best_params"],
        }
        print(n, d, run(n, train, n_jobs, spec=spec, extra={"grid_expansion": info}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="all", choices=[*FAMILY, "all"])
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--expand-edges", action="store_true", help="RULES['grid_edge'] uygula")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()
    train = load_train()  # test satırları belleğe hiç alınmaz
    if args.expand_edges:
        expand_edges(train, args.n_jobs)
        return
    if args.sensitivity:
        best = best_encoder()
        print(f"en iyi encoder (ham): {best}; C1 temiz metinle embed + CV", flush=True)
        embed_train(best, train, c1clean=True)
        print(run(best, train, args.n_jobs, c1clean=True), flush=True)
        return
    for n in FAMILY if args.encoder == "all" else [args.encoder]:
        assert ENCODERS[n].max_length == 512
        print(n, run(n, train, args.n_jobs), flush=True)


if __name__ == "__main__":
    main()

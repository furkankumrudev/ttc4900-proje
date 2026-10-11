"""Hedef sızıntısı denetimi: LR katsayılarında sınıf başına en ağırlıklı özellikler (rapor).

Sezgi: model etiketi metnin konusundan değil de etiketle birlikte gelen bir artefakttan
(kategori adı, ajans/kaynak imzası, sabit şablon) öğreniyorsa skor yüksek ama genellemesi
sahte olur. Çok terimli LR'de sınıf k'nın skoru z_k = w_k·x + b_k olduğundan, w_k'nin en büyük
bileşenleri o sınıfı en çok iten özelliklerdir; bunlara bakıp artefakt arayabiliriz.

Kurallar yalnızca İŞARET koyar (insan incelemesine aday); model ve veri değiştirilmez.
Modeller Faz 2 final parametreleriyle YALNIZCA train'de fit edilir; test belleğe alınmaz.

Kullanım: python -m ttc4900.feature_audit
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ttc4900 import report
from ttc4900.baselines import SPECS
from ttc4900.cv_compare import latest_phase2
from ttc4900.data import SPLIT_PATH, load_split, load_train
from ttc4900.paths import OUTPUTS

AUDIT_DIR = OUTPUTS / "leakage"
MODELS = ("char_lr", "word_lr")
TOP_K = 20
BOUNDARY_WORDS = 5  # şablon araması: metnin ilk/son 5 kelimesi
TEMPLATE_MIN_SHARE = 0.30  # sınıf belgelerinin >= %30'unda sınırda geçiyorsa şablon adayı
CONCENTRATED_SHARE = 0.95  # belge frekansının >= %95'i tek sınıfta ...
CONCENTRATED_MIN_DF = 20  # ... ve en az 20 belgede

# Kategori adları ve Türkçe harfli yazımları (etiket adları ASCII: dunya, kultur, saglik)
CATEGORY_TERMS = {
    "dunya": ("dunya", "dünya"),
    "ekonomi": ("ekonomi",),
    "kultur": ("kultur", "kültür"),
    "saglik": ("saglik", "sağlık"),
    "siyaset": ("siyaset",),
    "spor": ("spor",),
    "teknoloji": ("teknoloji",),
}
# Ajans / kaynak / künye imzaları (veride noktalama yok, kesme işareti boşluk).
# "cihan" (Cihan Haber Ajansı) ilk listede yoktu; ilk koşuda dunya'nın 1. özelliği çıkınca eklendi.
SOURCE_TERMS = frozenset(
    "aa dha iha anka cihan ihlas reuters afp ap bbc cnn ntv haber_merkezi muhabir kaynak foto "
    "www com http https".split()
)
# cp1252 "akıllı tırnak"lar (‘ ’ “ ” –) yanlış çözülünce C1 kontrol karakterleri (U+0080-U+009F)
# olarak kalır: kaynağa (kazıma hattına) bağlı bir artefakt, konuyla ilgisiz.
C1 = re.compile("[-]")
RULES = {
    "category_name": "özellik (char için örnek kelimesi) bir kategori adıyla başlıyor / içeriyor",
    "source_signature": f"kelime ajans/kaynak listesinde: {sorted(SOURCE_TERMS)}",
    "template": (
        f"sınıf belgelerinin >= {TEMPLATE_MIN_SHARE:.0%} kadarında ilk/son {BOUNDARY_WORDS} "
        "kelimede geçiyor"
    ),
    "encoding_artifact": "özellik C1 kontrol karakteri (U+0080-U+009F) içeriyor",
    "concentrated": (
        f"df'nin >= {CONCENTRATED_SHARE} oranı tek sınıfta ve toplam df >= {CONCENTRATED_MIN_DF}"
    ),
}


def _terms() -> list[str]:
    return [t for ts in CATEGORY_TERMS.values() for t in ts]


def word_flags(word: str) -> set[str]:
    """Tek bir kelime için kategori adı / kaynak işaretleri."""
    flags = set()
    if any(word.startswith(t) for t in _terms()):  # ekonomik, sporcu, siyasetçi ...
        flags.add("category_name")
    if word in SOURCE_TERMS:
        flags.add("source_signature")
    return flags


def char_flags(ngram: str, example_words: list[str]) -> set[str]:
    """char_wb n-gram: kendisi bir kategori adının parçasıysa ya da örnek kelimeleri işaretliyse.

    char_wb kelimeleri boşlukla sarar (" spo" = kelime başı "spo"); kelime başı ≥3 harfli bir
    n-gram kategori adının öneki, ya da ≥4 harfli bir n-gram adın alt dizesiyse işaretlenir.
    """
    s = ngram.strip()
    flags = {"encoding_artifact"} if C1.search(ngram) else set()
    for t in _terms():
        if (ngram.startswith(" ") and len(s) >= 3 and t.startswith(s)) or (len(s) >= 4 and s in t):
            flags.add("category_name")
    for w in example_words:
        flags |= word_flags(w)
    return flags


def _boundary(text: str) -> list[str]:
    w = text.split()
    return w[:BOUNDARY_WORDS] + w[-BOUNDARY_WORDS:]


def _in_boundary(feature: str, words: list[str], analyzer: str) -> bool:
    if analyzer == "word":
        return f" {feature} " in f" {' '.join(words)} "
    return any(feature in f" {w} " for w in words)


def example_words(ngram: str, word_freq: Counter, n: int = 3) -> list[str]:
    """n-gram'ı içeren (char_wb sarmalıyla) train'deki en sık n kelime."""
    hits = [(c, w) for w, c in word_freq.items() if ngram in f" {w} "]
    return [w for _, w in sorted(hits, key=lambda x: (-x[0], x[1]))[:n]]


def fit_final(name: str, train: pd.DataFrame, best_params: dict):
    """Faz 2 final konfigürasyonu, tüm train'de (CV yok; yalnızca katsayı incelemesi)."""
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in best_params.items()}
    pipe = SPECS[name].make().set_params(**params)
    return pipe.fit(train["text"], train["label"].to_numpy())


def audit_model(
    name: str, pipe, train: pd.DataFrame, label_names: list[str], top_k: int = TOP_K
) -> pd.DataFrame:
    """Sınıf başına en büyük top_k katsayı + belge frekansları + işaretler."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    analyzer = "word" if vec.analyzer == "word" else "char"
    feats = vec.get_feature_names_out()
    y = train["label"].to_numpy()
    B = (vec.transform(train["text"]) > 0).tocsc()  # ikili belge-özellik matrisi
    word_freq = Counter(w for t in train["text"] for w in t.split())
    bounds = [_boundary(t) for t in train["text"]]
    # iki sınıfta sklearn tek satır (sınıf 1 lehine) tutar: z_0 = -z_1 alınır
    coef = clf.coef_ if clf.coef_.shape[0] > 1 else np.vstack([-clf.coef_[0], clf.coef_[0]])
    rows = []
    for k, cls in enumerate(clf.classes_):
        in_k = y == cls
        for rank, j in enumerate(np.argsort(-coef[k])[:top_k], start=1):
            f = str(feats[j])
            col = B[:, j].toarray().ravel()
            df_in, df_out = int(col[in_k].sum()), int(col[~in_k].sum())
            ex = example_words(f, word_freq) if analyzer == "char" else []
            flags = char_flags(f, ex) if analyzer == "char" else word_flags(f)
            hit = [_in_boundary(f, bounds[i], analyzer) for i in np.flatnonzero(in_k)]
            b_share = float(np.mean(hit))
            if b_share >= TEMPLATE_MIN_SHARE:
                flags.add("template")
            share = df_in / max(df_in + df_out, 1)
            if share >= CONCENTRATED_SHARE and df_in + df_out >= CONCENTRATED_MIN_DF:
                flags.add("concentrated")
            rows.append(
                {
                    "model": name,
                    "class": label_names[int(cls)],
                    "rank": rank,
                    "feature": f,
                    "coef": float(coef[k, j]),
                    "df_in_class": df_in,
                    "df_other": df_out,
                    "share_in_class": share,
                    "boundary_share_in_class": b_share,
                    "flags": ";".join(sorted(flags)),
                    "example_words": " ".join(ex),
                }
            )
    return pd.DataFrame(rows)


def artifact_by_class(train: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    """Sınıf başına kaynak kelimesi ve C1 artefaktı içeren belge oranı (konudan bağımsız işaretler).

    Bir sınıfta oran diğerlerinden çok farklıysa model kaynağı/kazıma hattını öğrenebilir.
    """
    words = train["text"].map(lambda t: set(t.split()))
    marks = {f"src:{w}": words.map(lambda ws, w=w: w in ws) for w in sorted(SOURCE_TERMS)}
    marks["c1_chars"] = train["text"].map(lambda t: bool(C1.search(t)))
    df = pd.DataFrame(marks).assign(category=train["label"].map(lambda i: label_names[int(i)]))
    share = df.groupby("category").mean().T
    share = share[share.max(axis=1) > 0]  # hiç geçmeyen terimler atılır
    share["max_over_min"] = share.max(axis=1) / share.min(axis=1).clip(lower=1e-3)
    return share.sort_values("max_over_min", ascending=False)


def boundary_ngrams(train: pd.DataFrame, label_names: list[str], n: int = 3, top: int = 10):
    """Her sınıfta en sık ilk/son n kelime dizileri (şablon/künye için göz kontrolü)."""
    rows = []
    for lab, g in train.groupby("label"):
        for pos, take in (("first", lambda w: w[:n]), ("last", lambda w: w[-n:])):
            c = Counter(" ".join(take(t.split())) for t in g["text"])
            for seq, cnt in c.most_common(top):
                rows.append(
                    {
                        "class": label_names[int(lab)],
                        "position": pos,
                        "ngram": seq,
                        "count": cnt,
                        "share_in_class": cnt / len(g),
                    }
                )
    return pd.DataFrame(rows)


def main(path: Path = SPLIT_PATH, out_dir: Path = AUDIT_DIR) -> Path:
    train = load_train(path)  # test satırları belleğe hiç alınmaz
    label_names = load_split(path)["label_names"]
    phase2 = latest_phase2()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in MODELS:
        best = phase2[name]["final"]["best_params"]
        pipe = fit_final(name, train, best)
        df = audit_model(name, pipe, train, label_names)
        df.to_csv(out_dir / f"top_features_{name}.csv", index=False, encoding="utf-8")
        flagged = df[df["flags"] != ""]
        counts = Counter(f for fl in flagged["flags"] for f in fl.split(";"))
        summary[name] = {
            "source_metrics": phase2[name]["_file"],
            "best_params": best,
            "n_features": int(len(pipe.named_steps["tfidf"].vocabulary_)),
            "n_top": len(df),
            "n_flagged": len(flagged),
            "flag_counts": dict(sorted(counts.items())),
            "flagged": flagged.drop(columns=["model"]).to_dict(orient="records"),
        }
        print(f"{name}: {len(flagged)}/{len(df)} işaretli {dict(counts)}")
    art = artifact_by_class(train, label_names)
    art.to_csv(out_dir / "artifact_by_class.csv", encoding="utf-8")
    bn = boundary_ngrams(train, label_names)
    bn.to_csv(out_dir / "boundary_ngrams.csv", index=False, encoding="utf-8")
    out = {
        "experiment": "feature_audit",
        "phase": 2,
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": Path(path).stem,
        "scope": "Yalnızca train; Faz 2 final parametreleriyle tüm train'de fit. Test yok.",
        "top_k": TOP_K,
        "rules": RULES,
        "models": summary,
        "artifact_by_class": art.drop(columns="max_over_min").round(4).to_dict(orient="index"),
        "boundary_max_share": {
            pos: float(bn[bn["position"] == pos]["share_in_class"].max())
            for pos in ("first", "last")
        },
        "note": "Yalnızca rapor: işaret = insan incelemesine aday, sızıntı kanıtı değil.",
    }
    p = out_dir / "feature_audit.json"
    p.write_text(
        json.dumps(report.to_jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(p)
    return p


if __name__ == "__main__":
    main()

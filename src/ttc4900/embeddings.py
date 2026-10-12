"""Faz 3: dondurulmuş encoder'larla YALNIZCA train embedding'i (fine-tuning yok).

Akış: metin --AutoTokenizer--> input_ids, attention_mask [B, T]
           --AutoModel------> last_hidden_state h [B, T, H]
           --pooling--------> e [B, H]  (mask'li ortalama ya da CLS), ardından e / ||e||_2

Encoder donmuş olduğundan her metnin embedding'i yalnız kendi metnine bağlı deterministik bir
fonksiyondur (tokenizasyon gibi); bir kez hesaplanıp önbelleğe yazılması sızıntı değildir.
Veriden öğrenen her adım (StandardScaler, LR) sonra Pipeline içinde, fold'un train kısmında fit
edilir. Test satırları bu modülde hiç yüklenmez; gelirse `assert_train_only` durdurur.

Kullanım:
    python -m ttc4900.embeddings --lengths              # token uzunluğu dağılımı (ağ: tokenizer)
    python -m ttc4900.embeddings --benchmark --n 64     # CPU hız ölçümü (ağ: model ağırlıkları)
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ttc4900 import report
from ttc4900.data import RANDOM_STATE, SPLIT_PATH, cv_groups, load_train
from ttc4900.paths import OUTPUTS

CACHE_DIR = OUTPUTS / "cache"
C1_PROBE = "piyasa \u0091 güçlü \u0092 dedi \u0093 yatırım \u0094"  # Faz 2 denetimindeki artefakt


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    revision: str  # Hub commit SHA (hf-tutor ile API'den doğrulandı)
    pooling: str  # "mean" | "cls"
    prefix: str  # metnin başına eklenen sabit metin
    max_length: int
    source: str  # pooling/prefix kararının dayanağı


# Aday aile; hangilerinin CV'ye gireceği token uzunluğu ve hız ölçümünden SONRA, CV'den önce
# ayrı bir commit'le dondurulur (Faz 3 planı 3.0).
ENCODERS: dict[str, EncoderSpec] = {
    "e5_base": EncoderSpec(
        model_id="intfloat/multilingual-e5-base",
        revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        pooling="mean",
        prefix="query: ",
        max_length=512,
        source="model card: 'query: ' prefix for linear probing classification; 1_Pooling mean",
    ),
    "berturk_mean": EncoderSpec(
        model_id="dbmdz/bert-base-turkish-cased",
        revision="b6e1de16c983e0f2c70664591ea3f22810072608",
        pooling="mean",
        prefix="",
        max_length=512,
        source="sentence modeli değil, card pooling önermiyor; mask'li ortalama seçildi",
    ),
    "bge_m3": EncoderSpec(
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        pooling="cls",
        prefix="",
        max_length=512,  # model 8192'ye kadar destekler; CPU maliyeti için 512 (plan 3.6)
        source="1_Pooling/config.json: pooling_mode_cls_token=true; card: prefix gerekmez",
    ),
}


# --------------------------------------------------------------------------- yardımcılar
def assert_train_only(train: pd.DataFrame, path: Path = SPLIT_PATH) -> None:
    """Her row_id split'in train'inde olmalı; değilse (test satırı ya da yabancı) hata."""
    cv_groups(train, path)  # yabancı satırda ValueError atar


def row_ids_hash(row_ids) -> str:
    return hashlib.sha256(np.asarray(row_ids, dtype=np.int64).tobytes()).hexdigest()


def load_tokenizer(spec: EncoderSpec):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def load_model(spec: EncoderSpec, device: str = "cpu"):
    """Gövde encoder (sınıflandırma kafası yok), eval modunda: dropout kapalı."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
    got = getattr(model.config, "_commit_hash", None)
    if got is not None and got != spec.revision:
        raise ValueError(f"{spec.model_id}: beklenen revision {spec.revision}, yüklenen {got}")
    return model.to(device).eval()


def pool(h, mask, how: str):
    """[B, T, H] -> [B, H], ardından L2 normalize.

    mean: e = sum_t m_t h_t / sum_t m_t (padding ortalamaya girmez); cls: e = h_0.
    """
    import torch

    if how == "mean":
        m = mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    elif how == "cls":
        e = h[:, 0]
    else:
        raise ValueError(f"bilinmeyen pooling: {how}")
    return torch.nn.functional.normalize(e, dim=-1)


def _inputs(spec: EncoderSpec, texts) -> list[str]:
    return [spec.prefix + t for t in texts]


def _count(tok, inputs: list[str]) -> np.ndarray:
    enc = tok(inputs, add_special_tokens=True, truncation=False, verbose=False)
    return np.array([len(ids) for ids in enc["input_ids"]])


def token_lengths(tok, spec: EncoderSpec, texts) -> np.ndarray:
    """Kesmesiz token sayısı (özel token'lar ve prefix dahil)."""
    return _count(tok, _inputs(spec, texts))


def embed_texts(texts, tok, model, spec: EncoderSpec, batch_size: int = 16, device: str = "cpu"):
    """(E [n, H] float32, n_truncated). Uzunluğa göre sıralı batch'ler padding israfını azaltır;
    çıktı girdi sırasına geri dizilir. Mask'li pooling sayesinde sonuç batch kompozisyonundan
    bağımsızdır."""
    import torch

    texts = _inputs(spec, list(texts))
    lengths = _count(tok, texts)
    order = np.argsort(-lengths, kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():  # gradyan ve autograd kaydı yok
        for s in range(0, len(texts), batch_size):
            idx = order[s : s + batch_size]
            batch = tok(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=spec.max_length,
                return_tensors="pt",
            ).to(device)
            h = model(**batch).last_hidden_state
            out[idx] = pool(h, batch["attention_mask"], spec.pooling).float().cpu().numpy()
    return out, int((lengths > spec.max_length).sum())


# --------------------------------------------------------------------------- ölçümler
def _quantiles(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": int(x.max()),
    }


def token_length_stats(train: pd.DataFrame, name: str) -> dict:
    """Tokenizer veriden öğrenmez (sabit sözlük): train'de sayım yapmak sızıntı değildir."""
    spec = ENCODERS[name]
    tok = load_tokenizer(spec)
    L = token_lengths(tok, spec, train["text"])
    words = train["text"].str.split().str.len().to_numpy()
    over = L > spec.max_length
    kept = np.minimum(L, spec.max_length)
    by_class = {
        cat: float((L[train["category"].to_numpy() == cat] > spec.max_length).mean())
        for cat in sorted(train["category"].unique())
    }
    probe = tok.tokenize(C1_PROBE)
    clean = tok.tokenize(C1_PROBE.translate({c: None for c in range(0x91, 0x95)}))
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "tokenizer": type(tok).__name__,
        "prefix": spec.prefix,
        "max_length": spec.max_length,
        "tokens": _quantiles(L),
        "tokens_per_word_median": float(np.median(L / words)),
        "frac_truncated": float(over.mean()),
        "frac_truncated_by_class": by_class,
        "frac_tokens_dropped": float(1 - kept.sum() / L.sum()),
        "total_tokens_after_truncation": int(kept.sum()),
        "c1_probe": {
            "tokens": probe,
            "tokens_without_c1": clean,
            "c1_visible_to_model": probe != clean,
        },
    }


def benchmark(train: pd.DataFrame, name: str, n: int = 64, batch_size: int = 16) -> dict:
    """n rastgele train metninde (etiket kullanılmaz) CPU süresi; tüm train'e token oranıyla
    ölçeklenir. İlk batch ısınma için ayrıca koşulur ve ölçüme katılmaz."""
    import torch
    import transformers

    spec = ENCODERS[name]
    tok = load_tokenizer(spec)
    t0 = time.perf_counter()
    model = load_model(spec)
    load_s = time.perf_counter() - t0
    sample = train["text"].sample(n=n, random_state=RANDOM_STATE).tolist()
    embed_texts(sample[:batch_size], tok, model, spec, batch_size)  # ısınma
    t0 = time.perf_counter()
    E, _ = embed_texts(sample, tok, model, spec, batch_size)
    sec = time.perf_counter() - t0
    kept_sample = np.minimum(token_lengths(tok, spec, sample), spec.max_length).sum()
    kept_all = np.minimum(token_lengths(tok, spec, train["text"]), spec.max_length).sum()
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "hidden_size": int(E.shape[1]),
        "n_texts": n,
        "batch_size": batch_size,
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
        "model_load_seconds": round(load_s, 1),
        "seconds": round(sec, 2),
        "seconds_per_text": sec / n,
        "tokens_per_second": float(kept_sample / sec),
        "est_full_train_minutes_by_texts": sec / n * len(train) / 60,
        "est_full_train_minutes_by_tokens": kept_all / (kept_sample / sec) / 60,
        "embedding_norms_ok": bool(np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)),
    }


def _write(obj: dict, name: str) -> Path:
    return report.write_metrics(obj, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="all", choices=[*ENCODERS, "all"])
    ap.add_argument("--lengths", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    names = list(ENCODERS) if args.encoder == "all" else [args.encoder]
    train = load_train()  # test satırları belleğe hiç alınmaz
    assert_train_only(train)
    base = {
        "phase": 3,
        "git_commit": report.git_commit(),
        "split": Path(SPLIT_PATH).stem,
        "n_train": len(train),
        "train_row_ids_sha256": row_ids_hash(train["row_id"]),
        "encoders": {n: asdict(ENCODERS[n]) for n in names},
        "test": None,
    }
    if args.lengths:
        res = {n: token_length_stats(train, n) for n in names}
        p = _write(base | {"experiment": "phase3_token_lengths", "results": res}, "phase3_lengths")
        print(p)
        for n, r in res.items():
            t = r["tokens"]
            print(f"{n:13s} medyan {t['median']:.0f}  p95 {t['p95']:.0f}  max {t['max']}  "
                  f">{r['max_length']}: {r['frac_truncated']:.1%}  atılan token "
                  f"{r['frac_tokens_dropped']:.1%}  tok/kelime {r['tokens_per_word_median']:.2f}  "
                  f"C1 görünür: {r['c1_probe']['c1_visible_to_model']}")  # fmt: skip
    if args.benchmark:
        res = {}
        for n in names:
            res[n] = benchmark(train, n, args.n, args.batch_size)
            r = res[n]
            print(f"{n:13s} {r['seconds']:.1f} sn / {args.n} metin  "
                  f"tahmin (token oranı) {r['est_full_train_minutes_by_tokens']:.1f} dk  "
                  f"(metin oranı {r['est_full_train_minutes_by_texts']:.1f} dk)  "
                  f"{r['n_params'] / 1e6:.0f}M parametre", flush=True)  # fmt: skip
        p = _write(base | {"experiment": "phase3_benchmark", "results": res}, "phase3_benchmark")
        print(p)


if __name__ == "__main__":
    main()

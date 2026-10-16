"""Faz 5: final test değerlendirmesi. Test setinin AÇILDIĞI tek modül (`load_train_test`).

Yöntem `ttc4900.test_plan.TEST_PLAN`'dadır (aileler, exact McNemar, grup bootstrap, karar kuralı,
cümle şablonları `SENTENCES`). Bu modülün kendi kuralları `RULES`'tadır; ikisi de test açılmadan
"test öncesi son commit"te donduruldu. İnsan okuması: docs/faz5_protokol.md.

Akış (her adım ayrı komut; sıra zorunlu):
    python -m ttc4900.final_eval verify        # yalnız train: BERT ağırlık sha256 + in-sample F1
    python -m ttc4900.final_eval predict       # testi BİR KEZ açar; test_<model>.npz write-once
    python -m ttc4900.final_eval evaluate      # yalnız npz'ler: aileler, CI'lar, cümleler
    python -m ttc4900.final_eval v1-diagnosis  # ana sonuçlardan SONRA: split_v1 sızıntı teşhisi

Yeniden eğitim: klasik modeller ve bge_m3 + LR, Faz 2/3 json'larındaki `final.best_params` ile
tüm train'de (3630 satır) bir kez fit edilir; arama/ayar yok (RULES["refit"]). BERTurk modelleri
yeniden eğitilmez: Faz 4 final ağırlıkları (outputs/models/<seed>) kullanılır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ttc4900 import report
from ttc4900.baselines import SPECS
from ttc4900.cv_compare import latest_metrics
from ttc4900.data import (
    NEAR_DUP_THRESHOLD,
    SPLIT_PATH,
    SPLIT_V1_PATH,
    load_split,
    load_train,
    load_train_test,
    test_groups,
)
from ttc4900.embeddings import CACHE_DIR, ENCODERS, load_embeddings, row_ids_hash
from ttc4900.finetune import (
    FINAL_RULES,
    FT_CONFIG,
    check_config,
    check_labels,
    label_maps,
    oof_metrics,
    weights_sha256,
)
from ttc4900.finetune import NAME as BERT_NAME
from ttc4900.paired_test import (
    _effect,
    _pct_ci,
    bootstrap_weights,
    cluster_bootstrap,
    compare_family,
    group_index,
)
from ttc4900.paths import METRICS_DIR, MODELS_DIR, PROJECT_ROOT
from ttc4900.test_plan import (
    ALPHA,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    DESCRIPTIVE,
    FAMILIES,
    PRIMARY_SEED,
    SENTENCES,
    TEST_PLAN,
)

SPLIT_NAME = Path(SPLIT_PATH).stem
PREDS_DIR = METRICS_DIR / "preds"
CLASSICAL = ("char_lr", "char_lr_c1clean", "word_lr", "word_svc", "word_cnb")
ENCODER = "bge_m3"  # frozen.RULES['test_transfer'] ile teste taşınan tek Faz 3 modeli
BERT_SLOTS = tuple(FT_CONFIG["seeds"])  # 42, 43, 44


def bert_name(slot: int) -> str:
    """Birincil slot 'berturk_ft' (FAMILIES'teki ad), diğerleri 'berturk_ft_s<slot>'."""
    return BERT_NAME if slot == PRIMARY_SEED else f"{BERT_NAME}_s{slot}"


CANDIDATES = (*CLASSICAL, ENCODER, *(bert_name(s) for s in BERT_SLOTS))
IN_SAMPLE_TOL = 0.005
V1_MODEL = "char_lr"
V1_FROZEN = {"clf__C": 100.0, "tfidf__ngram_range": (2, 4)}  # kullanıcı onayıyla, sonuç görülmeden
V1_B = BOOTSTRAP_B

RULES = {
    "refit": (
        "Klasik modeller (char_lr, char_lr_c1clean, word_lr, word_svc, word_cnb) ve bge_m3 + LR, "
        "tüm train (split_v2, 3630 satır) üzerinde, Faz 2/3 json'larındaki dondurulmuş "
        "final.best_params ile BİR KEZ fit edilir. Testten önce ya da sonra hiçbir yeniden "
        "seçim/ayar yapılmaz: GridSearchCV, make_cv, iç CV, early stopping ya da başka bir "
        "arama yok. Pipeline'ın sabit ayarları baselines.SPECS / frozen.official_spec ile aynıdır; "
        "best_params'ın her anahtarı o spec'in grid'inde ve değeri grid'deki bir değerdir. "
        "bge_m3'ün train embedding'i Faz 3 önbelleğinden (meta doğrulanarak) okunur. BERTurk "
        "modelleri yeniden eğitilmez: Faz 4 final ağırlıkları kullanılır."
    ),
    "verify": (
        "Test açılmadan, yalnız train ile (load_train): her BERTurk final modeli için (1) "
        "ağırlık dosyalarının sha256'sı Faz 4 final json'undaki weights_sha256 ile TAM eşit, "
        "(2) config id2label split label_names ile aynı, (3) yerel tokenizer HF önbelleğindeki "
        "sabit revision'ın vocab.txt'siyle aynı ve tüm train metinlerinde Hub revision "
        "tokenizer'ıyla aynı input_ids'i üretiyor, (4) train in-sample macro-F1 yerel CPU'da "
        "(fp32, eval modu) yeniden hesaplanır; |yeni - kayıt| <= "
        f"{IN_SAMPLE_TOL} ve yeni >= {FT_CONFIG['collapse_threshold']} (FINAL_RULES). Kayıt "
        "Colab T4'te fp16 ile üretildi; tolerans sayısal farkı karşılar (~18 satır). Biri "
        "tutmazsa test açılmaz. Birincil model seed numarasından değil phase4_summary "
        "finals.primary_model_dir'den okunur (Colab yolu; son klasör adı outputs/models/'e "
        "eşlenir)."
    ),
    "test_access": (
        "load_train_test yalnız bu modülün predict ve v1-diagnosis adımlarında çağrılır. predict "
        "ön koşulları: çalışma ağacı temiz; aynı kodla (verify commit'inden bu yana src/ ve "
        "requirements.txt değişmemiş) üretilmiş, temiz commit'te geçmiş bir "
        "phase5_model_check json'u var; ağırlık sha256'ları yeniden tutuyor. Model fit'leri "
        "test metni belleğe alınmadan önce yapılır."
    ),
    "write_once": (
        "Her aday için outputs/metrics/preds/test_<model>.npz (row_id, y_true, y_pred, scores, "
        "meta) bir kez yazılır: geçici dosya + os.link (hedef varsa hata, üzerine yazmaz). "
        "predict yeniden çalıştırılırsa var olan npz yeniden üretilmez; yalnız satırları ve "
        "önceki manifest'teki sha256'sı denetlenir, eksik modeller üretilir. bge_m3 test "
        "embedding'i ayrı önbellekte: emb_bge_m3_TEST_<rev>_<maxlen>.npz, meta side='test' ve "
        "test_row_ids_sha256 ile; load_embeddings side='test' dosyasını reddeder."
    ),
    "evaluate": (
        "Yalnız test npz'leri okunur (metin yok). Tüm modeller için aynı grup bootstrap "
        "ağırlıkları (paired_test.cluster_bootstrap, B=10000, seed 42). Birincil ve ikincil "
        "aile paired_test.compare_family ile; betimsel çiftler aynı replikatlardan fark ve CI "
        "(p yok). BERT seed özeti: 3 seed'in test macro-F1'i ortalama ± std (ddof=1). Cümleler "
        "test_plan.SENTENCES şablonlarından, karar decide() çıktısına göre seçilir."
    ),
    "v1_diagnosis": (
        "Faz 5'in ayrı ve tek 'sızıntı etkisi' teşhis deneyi; ana sonuçlar (phase5_test json'u) "
        "yazıldıktan SONRA, temiz commit'te çalışır ve ana tablolara girmez. Model: char_lr, "
        "dondurulmuş ayarlarla (C=100, char_wb, ngram (2,4); json'daki final.best_params ile "
        "aynı olduğu denetlenir), AYARLAMA YOK. split_v1 train'inde (3630) fit, split_v1 "
        "test'inde (908) tahmin. Bileşenler: split_v2.json 'components' (TF-IDF kosinüsü > 0.8 "
        "grafının bağlı bileşenleri; v1 ve v2 aynı temiz kümeyi böler). 'Sızmış' satır: "
        "bileşeninde en az bir v1 train satırı olan v1 test satırı (zincirler dahil). Birincil "
        "teşhis büyüklüğü: accuracy(sızmış) - accuracy(temiz), %95 yüzdelik grup bootstrap CI "
        "(bileşen düzeyinde ortak w ~ Multinomial(G, 1/G), B=10000, seed 42; bir bileşen iki "
        "alt kümeye bölünmez). Küçük alt kümede macro-F1 farkı verilmez; alt küme başına n, "
        "accuracy ve sınıf sayıları verilir. v1 test'in genel macro-F1/accuracy'si aynı w'lerden "
        "grup bootstrap CI ile, v2 test char_lr macro-F1'i (CI'ıyla) yanında verilir: farklı "
        "test kümeleri, p yok, v1 - v2 farkı sızıntı etkisi olarak sunulmaz. Sınırlılık: sızmış "
        "alt küme rastgele değildir (konu/sınıf bileşimi farklı olabilir), fark yalnız "
        "ezberlemeye atfedilemez; 'temiz' alt küme sızıntıdan arınmış sayılmaz (<= 0.8 "
        "benzerlikli yeniden yazımlar olabilir). char_lr'nin ayarları (C, ngram) split_v2 "
        "train'inde seçildi ve bu train v1 test satırlarının çoğunu içerir; v1 genel skoru bu "
        "yüzden tamamen görülmemiş veride ayarlanmış bir modelin skoru değildir (grid dar, etki "
        "küçük beklenir)."
    ),
    "commits": (
        "Test koşusundan (predict + evaluate) sonra tahmin npz'leri, manifest ve sonuç json'u "
        "YORUMSUZ ayrı bir commit'e girer; yorum ve belge güncellemeleri sonraki commit'tedir."
    ),
}


# --------------------------------------------------------------------------- yardımcılar
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tuples(params: dict) -> dict:
    """json'daki ngram_range listeleri sklearn'ün beklediği tuple'a."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}


def _label_names(path: Path = SPLIT_PATH) -> list[str]:
    return load_split(path)["label_names"]


def env_info() -> dict:
    """Yerel CPU ortamı (finetune.env_info ile aynı alanlar + tokenizers/safetensors/CPU)."""
    import accelerate
    import datasets
    import safetensors
    import scipy
    import sklearn
    import tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
        "dtype": "float32",
    }


# --------------------------------------------------------------------------- donmuş klasik modeller
def frozen_params(name: str, metrics_dir: Path = METRICS_DIR) -> tuple[dict, dict]:
    """(best_params, kaynak json) — Faz 2 (klasik) ya da Faz 3 (bge_m3) son koşusu."""
    phase = 3 if name == ENCODER else 2
    m = latest_metrics(name, metrics_dir, phase=phase)
    if m is None:
        raise FileNotFoundError(f"phase{phase}_{name}_<ts>.json yok")
    if m["split"] != SPLIT_NAME:
        raise ValueError(f"{m['_file']}: split {m['split']}, beklenen {SPLIT_NAME}")
    return _tuples(m["final"]["best_params"]), m


def make_frozen(name: str, metrics_dir: Path = METRICS_DIR):
    """RULES['refit']: spec.make() + donmuş best_params (arama yok). Döner: (pipeline, bilgi)."""
    from ttc4900.frozen import official_spec

    params, m = frozen_params(name, metrics_dir)
    spec = official_spec(name, metrics_dir) if name == ENCODER else SPECS[name]
    if set(params) != set(spec.grid):
        raise ValueError(f"{name}: best_params anahtarları {set(params)} != grid {set(spec.grid)}")
    for k, v in params.items():
        if v not in [_tuples({k: g})[k] for g in spec.grid[k]]:
            raise ValueError(f"{name}: {k}={v} grid'de yok {spec.grid[k]}")
    pipe = spec.make().set_params(**params)
    info = {
        "source": m["_file"],
        "source_commit": m["git_commit"],
        "best_params": params,
        "fixed": spec.fixed,
        "grid": spec.grid,
    }
    return pipe, info


def fit_frozen(pipe, X, y):
    """Tek fit, tüm satırlarla (RULES['refit'])."""
    if len(X) != len(y):
        raise ValueError("X ve y satır sayısı farklı")
    return pipe.fit(X, np.asarray(y))


def _scores(pipe, X) -> tuple[np.ndarray, str]:
    clf = pipe[-1]
    if hasattr(clf, "predict_proba"):
        return np.asarray(pipe.predict_proba(X), dtype=np.float32), "predict_proba"
    return np.asarray(pipe.decision_function(X), dtype=np.float32), "decision_function"


# --------------------------------------------------------------------------- bge_m3 test önbelleği
def test_cache_path(name: str, cache_dir: Path = CACHE_DIR) -> Path:
    spec = ENCODERS[name]
    return Path(cache_dir) / f"emb_{name}_TEST_{spec.revision[:8]}_{spec.max_length}.npz"


test_cache_path.__test__ = False


def _test_meta(name: str, test: pd.DataFrame) -> dict:
    return {
        "side": "test",
        "encoder": name,
        **asdict(ENCODERS[name]),
        "c1clean": False,
        "text_transform": "yok",
        "n": len(test),
        "test_row_ids_sha256": row_ids_hash(test["row_id"]),
        "split": SPLIT_NAME,
    }


def test_embeddings(name: str, test: pd.DataFrame, encode=None, cache_dir: Path = CACHE_DIR):
    """Test embedding'i (Faz 3 ile aynı revision/max_length/pooling); önbellekte varsa doğrulanıp
    okunur. `encode(texts) -> (E, n_trunc)` testler için enjekte edilebilir."""
    p = test_cache_path(name, cache_dir)
    want = _test_meta(name, test)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            meta, row_id, X = json.loads(str(z["meta"])), z["row_id"], z["X"]
        bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        if bad or not np.array_equal(row_id, test["row_id"].to_numpy()):
            raise ValueError(f"{p.name} meta/satırlar uyuşmuyor: {bad}")
        return X, meta
    if encode is None:
        from ttc4900.embeddings import embed_texts, load_model, load_tokenizer

        spec = ENCODERS[name]
        tok, model = load_tokenizer(spec), load_model(spec, "cpu")

        def encode(t):
            return embed_texts(t, tok, model, spec, 16, "cpu")

    E, n_tr = encode(test["text"].tolist())
    meta = want | {
        "n_truncated": int(n_tr),
        "created_at": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        **_versions(),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(
        tmp,
        X=np.asarray(E, dtype=np.float32),
        row_id=test["row_id"].to_numpy(),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    try:
        os.link(tmp, p)  # hedef varsa hata: test önbelleği de üzerine yazılmaz
    finally:
        tmp.unlink()
    return np.asarray(E, dtype=np.float32), meta


test_embeddings.__test__ = False


def _versions() -> dict:
    try:
        import torch
        import transformers

        return {"torch": torch.__version__, "transformers": transformers.__version__}
    except ImportError:
        return {}


def train_features(name: str, train: pd.DataFrame):
    """Faz 3 train önbelleği (meta ve satır sırası doğrulanır)."""
    return load_embeddings(name, train)


# --------------------------------------------------------------------------- BERTurk modelleri
def bert_slots(metrics_dir: Path = METRICS_DIR, models_dir: Path = MODELS_DIR) -> dict[str, dict]:
    """phase4_summary.finals'tan slot başına kullanılan final model: ad -> kayıt/yerel dizin."""
    summ = latest_metrics("summary", metrics_dir, phase=4)
    if summ is None:
        raise FileNotFoundError("phase4_summary_<ts>.json yok")
    fin = summ["finals"]
    if fin.get("missing_slots"):
        raise ValueError(f"eksik final slotları: {fin['missing_slots']}")
    out = {}
    for slot in BERT_SLOTS:
        s = fin["slots"][str(slot)]
        rec = json.loads((Path(metrics_dir) / s["used_file"]).read_text(encoding="utf-8"))
        rec["_file"] = s["used_file"]
        check_config(rec)
        if rec.get("final_collapse_rule") != FINAL_RULES["final_collapse"]:
            raise ValueError(f"{s['used_file']}: final_collapse_rule farklı")
        if rec["model_dir"] != s["used_model_dir"] or rec["split"] != SPLIT_NAME:
            raise ValueError(f"{s['used_file']}: model_dir/split özetle tutarsız")
        local = Path(models_dir) / re.split(r"[\\/]", s["used_model_dir"])[-1]
        out[bert_name(slot)] = {
            "slot": slot,
            "seed": s["used_seed"],
            "file": s["used_file"],
            "recorded_model_dir": s["used_model_dir"],
            "local_dir": local,
            "weights_sha256": rec["weights_sha256"],
            "train_in_sample_macro_f1": rec["train_in_sample_macro_f1"],
            "tokenizer_check": rec.get("tokenizer_check"),
            "primary_for_test": bool(rec.get("primary_for_test")),
        }
    prim = out[BERT_NAME]
    if not prim["primary_for_test"] or fin["primary_model_dir"] != prim["recorded_model_dir"]:
        raise ValueError("birincil model phase4_summary.finals.primary_model_dir ile tutarsız")
    if fin["primary_model_sha256"] != prim["weights_sha256"]:
        raise ValueError("phase4_summary.primary_model_sha256 final json'la tutarsız")
    return out


def load_bert(local_dir: Path, names: list[str]):
    """Yerel final modeli (CPU, fp32, eval). Döner: (texts -> logits fonksiyonu, denetim bilgisi).

    Tokenizer denetimi: sabit revision'ın vocab.txt'si (finetune.check_tokenizer_revision) ve
    Hub revision tokenizer'ıyla aynı input_ids (`same_ids` ile train metinlerinde)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ttc4900.finetune import check_tokenizer_revision
    from ttc4900.finetune import load_tokenizer as hub_tokenizer

    tok = AutoTokenizer.from_pretrained(local_dir)
    tok_check = check_tokenizer_revision(tok)
    hub = hub_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(local_dir).to("cpu").eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    max_len = FT_CONFIG["max_length"]

    def ids(t, texts):
        return t(list(texts), truncation=True, max_length=max_len)["input_ids"]

    def same_ids(texts) -> bool:
        return ids(tok, texts) == ids(hub, texts)

    def logits(texts, batch_size: int = 16) -> np.ndarray:
        texts = list(texts)
        lengths = np.array([len(x) for x in ids(tok, texts)])
        order = np.argsort(-lengths, kind="stable")
        out = np.zeros((len(texts), len(names)), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                idx = order[s : s + batch_size]
                batch = tok(
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                out[idx] = model(**batch).logits.float().numpy()
        return out

    info = {
        "id2label_ok": id2label == label_maps(names)[0],
        "tokenizer_check": tok_check,
        "same_ids": same_ids,
    }
    return logits, info


# --------------------------------------------------------------------------- 1) verify
def verify_models(
    metrics_dir: Path = METRICS_DIR,
    models_dir: Path = MODELS_DIR,
    out_dir: Path = METRICS_DIR,
    train: pd.DataFrame | None = None,
    loader: Callable = load_bert,
    path: Path = SPLIT_PATH,
    env: Callable = env_info,
) -> tuple[Path, dict]:
    """RULES['verify']: yalnız train. Döner: (json yolu, kayıt); kayıt['passed']."""
    train = load_train(path) if train is None else train  # test belleğe alınmaz
    names = _label_names(path)
    check_labels(train, names)
    y = train["label"].to_numpy()
    started = report.now()
    rows = {}
    for name, s in bert_slots(metrics_dir, models_dir).items():
        t0 = time.perf_counter()
        sha = weights_sha256(s["local_dir"])
        row = {
            "slot": s["slot"],
            "seed": s["seed"],
            "source": s["file"],
            "local_dir": str(Path(s["local_dir"]).relative_to(PROJECT_ROOT))
            if Path(s["local_dir"]).is_relative_to(PROJECT_ROOT)
            else str(s["local_dir"]),
            "recorded_model_dir": s["recorded_model_dir"],
            "primary_for_test": s["primary_for_test"],
            "weights_sha256": sha,
            "weights_sha256_recorded": s["weights_sha256"],
            "sha256_ok": sha == s["weights_sha256"],
            "model_files_sha256": model_files_sha256(s["local_dir"]),
        }
        if row["sha256_ok"]:
            logits_fn, info = loader(s["local_dir"], names)
            texts = train["text"].tolist()
            rec_tok = (s.get("tokenizer_check") or {}).get("vocab_sha256")
            row["id2label_ok"] = bool(info["id2label_ok"])
            row["tokenizer_vocab_ok"] = info["tokenizer_check"]["vocab_sha256"] == rec_tok
            row["tokenizer_same_ids_as_hub"] = bool(info["same_ids"](texts))
            pred = np.asarray(logits_fn(texts)).argmax(-1)
            ins = oof_metrics(y, pred, names)
            diff = ins["macro_f1"] - s["train_in_sample_macro_f1"]
            row |= {
                "train_in_sample_macro_f1": ins["macro_f1"],
                "train_in_sample_accuracy": ins["accuracy"],
                "train_in_sample_macro_f1_recorded": s["train_in_sample_macro_f1"],
                "abs_diff": abs(diff),
                "in_sample_ok": abs(diff) <= IN_SAMPLE_TOL,
                "not_collapsed": ins["macro_f1"] >= FT_CONFIG["collapse_threshold"],
            }
        checks = (
            "sha256_ok",
            "id2label_ok",
            "tokenizer_vocab_ok",
            "tokenizer_same_ids_as_hub",
            "in_sample_ok",
            "not_collapsed",
        )
        row["passed"] = all(row.get(k) is True for k in checks)
        row["runtime_seconds"] = round(time.perf_counter() - t0, 1)
        rows[name] = row
    out = {
        "experiment": "phase5_model_check",
        "phase": 5,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": Path(path).stem,
        "n_train": len(train),
        "rule": RULES["verify"],
        "in_sample_tolerance": IN_SAMPLE_TOL,
        "env": env(),
        "models": rows,
        "passed": all(r["passed"] for r in rows.values()),
        "test": None,
    }
    p = report.write_metrics(out, "phase5_model_check", out_dir, when=started)
    return p, out


# --------------------------------------------------------------------------- 2) predict
# verify ile predict arasında değişmemesi gereken girdiler: kod, bağımlılıklar, donmuş parametre
# json'ları (Faz 2/3/4), split ve yakın-tekrar dosyaları. Faz 5'in kendi çıktıları hariç.
GATED_PATHS = ("src", "requirements.txt", "outputs/metrics", "outputs/splits", "outputs/dedup")
_PHASE5_OUTPUT = re.compile(
    r"outputs/metrics/(phase5_[a-z_]+_\d{8}-\d{6}\.json|preds/test_\w+\.npz)"
)


def _git(*args: str) -> list[str]:
    run = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                         check=True)  # fmt: skip
    return run.stdout.split()


def gate_filter(name_status: list[str]) -> list[str]:
    """`git diff --name-status` satırlarından kapıyı tetikleyenler: yalnız YENİ EKLENEN (A) Faz 5
    çıktıları muaftır; değişen/silinen/yeniden adlanan her dosya (Faz 5 json'ları dahil) sayılır."""
    out = []
    for line in name_status:
        status, *paths = line.split("\t")
        if status == "A" and _PHASE5_OUTPUT.fullmatch(paths[-1]):
            continue
        out.append(paths[-1])
    return out


def _src_changed_since(commit: str) -> list[str]:
    """verify commit'inden HEAD'e GATED_PATHS'te kapıyı tetikleyen değişiklikler."""
    run = subprocess.run(
        ["git", "diff", "--name-status", commit, "HEAD", "--", *GATED_PATHS],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return gate_filter(run.stdout.splitlines())


def _is_tracked(path: Path) -> bool:
    """Dosya git'te izleniyor mu (proje dışındaysa hayır)."""
    p = Path(path).resolve()
    if not p.is_relative_to(PROJECT_ROOT):
        return False
    return bool(_git("ls-files", "--", p.relative_to(PROJECT_ROOT).as_posix()))


def model_files_sha256(model_dir: Path) -> dict[str, str]:
    """Model dizinindeki TÜM dosyaların sha256'sı (ağırlık + config + tokenizer)."""
    return {f.name: sha256_file(f) for f in sorted(Path(model_dir).iterdir()) if f.is_file()}


def _untracked_inputs() -> list[str]:
    """GATED_PATHS'te izlenmeyen dosyalar (report.git_commit bunları görmez): ör. sonradan düşen
    bir phase2_*.json latest_metrics'te 'en son' seçilirdi. Faz 5 çıktıları hariç."""
    found = _git("ls-files", "--others", "--exclude-standard", "--", *GATED_PATHS)
    return [f for f in found if not _PHASE5_OUTPUT.fullmatch(f)]


def check_predict_ready(metrics_dir: Path = METRICS_DIR) -> dict:
    """RULES['test_access'] ön koşulları; tutmazsa hata (test açılmaz)."""
    head = report.git_commit()
    if head == "unknown" or head.endswith("-dirty"):
        raise RuntimeError(f"çalışma ağacı temiz değil ya da git yok ({head}); test açılmaz")
    chk = latest_metrics("model_check", metrics_dir, phase=5)
    if chk is None:
        raise RuntimeError("phase5_model_check json'u yok: önce `final_eval verify`")
    if not chk.get("passed"):
        raise RuntimeError(f"{chk['_file']}: model doğrulaması geçmedi; test açılmaz")
    if chk["git_commit"].endswith("-dirty") or chk["git_commit"] == "unknown":
        raise RuntimeError(f"{chk['_file']}: doğrulama temiz bir commit'te yapılmamış")
    changed = _src_changed_since(chk["git_commit"])
    if changed:
        raise RuntimeError(f"doğrulamadan sonra kod/girdi değişmiş: {changed}; verify yeniden")
    untracked = _untracked_inputs()
    if untracked:
        raise RuntimeError(f"izlenmeyen girdi dosyaları var: {untracked}; test açılmaz")
    return {
        "git_commit": head,
        "model_check": chk["_file"],
        "model_check_commit": chk["git_commit"],
    }


def write_once(path: Path, **arrays) -> str:
    """npz'i bir kez yazar: geçici dosya + os.link (hedef varsa FileExistsError). sha256 döner."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} zaten var; üzerine yazılmaz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return sha256_file(path)


def preds_path(name: str, preds_dir: Path = PREDS_DIR) -> Path:
    return Path(preds_dir) / f"test_{name}.npz"


def _manifest_shas(metrics_dir: Path) -> dict[str, str]:
    """Önceki predict manifest'lerindeki sha256'lar (ilk kayıt esas)."""
    out: dict[str, str] = {}
    pat = re.compile(r"phase5_test_preds_\d{8}-\d{6}\.json")
    for f in sorted(Path(metrics_dir).glob("phase5_test_preds_*.json")):
        if pat.fullmatch(f.name):
            for n, e in json.loads(f.read_text(encoding="utf-8"))["files"].items():
                out.setdefault(n, e["sha256"])
    return out


def _check_existing(p: Path, test: pd.DataFrame, known_sha: str | None) -> str:
    sha = sha256_file(p)
    if known_sha is not None and sha != known_sha:
        raise ValueError(f"{p.name}: sha256 önceki manifest'tekiyle aynı değil (dosya değişmiş)")
    with np.load(p, allow_pickle=False) as z:
        ok = np.array_equal(z["row_id"], test["row_id"].to_numpy()) and np.array_equal(
            z["y_true"], test["label"].to_numpy()
        )
    if not ok:
        raise ValueError(f"{p.name}: satırlar/etiketler bu test kümesiyle aynı değil")
    return sha


def predict(
    metrics_dir: Path = METRICS_DIR,
    models_dir: Path = MODELS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    path: Path = SPLIT_PATH,
    load_data: Callable = load_train_test,
    train_loader: Callable = load_train,
    loader: Callable = load_bert,
    features: Callable = train_features,
    encode=None,
    cache_dir: Path = CACHE_DIR,
    env: Callable = env_info,
    ready: Callable = check_predict_ready,
) -> tuple[Path, dict]:
    """Testi BİR KEZ açar; eksik test_<model>.npz'leri yazar (var olanlar yalnız denetlenir)."""
    gate = ready(metrics_dir)  # hata: test açılmaz
    names = _label_names(path)
    slots = bert_slots(metrics_dir, models_dir)
    chk = latest_metrics("model_check", metrics_dir, phase=5)
    for n, s in slots.items():  # dosyalar doğrulamadan sonra değişmemiş olmalı (ucuz)
        if weights_sha256(s["local_dir"]) != s["weights_sha256"]:
            raise RuntimeError(f"{n}: ağırlık sha256'sı kayıtla aynı değil; test açılmaz")
        # config/tokenizer dahil tüm model dosyaları verify'dakiyle aynı (gitignore'lu dizin)
        if chk is not None and model_files_sha256(s["local_dir"]) != chk["models"][n].get(
            "model_files_sha256"
        ):
            raise RuntimeError(f"{n}: model dosyaları verify'dan sonra değişmiş; test açılmaz")
    known = _manifest_shas(out_dir)
    todo = [n for n in CANDIDATES if not preds_path(n, preds_dir).exists()]
    started = report.now()
    environment = env()
    # 1) Fit'ler: yalnız train, test metni henüz bellekte değil
    train = train_loader(path) if todo else None
    fitted, infos = {}, {}
    for n in (n for n in todo if n in (*CLASSICAL, ENCODER)):
        t0 = time.perf_counter()
        pipe, info = make_frozen(n, metrics_dir)
        X = features(n, train) if n == ENCODER else train["text"]
        fitted[n] = fit_frozen(pipe, X, train["label"])
        infos[n] = info | {"n_fit": len(train), "fit_seconds": round(time.perf_counter() - t0, 1)}
        if n == ENCODER:  # gitignore'lu önbellek: içerik özeti tekrar üretilebilirlik için
            arr = np.ascontiguousarray(X, dtype=np.float32)
            infos[n]["train_embedding_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    # 2) Test (yalnız burada)
    tr2, test = load_data(path)
    if train is not None and not np.array_equal(tr2["row_id"], train["row_id"]):
        raise ValueError("load_train_test train'i load_train ile aynı sırada değil")
    check_labels(test, names)
    if not np.array_equal(np.sort(test["row_id"]), np.sort(load_split(path)["test_ids"])):
        raise ValueError("test satırları split dosyasıyla aynı değil")
    del tr2
    files, runtimes = {}, {}
    y = test["label"].to_numpy()
    base = {"row_id": test["row_id"].to_numpy(), "y_true": y}

    def manifest(done: bool) -> tuple[Path, dict]:
        """Her npz'den hemen sonra (aynı dosya adına) yazılır: yarıda kalan koşuda da yazılmış
        her npz'in sha256'sı kayıtlıdır (RULES['write_once'])."""
        out = {
            "experiment": "phase5_test_preds",
            "phase": 5,
            "timestamp": started.isoformat(timespec="seconds"),
            **gate,
            "split": Path(path).stem,
            "complete": done,
            "n_test": len(test),
            "test_row_ids_sha256": row_ids_hash(test["row_id"]),
            "files": files,
            "models": infos,
            "predict_seconds": runtimes,
            "env": environment,
            "rules": {k: RULES[k] for k in ("refit", "test_access", "write_once")},
        }
        return report.write_metrics(out, "phase5_test_preds", out_dir, when=started), out

    for n in CANDIDATES:
        p = preds_path(n, preds_dir)
        if p.exists():
            files[n] = {"file": p.name, "sha256": _check_existing(p, test, known.get(n)),
                        "status": "vardı (yeniden üretilmedi)"}  # fmt: skip
            continue
        t0 = time.perf_counter()
        if n in fitted:
            X = test["text"]
            if n == ENCODER:
                X, emb_meta = test_embeddings(n, test, encode, cache_dir)
                arr = np.ascontiguousarray(X, dtype=np.float32)
                infos[n]["test_embedding_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
                infos[n]["test_embedding"] = {
                    "file": test_cache_path(n, cache_dir).name,
                    **{k: emb_meta.get(k) for k in ("side", "n", "test_row_ids_sha256",
                                                     "n_truncated", "revision", "max_length",
                                                     "pooling")},
                }  # fmt: skip
            scores, kind = _scores(fitted[n], X)
            y_pred = np.asarray(fitted[n].predict(X))
            meta = {"model": n, "kind": "frozen_refit", **infos[n], "score_kind": kind}
        else:
            s = slots[n]
            logits_fn, info = loader(s["local_dir"], names)
            if not info["id2label_ok"]:
                raise ValueError(f"{n}: id2label split label_names ile aynı değil")
            scores, kind = np.asarray(logits_fn(test["text"].tolist()), np.float32), "logits"
            y_pred = scores.argmax(-1)
            meta = {"model": n, "kind": "phase4_final", "slot": s["slot"], "seed": s["seed"],
                    "source": s["file"], "weights_sha256": s["weights_sha256"],
                    "score_kind": kind}  # fmt: skip
        runtimes[n] = round(time.perf_counter() - t0, 1)
        meta |= {"git_commit": gate["git_commit"], "split": Path(path).stem,
                 "created_at": report.now().isoformat(timespec="seconds"),
                 "predict_seconds": runtimes[n]}  # fmt: skip
        sha = write_once(
            p, **base, y_pred=np.asarray(y_pred, dtype=np.int64), scores=scores,
            meta=np.array(json.dumps(report.to_jsonable(meta), ensure_ascii=False)),
        )  # fmt: skip
        files[n] = {"file": p.name, "sha256": sha, "status": "üretildi"}
        manifest(done=False)
        print(f"{n}: {p.name} ({runtimes[n]:.0f} sn)", flush=True)
    return manifest(done=True)


# --------------------------------------------------------------------------- 3) evaluate
def load_test_preds(preds_dir: Path = PREDS_DIR, path: Path = SPLIT_PATH) -> dict[str, dict]:
    """Tüm adayların test npz'leri; satırlar/etiketler hepsinde aynı ve split'in test'i."""
    out = {}
    for n in CANDIDATES:
        p = preds_path(n, preds_dir)
        if not p.exists():
            raise FileNotFoundError(f"{p.name} yok: önce `final_eval predict`")
        with np.load(p, allow_pickle=False) as z:
            out[n] = {k: z[k] for k in ("row_id", "y_true", "y_pred")} | {
                "sha256": sha256_file(p),
                "meta": json.loads(str(z["meta"])),
            }
    ref = out[CANDIDATES[0]]
    for n, d in out.items():
        if not (
            np.array_equal(d["row_id"], ref["row_id"])
            and np.array_equal(d["y_true"], ref["y_true"])
        ):
            raise ValueError(f"{n}: satırlar/etiketler diğer modellerle aynı değil")  # fmt: skip
    if not np.array_equal(np.sort(ref["row_id"]), np.sort(load_split(path)["test_ids"])):
        raise ValueError("npz satırları split'in test'i değil")
    return out


def _power_lookup(metrics_dir: Path) -> dict[tuple[str, str], dict]:
    pat = re.compile(r"test_power_\d{8}-\d{6}\.json")
    files = sorted(f for f in Path(metrics_dir).glob("test_power_*.json") if pat.fullmatch(f.name))
    if not files:
        return {}
    pw = json.loads(files[-1].read_text(encoding="utf-8"))
    return {(r["a"], r["b"]): r | {"_file": files[-1].name} for r in pw["pairs"] if "by_alpha" in r}


def power_text(row: dict | None, m: int) -> str:
    """Önceden (train OOF'tan) hesaplanan güç ve MDE, α başına; ikincilde Holm'un en kötü (α/m)
    ve en iyi (α) adımı ayrı ayrı."""
    if not row:
        return "güç hesaplanmadı"
    parts = []
    for k in sorted(row["by_alpha"], key=float):
        v, a = row["by_alpha"][k], float(k)
        if m == 1:
            label = f"α = {a:g}"
        elif np.isclose(a, ALPHA / m):
            label = f"α/{m} = {a:g} (Holm'un en kötü adımı)"
        else:
            label = f"α = {a:g} (Holm'un en iyi adımı)"
        parts.append(
            f"{label}: güç {v['power_at_oof_acc_diff']:.2f}, accuracy MDE {v['mde_acc']:.3f}, "
            f"macro-F1 MDE (kaba) {v['mde_macro_f1_rough']:.3f}"
        )
    return (
        "; ".join(parts)
        + f" ({row['_file']}; train OOF'tan yaklaşık: OOF modelleri ~2904 satırla, BERT ~2323 "
        "satırla eğitildi; güç, gerçek farkın OOF accuracy farkına "
        f"({row['oof_acc_diff_mean']:+.4f}) eşit olduğu varsayımıyla; bu fark çok küçükse güç "
        "I. tip hata düzeyine yaklaşır ve MDE daha bilgilendiricidir"
        + (
            "; birincil çiftte OOF farkı kazananın laneti açısından aşağı yanlıdır, kazanan "
            "char_lr b tarafındadır)"
            if row.get("role") == "primary"
            else ")"
        )
    )


_CONDITIONS = {
    "p_holm_below_alpha": f"Holm p < {ALPHA}",
    "ci_excludes_zero": "CI 0'ı dışlıyor",
    "directions_consistent": "McNemar ve CI yönleri tutarlı",
}


def _side(sign: int, a: str, b: str) -> str:
    return {1: f"{a} lehine", -1: f"{b} lehine", 0: "yönsüz"}[int(sign)]


def render_comparison(r: dict, family: str, m: int, power: dict | None = None) -> str:
    """decide() çıktısına göre SENTENCES şablonu (sonuç dili test görülmeden donduruldu)."""
    lo, hi = r["diff_ci95"]
    ci_sign = 1 if lo > 0 else (-1 if hi < 0 else 0)
    common = {
        "a": r["a"],
        "b": r["b"],
        "diff": r["diff"],
        "lo": lo,
        "hi": hi,
        "n10": r["n10"],
        "n01": r["n01"],
        "p": r["p"],
        "p_holm": r["p_holm"],
        "aile": family,
        "m": m,
    }
    if r["verdict"] == "ayırt edildi":
        s = SENTENCES["ayırt edildi"].format(
            **common, yon="daha yüksek" if r["direction"] == "a > b" else "daha düşük"
        )
    elif r["verdict"] == "ayırt edilemedi (yön çelişkisi)":
        s = SENTENCES["ayırt edilemedi (yön çelişkisi)"].format(
            **common,
            mc_yon=_side(np.sign(r["n10"] - r["n01"]), r["a"], r["b"]),
            ci_yon=_side(ci_sign, r["a"], r["b"]),
        )
    else:
        met = [t for k, t in _CONDITIONS.items() if r[k]]
        unmet = [t for k, t in _CONDITIONS.items() if not r[k]]
        if ci_sign == 0:  # CI işaretsizken yön tutarlılığı tanımsız: "sağlanmadı" denmez
            dc = _CONDITIONS["directions_consistent"]
            unmet = [t for t in unmet if t != dc] + [
                f"{dc} (CI 0'ı içerdiği için değerlendirilemez)"
            ]
        s = SENTENCES["ayırt edilemedi"].format(
            **common,
            saglananlar=", ".join(met) or "hiçbiri",
            saglanmayanlar=", ".join(unmet),
            guc=power_text(power, m),
        )
    if family == "birincil":
        s += " " + SENTENCES["birincil_kosul"]
    return s


def _ci_sign(ci) -> int:
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def render_seed_consistency(prim: dict, others: list[dict]) -> str:
    """Birincil farkın CI işareti ile diğer seed'lerin (betimsel çiftler) CI işaretleri."""
    ps = _ci_sign(prim["diff_ci95"])
    prim_dir = {1: f"{prim['a']} lehine", -1: f"{prim['b']} lehine", 0: "işaretsiz (0'ı içeriyor)"}
    parts = []
    for d in others:
        s, (lo, hi) = _ci_sign(d["diff_ci95"]), d["diff_ci95"]
        if s == 0:
            rel = "CI 0'ı içeriyor"
        elif ps == 0:
            rel = f"CI {d['a'] if s > 0 else d['b']} lehine, birincil CI işaretsiz"
        else:
            rel = "aynı yön" if s == ps else "ters yön"
        parts.append(f"{d['a']} − {d['b']} fark {d['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]: {rel}")
    return SENTENCES["seed_tutarlilik"].format(
        birincil_yon=prim_dir[ps], ayrintilar="; ".join(parts)
    )


def render_descriptive(d: dict) -> str:
    lo, hi = d["diff_ci95"]
    return SENTENCES["betimsel"].format(a=d["a"], b=d["b"], diff=d["diff"], lo=lo, hi=hi,
                                        fa=d["macro_f1_a"], fb=d["macro_f1_b"])  # fmt: skip


def accuracy_bootstrap(
    y, preds: dict[str, np.ndarray], groups, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> dict[str, np.ndarray]:
    """cluster_bootstrap ile AYNI w'lerden (paired_test.bootstrap_weights) accuracy replikatları:
    acc* = Σ_g w_g c_g / Σ_g w_g n_g (c_g: gruptaki doğru sayısı). Betimseldir; karar kuralına
    girmez, McNemar'ın (accuracy) yönünü grup bağımlılığını hesaba katarak gösterir."""
    y = np.asarray(y)
    inv, G = group_index(groups)
    n_g = np.bincount(inv, minlength=G).astype(float)
    c = {
        m: np.bincount(inv, weights=(np.asarray(p) == y).astype(float), minlength=G)
        for m, p in preds.items()
    }
    out = {m: np.empty(B) for m in preds}
    for start, w in bootstrap_weights(G, B, seed):
        tot = w @ n_g
        for m in preds:
            out[m][start : start + len(w)] = (w @ c[m]) / tot
    return out


def _acc_effect(y, pa, pb, ra: np.ndarray, rb: np.ndarray) -> dict:
    y = np.asarray(y)
    return {
        "acc_diff": float((np.asarray(pa) == y).mean() - (np.asarray(pb) == y).mean()),
        "acc_diff_ci95": _pct_ci(ra - rb),
    }


def evaluate(
    metrics_dir: Path = METRICS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    path: Path = SPLIT_PATH,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Path, dict]:
    """RULES['evaluate']: TEST_PLAN aileleri, betimsel çiftler, seed özeti ve cümleler."""
    P = load_test_preds(preds_dir, path)
    names = _label_names(path)
    ref = P[CANDIDATES[0]]
    y = ref["y_true"]
    g = test_groups(pd.DataFrame({"row_id": ref["row_id"]}), path)
    preds = {n: d["y_pred"] for n, d in P.items()}
    reps = cluster_bootstrap(y, preds, g, B, seed)  # tüm modeller aynı w'ler
    acc_reps = accuracy_bootstrap(y, preds, g, B, seed)  # aynı w'ler
    models = {
        n: oof_metrics(y, preds[n], names)
        | {"macro_f1_ci95": _pct_ci(reps[n]), "accuracy_ci95": _pct_ci(acc_reps[n])}
        for n in CANDIDATES
    }
    power = _power_lookup(metrics_dir)
    fams, sentences = {}, {"aileler": {}, "betimsel": [], "seed": None}
    for fam, tr in (("primary", "birincil"), ("secondary", "ikincil")):
        comps = FAMILIES[fam]["comparisons"]
        rows = compare_family(y, preds, comps, g, ALPHA, B, seed)
        for r in rows:  # betimsel ek: accuracy farkının grup bootstrap CI'ı (karara girmez)
            r |= _acc_effect(y, preds[r["a"]], preds[r["b"]], acc_reps[r["a"]], acc_reps[r["b"]])
        fams[fam] = {"m": len(comps), "comparisons": rows}
        sentences["aileler"][fam] = [
            render_comparison(r, tr, len(comps), power.get((r["a"], r["b"]))) for r in rows
        ]
    desc = []
    for a, b in DESCRIPTIVE:
        d = {"a": a, "b": b, "inference": "betimsel; p hesaplanmaz"}
        d |= _effect(y, preds[a], preds[b], reps[a], reps[b])
        d |= _acc_effect(y, preds[a], preds[b], acc_reps[a], acc_reps[b])
        desc.append(d)
        sentences["betimsel"].append(render_descriptive(d))
    bert = [bert_name(s) for s in BERT_SLOTS]
    prim = fams["primary"]["comparisons"][0]
    others = [d for d in desc if d["a"] in bert[1:] and d["b"] == prim["b"]]
    sentences["seed_tutarlilik"] = render_seed_consistency(prim, others)
    # kullanıcı kararı: tutarlılık satırı birincil cümlenin parçasıdır (tek başına alıntılanmasın)
    sentences["aileler"]["primary"][0] += " " + sentences["seed_tutarlilik"]
    f1s = np.array([models[n]["macro_f1"] for n in bert])
    seed_summary = {
        "models": bert,
        "macro_f1": dict(zip(bert, f1s.tolist(), strict=True)),
        "mean": float(f1s.mean()),
        "std_ddof1": float(f1s.std(ddof=1)),
        "note": "seed değişkenliği (3 eğitim); test örneklemesi belirsizliği değil",
    }
    sentences["seed"] = SENTENCES["seed_ozeti"].format(
        mean=seed_summary["mean"], std=seed_summary["std_ddof1"], k=len(bert),
        primary=BERT_NAME, seed=PRIMARY_SEED, lo=f1s.min(), hi=f1s.max(),
        values=", ".join(f"{n} {v:.4f}" for n, v in zip(bert, f1s, strict=True)),
    )  # fmt: skip
    sentences["cv_test"] = SENTENCES["cv_test_notu"]
    sentences["sinirliliklar"] = SENTENCES["sinirliliklar"]
    sizes = pd.Series(g).value_counts()
    out = {
        "experiment": "phase5_test",
        "phase": 5,
        "timestamp": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": Path(path).stem,
        "n_test": int(len(y)),
        "test_group_structure": {
            "n_groups": int(len(sizes)),
            "group_size_hist": {
                str(int(k)): int(v) for k, v in sizes.value_counts().sort_index().items()
            },
        },
        "preds": {
            n: {
                "file": preds_path(n, preds_dir).name,
                "sha256": d["sha256"],
                "git_commit": d["meta"].get("git_commit"),
            }
            for n, d in P.items()
        },  # fmt: skip
        "bootstrap": {"B": B, "seed": seed, "unit": "split_v2 components (test_groups)"},
        "models": models,
        "families": fams,
        "descriptive": desc,
        "seed_summary": seed_summary,
        "sentences": sentences,
        "test_plan": TEST_PLAN,
        "rules": {k: RULES[k] for k in ("evaluate", "commits")},
        "power_source": next(iter(power.values()))["_file"] if power else None,
    }
    return report.write_metrics(out, "phase5_test", out_dir), out


# --------------------------------------------------------------------------- 4) v1 teşhisi
def _component_of(components) -> dict[int, int]:
    """row_id -> bileşen id'si (bileşenin en küçük row_id'si; data._group_ids ile aynı)."""
    return {int(i): int(c[0]) for c in components for i in c}


def leaked_rows(train_ids, test_ids, components) -> set[int]:
    """v1 test satırlarından, yakın-tekrar bileşeninde (split_v2 components: kosinüs > 0.8
    grafının bağlı bileşenleri) en az bir v1 train satırı olanlar. Zincirler (test-test-train)
    de sayılır."""
    comp = _component_of(components)
    with_train = {comp[i] for i in map(int, train_ids) if i in comp}
    return {int(i) for i in test_ids if comp.get(int(i)) in with_train}


def v1_groups(test_ids, components) -> np.ndarray:
    """v1 test satırlarının bootstrap grupları: bileşen id'si; tekil satır -(row_id+1)."""
    comp = _component_of(components)
    return np.array([comp.get(int(r), -(int(r) + 1)) for r in test_ids], dtype=np.int64)


def group_subset_bootstrap(
    y, pred, groups, mask, B: int = V1_B, seed: int = BOOTSTRAP_SEED
) -> tuple[np.ndarray, int]:
    """accuracy(mask) - accuracy(~mask), grup bootstrap'iyle: tüm test grupları ortak
    w ~ Multinomial(G, 1/G) ile örneklenir (paired_test.bootstrap_weights);
    acc_L* = Σ_{g∈L} w_g c_g / Σ_{g∈L} w_g n_g. Bir grup iki alt kümeye bölünemez. Paydası 0
    olan replikatlar düşer. Döner: (fark replikatları, düşen sayısı)."""
    y, pred, mask = np.asarray(y), np.asarray(pred), np.asarray(mask, dtype=bool)
    inv, G = group_index(groups)
    n_g = np.bincount(inv, minlength=G).astype(float)
    m_g = np.bincount(inv, weights=mask.astype(float), minlength=G)
    if not np.all((m_g == 0) | (m_g == n_g)):
        raise ValueError("bir grup iki alt kümeye bölünmüş (sızma bileşen düzeyinde olmalı)")
    L = m_g > 0
    c = np.bincount(inv, weights=(y == pred).astype(float), minlength=G)
    d = np.empty(B)
    for start, w in bootstrap_weights(G, B, seed):
        with np.errstate(invalid="ignore", divide="ignore"):
            a_l = (w[:, L] @ c[L]) / (w[:, L] @ n_g[L])
            a_c = (w[:, ~L] @ c[~L]) / (w[:, ~L] @ n_g[~L])
        d[start : start + len(w)] = a_l - a_c
    ok = np.isfinite(d)
    return d[ok], int((~ok).sum())


def v1_diagnosis(
    metrics_dir: Path = METRICS_DIR,
    preds_dir: Path = PREDS_DIR,
    out_dir: Path = METRICS_DIR,
    load_data: Callable = load_train_test,
    v1_path: Path = SPLIT_V1_PATH,
    groups_path: Path = SPLIT_PATH,
    B: int = V1_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Path, dict]:
    """RULES['v1_diagnosis']: ana sonuçlardan sonra, char_lr dondurulmuş ayarlarla, ayar yok."""
    head = report.git_commit()
    if head == "unknown" or head.endswith("-dirty"):
        raise RuntimeError(f"çalışma ağacı temiz değil ({head}); v1 teşhisi temiz commit'te koşar")
    main_res = latest_metrics("test", out_dir, phase=5)
    if main_res is None:
        raise RuntimeError("phase5_test json'u yok: v1 teşhisi ana sonuçlardan SONRA çalışır")
    if not _is_tracked(Path(out_dir, main_res["_file"])):
        raise RuntimeError(f"{main_res['_file']} commit'lenmemiş: önce ana sonuç commit'i")
    untracked = _untracked_inputs()
    if untracked:
        raise RuntimeError(f"izlenmeyen girdi dosyaları var: {untracked}; v1 teşhisi koşmaz")
    pipe, info = make_frozen(V1_MODEL, metrics_dir)
    if info["best_params"] != V1_FROZEN or pipe.get_params()["tfidf__analyzer"] != "char_wb":
        raise ValueError(f"char_lr donmuş ayarları {info['best_params']} != {V1_FROZEN}/char_wb")
    split1, split2 = load_split(v1_path), load_split(groups_path)
    names = split1["label_names"]
    if set(split1["train_ids"]) | set(split1["test_ids"]) != set(split2["train_ids"]) | set(
        split2["test_ids"]
    ):
        raise ValueError("v1 ve v2 aynı temiz veri kümesini bölmüyor; bileşenler uygulanamaz")
    comps = split2["components"]
    leaked = leaked_rows(split1["train_ids"], split1["test_ids"], comps)
    t0, started = time.perf_counter(), report.now()
    train1, test1 = load_data(v1_path)
    fit_frozen(pipe, train1["text"], train1["label"])
    y, pred = test1["label"].to_numpy(), np.asarray(pipe.predict(test1["text"]))
    mask = test1["row_id"].isin(leaked).to_numpy()
    g = v1_groups(test1["row_id"], comps)
    acc_d, n_drop = group_subset_bootstrap(y, pred, g, mask, B, seed)
    f1_reps = cluster_bootstrap(y, {"v1": pred}, g, B, seed)["v1"]
    acc_reps = accuracy_bootstrap(y, {"v1": pred}, g, B, seed)["v1"]

    def subset(m: np.ndarray) -> dict:
        counts = np.bincount(y[m], minlength=len(names))
        return {
            "n": int(m.sum()),
            "n_groups": int(len(np.unique(g[m]))),
            "accuracy": float((y[m] == pred[m]).mean()),
            "class_counts": dict(zip(names, counts.tolist(), strict=True)),
        }

    sub = {"leaked": subset(mask), "clean": subset(~mask)}
    p_npz = Path(preds_dir) / "test_v1_char_lr.npz"
    sha = write_once(
        p_npz, row_id=test1["row_id"].to_numpy(), y_true=y, y_pred=pred.astype(np.int64),
        leaked=mask, groups=g,
        meta=np.array(json.dumps({"split": "split_v1", "model": V1_MODEL})),
    )  # fmt: skip
    v2 = main_res["models"][V1_MODEL]
    out = {
        "experiment": "phase5_v1_leakage",
        "phase": 5,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": "split_v1",
        "rule": RULES["v1_diagnosis"],
        "model": V1_MODEL,
        "frozen_params": info,
        "n_train": len(train1),
        "n_test": len(test1),
        "components_source": {
            "file": Path(groups_path).name,
            "sha256": sha256_file(groups_path),
            "threshold": NEAR_DUP_THRESHOLD,
        },
        "overall_v1_test": oof_metrics(y, pred, names)
        | {"macro_f1_ci95": _pct_ci(f1_reps), "accuracy_ci95": _pct_ci(acc_reps)},
        "subsets": sub,
        "diagnostic": {
            "acc_leaked_minus_clean": sub["leaked"]["accuracy"] - sub["clean"]["accuracy"],
            "acc_diff_ci95": _pct_ci(acc_d),
            "bootstrap": {
                "B": B,
                "seed": seed,
                "scheme": "split_v2 components düzeyinde grup bootstrap, ortak w",
                "n_dropped_replicates": n_drop,
            },
        },
        "v2_reference": {
            "source": main_res["_file"],
            "char_lr_v2_test_macro_f1": v2["macro_f1"],
            "char_lr_v2_test_macro_f1_ci95": v2.get("macro_f1_ci95"),
            "note": (
                "farklı test kümeleri (kısmen örtüşür, eğitim bileşimi de farklı); yalnız yan "
                "yana. v1 - v2 farkı sızıntı etkisi olarak sunulmaz; p yok."
            ),
        },
        "preds_file": {"file": p_npz.name, "sha256": sha},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    return report.write_metrics(out, "phase5_v1_leakage", out_dir, when=started), out


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["verify", "predict", "evaluate", "v1-diagnosis"])
    args = ap.parse_args()
    if args.cmd == "verify":
        p, out = verify_models()
        for n, r in out["models"].items():
            print(f"{n}: sha256 {r['sha256_ok']}, in-sample "
                  f"{r.get('train_in_sample_macro_f1', float('nan')):.4f} (kayıt "
                  f"{r.get('train_in_sample_macro_f1_recorded', float('nan')):.4f}), "
                  f"geçti {r['passed']}")  # fmt: skip
        print(p, "GEÇTİ" if out["passed"] else "GEÇMEDİ: test açılmaz")
        raise SystemExit(0 if out["passed"] else 1)
    if args.cmd == "predict":
        print(predict()[0])
    elif args.cmd == "evaluate":
        p, out = evaluate()
        print(p)
        for fam in out["sentences"]["aileler"].values():
            print(*fam, sep="\n")
        s = out["sentences"]
        print(*s["betimsel"], s["seed"], s["sinirliliklar"], s["cv_test"], sep="\n")
    else:
        print(v1_diagnosis()[0])


if __name__ == "__main__":
    main()

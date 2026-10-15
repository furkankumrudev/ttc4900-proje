"""Faz 4: BERTurk fine-tuning. Dondurulmuş konfigürasyon (FT_CONFIG) ve kurallar (RULES).

Bu dosyadaki değerler EĞİTİMDEN ÖNCE ayrı bir commit'le donduruldu ("Faz 4a"). Ayar taraması
yapılmaz; CV ya da test sonucuna göre hiçbir değer değiştirilmez. Önceden tanımlı iki istisna
vardır: RULES["collapse"] (çöken CV koşusu) ve FINAL_RULES["final_collapse"] (çöken final model;
Faz 4a eki, tüm eğitimden önce donduruldu). Eğitim kodu (Trainer, Colab CLI'ı) Faz 4b'de bu
dosyaya eklenir; FT_CONFIG, RULES ve FINAL_RULES ona dokunmaz.

Sezgi: Faz 3'te encoder donmuştu, öğrenilen yalnızca φ(x) üstündeki LR'ydi (konveks problem).
Burada AutoModelForSequenceClassification = encoder + [CLS] pooler + dropout + Linear(768 -> 7);
kayıp L(θ, W, b) = (1/n) Σ_i -log softmax(W φ_θ(x_i) + b)_{y_i} ve TÜM θ (~110M) güncellenir.
Problem konveks değildir, sonuç başlangıca/veri sırasına/dropout'a bağlıdır: çok seed gerekir.

İnsan okuması: docs/faz4_protokol.md. Test yöntemi: ttc4900.test_plan.TEST_PLAN.

Kullanım (Faz 4b; eğitim yalnız GPU'da, --smoke dışında):
    python -m ttc4900.finetune cv --out-dir <metrics> --ckpt-dir <geçici>   # 5 fold x 3 seed
    python -m ttc4900.finetune cv --folds 0 --seeds 42 ...                   # hız ölçümü
    python -m ttc4900.finetune final --out-dir <metrics> --model-dir <modeller>
    python -m ttc4900.finetune summarize                                     # CPU, yerelde
    python -m ttc4900.finetune cv --smoke ; python -m ttc4900.finetune final --smoke
    python -m ttc4900.finetune folds    # YEREL, bir kez: outputs/splits/phase4_folds.json
    python -m ttc4900.finetune verify   # eğitimsiz: train parmak izi + fold dosyası (Colab'da da)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from ttc4900 import report
from ttc4900.cv_compare import latest_metrics, nadeau_bengio, nb_mean_ci
from ttc4900.data import (
    DATASET_REVISION,
    RANDOM_STATE,
    SPLIT_PATH,
    cv_groups,
    load_split,
    make_cv,
)
from ttc4900.embeddings import ENCODERS, row_ids_hash
from ttc4900.paths import METRICS_DIR, MODELS_DIR, SPLITS_DIR
from ttc4900.test_plan import PRIMARY_SEED

NAME = "berturk_ft"
REFERENCE = "char_lr"  # Faz 2 birincil klasik baseline (ham metin), sabit
_BERTURK = ENCODERS["berturk_mean"]  # Faz 3 ile AYNI model ve revision (tek kaynak)

FT_CONFIG = {
    # model
    "model_id": _BERTURK.model_id,  # dbmdz/bert-base-turkish-cased
    "revision": _BERTURK.revision,
    "head": "AutoModelForSequenceClassification(num_labels=7), id2label = split label_names",
    "num_labels": 7,
    # girdi
    "max_length": _BERTURK.max_length,  # 512, Faz 3 ile aynı
    "truncation": "head",  # baştan kesme (tokenizer truncation=True); Faz 3 ile aynı
    "padding": "dinamik (DataCollatorWithPadding)",
    "input_text": "ham metin (char_lr ile aynı girdi)",
    # optimizasyon
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 5,  # üst sınır ve lr schedule'ın uzunluğu
    "early_stopping_patience": 2,  # epoch; val macro-F1 iyileşmezse
    "metric_for_best_model": "macro_f1",
    "greater_is_better": True,
    "load_best_model_at_end": True,  # yalnız CV koşularında (final'de validation yok)
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "optim": "adamw_torch",  # betas (0.9, 0.999), eps 1e-8: transformers varsayılanları
    "dropout": "model varsayılanı (hidden/attention 0.1)",
    "fp16": True,
    # tekrar
    "seeds": (42, 43, 44),
    "primary_seed": PRIMARY_SEED,
    # CV
    "cv_random_state": RANDOM_STATE,  # = char_lr dış CV'sinin 1. tekrarı (repeat 0)
    "cv_n_splits": 5,
    "val_n_splits": 5,  # dış train parçasında make_cv(tr_df, n_splits=5, random_state=42)
    "val_fold": 0,  # ... ilk fold'u validation
    # başarısız koşu
    "collapse_threshold": 0.5,  # best-epoch val macro-F1 bunun altındaysa koşu çöktü
    "collapse_reseed_offset": 100,
}

LITERATURE = {
    "devlin2019": (
        "Devlin vd. 2019 (BERT): fine-tuning için lr ∈ {5e-5, 3e-5, 2e-5}, batch ∈ {16, 32}, "
        "epoch ∈ {2, 3, 4} önerir."
    ),
    "sun2019": (
        "Sun vd. 2019 (How to Fine-Tune BERT for Text Classification?): lr 2e-5, warmup oranı "
        "0.1, max_len 512; uzun metinde head+tail biraz daha iyi (burada tutarlılık için head)."
    ),
    "mosbach2021": (
        "Mosbach vd. 2021 (On the Stability of Fine-tuning BERT): küçük veride kararsızlık "
        "(çöken koşular) yaygın; küçük lr, bias düzeltmeli AdamW, warmup ve daha uzun eğitim "
        "önerir."
    ),
    "dodge2020": (
        "Dodge vd. 2020: aynı konfigürasyonda seed (başlık başlangıcı, veri sırası) sonucu "
        "belirgin değiştirir; birden çok seed raporlanmalı."
    ),
}

RULES = {
    "config_frozen": (
        "FT_CONFIG eğitimden önce donduruldu (Faz 4a commit'i). Ayar taraması yok. Gerekçeler "
        "literatürdendir (LITERATURE): lr 2e-5 (Devlin aralığı; Sun ve Mosbach'ın küçük veri "
        "önerisi), etkin batch 16 (Devlin; 512 token fp16'da T4 belleği için 8 x 2 birikim, "
        "BERT LayerNorm kullandığı için birikim batch 16 ile eşdeğer), en fazla 5 epoch + early "
        "stopping patience 2 (Devlin 2-4 ile Mosbach'ın daha uzun eğitim önerisi arasında "
        "uzlaşma), warmup 0.1 + linear decay (Sun), weight decay 0.01 ve grad norm 1.0 (BERT "
        "standardı), max_length 512 baştan kesme (Faz 3 ile aynı)."
    ),
    "cv_protocol": (
        "Dış fold'lar: make_cv(train, n_splits=5, random_state=42) = char_lr nested CV'sinin 1. "
        "tekrarı (repeat 0). Yalnız bu 5 fold koşulur (5 fold x 3 seed = 15 koşu). Fold eşleşmesi "
        f"kodda {REFERENCE} preds npz'indeki fold_id[0] ile doğrulanır (eşleşmiş karşılaştırma)."
    ),
    "validation": (
        "Early stopping ve load_best_model_at_end için validation, her dış fold'un TRAIN "
        "kısmından make_cv(tr_df, n_splits=5, random_state=42)'nin ilk fold'u (~581 val / ~2323 "
        "fit) olarak ayrılır; train_test_split kullanılmaz. Fit ve val'in grup id'leri (cv_groups) "
        "ayrık olmalı, val dış train'in alt kümesi olmalı, dış val ve test_ids ile kesişmemeli; "
        "bunlar testle doğrulanır. Trainer'ın eval_dataset'i her zaman bu validation parçasıdır."
    ),
    "seeds": (
        "Seed {42, 43, 44}; her CV koşusunda ve final modelde. Fold skoru = 3 seed'in ortalaması; "
        "seed std'si fold std'sinden ayrı raporlanır (seed, başlık başlangıcını, veri sırasını ve "
        "dropout maskelerini birlikte belirler)."
    ),
    "collapse": (
        "Başarısız koşu (Mosbach 2021): best-epoch val macro-F1 < 0.5 ise koşu 'çöktü' sayılır, "
        "seed + 100 ile BİR KEZ yeniden koşulur; iki koşu da raporlanır, yeniden koşu kullanılır. "
        "Bu, FT_CONFIG'in tek önceden tanımlı istisnasıdır; başka hiçbir durumda koşu tekrarlanmaz."
    ),
    "final_model": (
        "Final modeller tüm train (3630 satır) ile, validation olmadan eğitilir. Epoch sayısı "
        "E_final = 15 CV koşusundaki best epoch'ların medyanı (tam sayı). lr schedule CV ile "
        "aynıdır (num_train_epochs=5 üzerinden warmup + linear decay) ve eğitim E_final'ın "
        "sonunda bir callback ile durur; böylece lr eğrisi CV'deki best checkpoint'inkiyle aynı "
        "biçimdedir. Final modeller her seed için eğitilir; git'e girmez."
    ),
    "test_transfer": (
        f"Teste taşınan Faz 4 modeli: seed {PRIMARY_SEED} final modeli (TEST_PLAN "
        "'primary_seed_rule'). CV sonucundan BAĞIMSIZ taşınır: char_lr'yi geçip geçmediğine "
        "bakılmaz. Birincil aile m=1 (berturk_ft - char_lr), exact McNemar + grup bootstrap + "
        "TEST_PLAN karar kuralı. Diğer seed'lerin test macro-F1'i betimseldir (p yok). Test "
        "sonucuna göre konfigürasyon, epoch ya da seed değiştirilmez. Öneri: berturk_ft - bge_m3 "
        "yalnız betimsel CI; son karar TEST_PLAN['membership_note']'a göre test açılmadan önceki "
        "son commit'te FAMILIES'e yazılır."
    ),
    "cv_comparison": (
        f"Keşifsel: d_j = {NAME}_j (seed ortalaması) - {REFERENCE}_j (repeat 0), j = 1..5; "
        "Nadeau-Bengio düzeltmeli t (J=5, df=4), Holm yok (tek karşılaştırma). Karşılaştırılan "
        "prosedürlerdir (char_lr'de iç ayarlama var, BERT'te yok). Eğitim boyutu notu: CV'de BERT "
        "~2323 satırla (validation ayrıldığı için), char_lr 2904 satırla eğitildi; CV "
        "karşılaştırması BERT için muhafazakârdır. Final modellerde ikisi de 3630 satırla "
        "eğitilir."
    ),
    "input_text": (
        "Ham metin. ASİMETRİ: BERTurk tokenizer'ı U+0091-U+0094 kontrol karakterlerini siler, "
        "char_lr görür (docs/shortcut_learning.md). Birincil karşılaştırmada bu asimetri char_lr "
        "lehinedir."
    ),
    "outputs": (
        "Her CV koşusu: phase4_berturk_ft_f{k}_s{seed}_<ts>.json (val eğrisi, best epoch, OOF "
        "macro-F1/accuracy/sınıf F1) ve OOF npz (row_id, y_true, y_pred, logits, fold_id). "
        "Logitler Faz 6'da temperature scaling'i train içinden öğrenmek için saklanır. Özet "
        f"phase4_summary_<ts>.json; ayrıca seed {PRIMARY_SEED} OOF'u phase4_berturk_ft_<ts>.json + "
        "preds npz (y_pred/fold_id şekli 1 x 3630, char_lr repeat 0 fold'ları) olarak yazılır; "
        "birincil güç analizi (paired_test --power) bununla hesaplanır. Her json: split, "
        "git_commit, model revision, torch/transformers sürümü, GPU adı, test: null."
    ),
    "colab": (
        "Kod Faz 4b commit'ine checkout edilerek koşulur. Kurulum Colab'ın CUDA'lı torch'una "
        "dokunmaz (requirements'tan torch satırı atlanır). Kurulumdan sonra "
        "torch.cuda.is_available() kontrol edilir; GPU yoksa eğitim başlamaz (CLI de --smoke "
        "dışında çıkar). Metrikler Drive'a yazılır, koşu kopsa kaldığı yerden devam eder; "
        "checkpoint'ler geçici diske yazılıp silinir; final modeller Drive'da, git dışında."
    ),
    "no_test": (
        "Faz 4 test setine dokunmaz: veri yalnız load_train() ile okunur; load_train_test ve "
        "test metni/etiketi kullanılmaz. test_ids yalnız pytest'teki ayrıklık kontrolünde (id "
        "düzeyinde) okunur. Test değerlendirmesi Faz 5'tedir."
    ),
}

# Faz 4a eki: tüm eğitimden (CV dahil) ÖNCE donduruldu (FT_CONFIG / RULES / LITERATURE'a dokunmaz).
# Eşik ve offset FT_CONFIG'ten okunur; yeni sayı tanımlanmaz.
FINAL_RULES = {
    "final_collapse": (
        "Faz 4a eki, tüm eğitimden (CV dahil) önce donduruldu. RULES['collapse'] CV koşuları "
        "için, bu kural final modeller için geçerlidir. Final modelde validation olmadığı için "
        "RULES['collapse'] uygulanamaz. Yerine: final modelin tüm train (3630 satır) üzerindeki "
        f"in-sample macro-F1'i < {FT_CONFIG['collapse_threshold']} ise model çökmüş sayılır ve "
        "aynı E_final (aynı e_final_sources) ile seed + "
        f"{FT_CONFIG['collapse_reseed_offset']} kullanılarak BİR KEZ yeniden eğitilir. İki model "
        "de raporlanır (ayrı json'lar), yeniden eğitilen kullanılır; o da çökerse yine o "
        "kullanılır ve reseed_also_collapsed işaretlenir. Seed "
        f"{PRIMARY_SEED} slotu çökerse teste taşınan birincil model seed "
        f"{PRIMARY_SEED + FT_CONFIG['collapse_reseed_offset']} modelidir "
        "(TEST_PLAN['primary_seed_rule']); primary_for_test ve primary_model_dir bunu gösterir. "
        "Ölçüt yalnız train'dir; test ya da CV skoru kullanılmaz. Not: in-sample macro-F1 "
        "iyimser olduğundan aynı eşik finalde CV'dekinden daha gevşektir (yalnız bariz çöküşü, "
        "ör. sabit sınıf tahminini yakalar); kural bu yüzden değiştirilmez."
    ),
}


# =========================================================================== Faz 4b: eğitim kodu
# Bu bölüm FT_CONFIG / RULES'u yalnız OKUR. torch / transformers / datasets yalnız eğitim
# fonksiyonlarının içinde import edilir: fold kurulumu, seçim kuralları ve summarize GPU'suz ve
# torch'suz test edilebilir. Veri yalnız data.load_train() ile okunur (RULES["no_test"]).

SPLIT_NAME = Path(SPLIT_PATH).stem
RUN_RE = re.compile(rf"phase4_{NAME}_f(\d+)_s(\d+)_(\d{{8}}-\d{{6}})\.json")
FINAL_RE = re.compile(rf"phase4_{NAME}_final_s(\d+)_(\d{{8}}-\d{{6}})\.json")
DEFAULT_CKPT_DIR = MODELS_DIR / "ckpt"  # outputs/models/ git dışında (.gitignore)
DEFAULT_MODEL_DIR = MODELS_DIR / NAME
SMOKE_DIR = Path(tempfile.gettempdir()) / "ttc4900_faz4_smoke"
SMOKE_SIZES = {"fit": 40, "val": 12, "oof": 12}  # toplam 64 satır
SMOKE_EPOCHS = 1
SMOKE_MODELS = ("tiny", "berturk")  # tiny: BERTurk tokenizer'ı + rastgele 2 katmanlı küçük BERT


# --------------------------------------------------------------------------- fold planı (saf)
@dataclass(frozen=True)
class FoldPlan:
    """Bir dış fold'un satır konumları (train DataFrame'inde iloc).

    outer_train ∪ outer_val = tüm train; fit ∪ val = outer_train (ayrık). Trainer'da
    train_dataset = fit, eval_dataset = val (early stopping + best checkpoint), OOF = outer_val.
    """

    fold: int
    outer_train: np.ndarray
    outer_val: np.ndarray
    fit: np.ndarray
    val: np.ndarray


def make_plan(train: pd.DataFrame, path: Path = SPLIT_PATH) -> list[FoldPlan]:
    """Dış fold'lar ve validation parçaları. Saf: yalnız row_id ve label kullanır.

    Dış: make_cv(train, 5, random_state=42), char_lr nested CV'sinin repeat 0'ı.
    Validation: tr_df = dış train parçası, make_cv(tr_df, 5, random_state=42)'nin val_fold'u.
    StratifiedGroupKFold X'ten yalnız uzunluğu kullanır; satır sırası load_train() ile aynı
    olmalıdır (gruplar ve fold'lar sıraya hizalı).
    """
    c = FT_CONFIG
    train = train.reset_index(drop=True)
    cv, g = make_cv(train, path, n_splits=c["cv_n_splits"], random_state=c["cv_random_state"])
    plans = []
    for k, (tr, va) in enumerate(cv.split(np.zeros(len(train)), train["label"].to_numpy(), g)):
        tr_df = train.iloc[tr].reset_index(drop=True)
        inner, g_in = make_cv(
            tr_df, path, n_splits=c["val_n_splits"], random_state=c["cv_random_state"]
        )
        splits = list(inner.split(np.zeros(len(tr_df)), tr_df["label"].to_numpy(), g_in))
        fit_l, val_l = splits[c["val_fold"]]
        plans.append(FoldPlan(k, tr, va, tr[fit_l], tr[val_l]))
    return plans


def fold_assignment(plans: list[FoldPlan], n: int) -> np.ndarray:
    """Her train satırının dış fold numarası (char_lr npz'indeki fold_id[0] ile aynı biçim)."""
    out = np.full(n, -1, dtype=np.int64)
    for p in plans:
        out[p.outer_val] = p.fold
    return out


def smoke_subset(p: FoldPlan, seed: int = 0) -> FoldPlan:
    """Duman testi: her parçadan küçük alt küme (alt küme ilişkileri korunur)."""
    rng = np.random.default_rng(seed)

    def pick(idx: np.ndarray, n: int) -> np.ndarray:
        return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))

    return FoldPlan(
        p.fold,
        p.outer_train,
        pick(p.outer_val, SMOKE_SIZES["oof"]),
        pick(p.fit, SMOKE_SIZES["fit"]),
        pick(p.val, SMOKE_SIZES["val"]),
    )


# --------------------------------------------------------------------------- referans ve etiketler
def load_reference(ref_dir: Path = METRICS_DIR) -> dict:
    """char_lr'nin (Faz 2) son koşusu: repeat 0 dış fold'ları, y_true, etiket adları, fold skorları.

    npz, json'daki `preds_file` alanından okunur (cv_compare.check_same_folds ile aynı yol).
    """
    m = latest_metrics(REFERENCE, ref_dir, phase=2)
    if m is None:
        raise FileNotFoundError(f"{ref_dir}: phase2_{REFERENCE}_<ts>.json yok")
    with np.load(Path(ref_dir) / m["preds_file"]) as z:  # Windows: tutamaç hemen kapansın
        row_id, y_true, fold0 = z["row_id"].copy(), z["y_true"].copy(), z["fold_id"][0].copy()
    f0 = sorted((f for f in m["cv"]["folds"] if f["repeat"] == 0), key=lambda f: f["fold"])
    return {
        "file": m["_file"],
        "preds_file": m["preds_file"],
        "git_commit": m["git_commit"],
        "split": m["split"],
        "label_names": list(m["label_names"]),
        "row_id": row_id,
        "y_true": y_true,
        "fold_id0": fold0,
        "macro_f1_folds0": [float(f["macro_f1"]) for f in f0],
        "n_train0": [int(f["n_train"]) for f in f0],
        "n_val0": [int(f["n_val"]) for f in f0],
    }


def check_reference(
    train: pd.DataFrame, plans: list[FoldPlan], ref: dict, names: list[str]
) -> None:
    """Eşleştirme varsayımı: aynı split, aynı satırlar aynı sırada, aynı etiket kodlaması ve
    char_lr fold_id[0] ile aynı dış fold'lar. Biri tutmazsa ValueError."""
    if ref["split"] != SPLIT_NAME:
        raise ValueError(f"{REFERENCE} {ref['split']} split'inde, beklenen {SPLIT_NAME}")
    if list(names) != ref["label_names"]:
        raise ValueError(f"Etiket adları/sırası {REFERENCE} ile aynı değil: {names}")
    if not np.array_equal(ref["row_id"], train["row_id"].to_numpy()):
        raise ValueError(f"train satır sırası {REFERENCE} npz'iyle aynı değil (fold'lar kayar).")
    if not np.array_equal(ref["y_true"], train["label"].to_numpy()):
        raise ValueError(f"Etiket kodlaması {REFERENCE} y_true ile aynı değil.")
    if not np.array_equal(ref["fold_id0"], fold_assignment(plans, len(train))):
        raise ValueError(f"Dış fold'lar {REFERENCE} fold_id[0] ile aynı değil.")


# --------------------------------------------------------------------------- fold dosyası
# StratifiedGroupKFold(shuffle=True) aynı girdi ve random_state ile sklearn < 1.8 ve >= 1.8'de
# FARKLI fold'lar üretir (1.7.2 ile 1.8.0 arasında değişti; Colab'da 1.6.1 vardı). Bu yüzden eğitim
# (cv/final/smoke) ve summarize make_cv'yi çalıştırmaz: dış fold'lar char_lr npz'indeki
# fold_id[0]'dan, iç validation parçaları yerelde bir kez make_plan (= make_cv) ile üretilip
# commit'lenen FOLDS_PATH'ten okunur. Fold'ların kendisi değişmez: Faz 4a'da tanımlanan make_cv
# çağrılarının (char_lr ile aynı, sklearn 1.9.1) çıktısıdır. Train'in parmak izi (row_id sırası,
# etiketler, metin) dosyadakiyle aynı değilse eğitim başlamaz.
FOLDS_PATH = SPLITS_DIR / "phase4_folds.json"
FOLD_SOURCE = (
    f"Dış fold'lar: {REFERENCE} preds npz'indeki fold_id[0] (sha256 fold dosyasında). İç "
    "validation: outputs/splits/phase4_folds.json'daki val_row_ids (yerelde make_cv(tr_df, "
    "n_splits=5, random_state=42) fold 0 ile bir kez üretildi). Eğitim ortamında make_cv "
    "çalıştırılmaz (StratifiedGroupKFold fold'ları sklearn sürümüne bağlı)."
)


def texts_sha256(texts) -> str:
    """Metinlerin sırayla sha256'sı (her metin uzunluk önekli UTF-8: birleştirme belirsiz değil)."""
    h = hashlib.sha256()
    for t in texts:
        b = str(t).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def train_fingerprint(train: pd.DataFrame) -> dict:
    """Train satır sırası, etiketler ve (varsa) metinlerin parmak izi."""
    return {
        "n": int(len(train)),
        "row_id_sha256": row_ids_hash(train["row_id"].to_numpy()),
        "label_sha256": row_ids_hash(train["label"].to_numpy()),
        "text_sha256": texts_sha256(train["text"]) if "text" in train else None,
    }


def check_fingerprint(train: pd.DataFrame, folds: dict) -> None:
    """Bu ortamdaki train (load_train) fold dosyasının üretildiği train ile birebir aynı mı?"""
    got, want = train_fingerprint(train), folds["train_fingerprint"]
    if want.get("text_sha256") is None or got["text_sha256"] is None:
        raise ValueError("Parmak izinde metin hash'i yok: train 'text' sütunuyla verilmeli.")
    diff = {k: (got[k], want[k]) for k in want if got.get(k) != want[k]}
    if diff:
        raise ValueError(
            f"train fold dosyasıyla aynı değil (şimdiki, dosya): {diff}. Eğitim başlamaz "
            "(satır sırası, etiket ya da metin farklı; datasets/pandas sürümünü kontrol edin)."
        )


def fold_file_dict(
    train: pd.DataFrame, ref: dict, names: list[str], path: Path = SPLIT_PATH
) -> dict:
    """Yerelde bir kez: make_plan (make_cv) ile iç validation parçalarını üretir, dış fold'ları
    char_lr fold_id[0] ile doğrular ve dosya içeriğini döndürür. train 'text' içermelidir."""
    import platform

    import scipy
    import sklearn

    if "text" not in train:
        raise ValueError("fold dosyası metin parmak iziyle üretilir: train 'text' içermeli.")
    train = train.reset_index(drop=True)
    plans = make_plan(train, path)
    check_reference(train, plans, ref, names)
    ids = train["row_id"].to_numpy()
    return {
        "description": FOLD_SOURCE,
        "split": SPLIT_NAME,
        "data_revision": DATASET_REVISION,
        "method": (
            f"Dış: make_cv(train, n_splits={FT_CONFIG['cv_n_splits']}, random_state="
            f"{FT_CONFIG['cv_random_state']}) = {REFERENCE} fold_id[0] (doğrulandı; dosyada "
            "yalnız sha256). İç: dış train parçası (load_train sırası korunarak) tr_df, "
            f"make_cv(tr_df, n_splits={FT_CONFIG['val_n_splits']}, random_state="
            f"{FT_CONFIG['cv_random_state']}) fold {FT_CONFIG['val_fold']} = val; fit = dış "
            "train - val."
        ),
        "reference": {
            "file": ref["file"],
            "preds_file": ref["preds_file"],
            "fold_id0_sha256": row_ids_hash(ref["fold_id0"]),
        },
        "train_fingerprint": train_fingerprint(train),
        "n_fit": {str(p.fold): len(p.fit) for p in plans},
        "n_val": {str(p.fold): len(p.val) for p in plans},
        "val_row_ids": {str(p.fold): sorted(int(i) for i in ids[p.val]) for p in plans},
        "generated_with": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "created_at": report.now().isoformat(timespec="seconds"),
    }


def write_fold_file(obj: dict, out: Path = FOLDS_PATH) -> Path:
    """Fold dosyası bir kez yazılır (split gibi); var olanın üzerine yazılmaz."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} zaten var; fold dosyası yalnızca bir kez üretilir.")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")  # Windows'ta da LF
    return out


def fold_file_sha256(obj: dict) -> str:
    """İçerik sha256'sı (kanonik JSON; '_' önekli yükleme alanları hariç). Bayt sha'sından farklı
    olarak satır sonlarından (git autocrlf: yerelde CRLF, Colab'da LF) bağımsızdır."""
    body = {k: v for k, v in obj.items() if not k.startswith("_")}
    canon = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_fold_file(path: Path = FOLDS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok: yerelde 'python -m ttc4900.finetune folds' ile üretin."
        )
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj | {"_file": path.name, "_sha256": fold_file_sha256(obj)}


def check_fold_file(folds: dict, ref: dict) -> None:
    """Fold dosyası bu split, veri revision'ı ve char_lr referansı (aynı npz, aynı fold_id[0],
    aynı row_id sırası ve etiketler) için mi üretilmiş? Değilse ValueError."""
    if folds["split"] != SPLIT_NAME or folds["data_revision"] != DATASET_REVISION:
        raise ValueError("Fold dosyası başka bir split ya da veri revision'ı için.")
    r = folds["reference"]
    if r["preds_file"] != ref["preds_file"]:
        raise ValueError(f"Fold dosyası başka bir {REFERENCE} npz'ine göre: {r['preds_file']}")
    if r["fold_id0_sha256"] != row_ids_hash(ref["fold_id0"]):
        raise ValueError(f"{REFERENCE} fold_id[0] fold dosyasındaki sha256 ile aynı değil.")
    fp = folds["train_fingerprint"]
    if fp["n"] != len(ref["row_id"]) or fp["row_id_sha256"] != row_ids_hash(ref["row_id"]):
        raise ValueError(f"Fold dosyasının row_id sırası {REFERENCE} npz'iyle aynı değil.")
    if fp["label_sha256"] != row_ids_hash(ref["y_true"]):
        raise ValueError(f"Fold dosyasının etiketleri {REFERENCE} y_true ile aynı değil.")


def build_plans(
    train: pd.DataFrame,
    fold0,
    val_row_ids: dict,
    path: Path = SPLIT_PATH,
    n_val: dict | None = None,
    n_fit: dict | None = None,
) -> list[FoldPlan]:
    """make_cv çalıştırmadan fold planı: dış fold k = {i: fold0[i] == k}, val = dosyadaki
    val_row_ids[k], fit = dış train - val. Doğrular: val ⊂ dış train; fit/val ve dış train/val
    yakın-tekrar grupları ayrık (cv_groups yalnız split dosyasını okur; split'in train'inde
    olmayan satırda hata); fit ve val her sınıfı içerir; verilirse boyutlar dosyadakiyle aynı.
    make_plan ile aynı FoldPlan'ları verir (testle doğrulanır)."""
    train = train.reset_index(drop=True)
    n_splits = FT_CONFIG["cv_n_splits"]
    fold0 = np.asarray(fold0, dtype=np.int64)
    if len(fold0) != len(train) or set(np.unique(fold0).tolist()) != set(range(n_splits)):
        raise ValueError(f"fold_id[0] train ile hizalı değil ya da 0..{n_splits - 1} dışında.")
    if set(val_row_ids) != {str(k) for k in range(n_splits)}:
        raise ValueError(f"val_row_ids fold anahtarları 0..{n_splits - 1} değil.")
    g = cv_groups(train, path)
    pos = {int(r): i for i, r in enumerate(train["row_id"])}
    y = train["label"].to_numpy()
    n_classes = FT_CONFIG["num_labels"]
    plans = []
    for k in range(n_splits):
        outer_val, outer_train = np.flatnonzero(fold0 == k), np.flatnonzero(fold0 != k)
        vids = [int(r) for r in val_row_ids[str(k)]]
        if len(set(vids)) != len(vids) or not set(vids) <= pos.keys():
            raise ValueError(f"fold {k}: val_row_ids yinelenen ya da train'de olmayan id içeriyor.")
        val = np.sort(np.array([pos[r] for r in vids], dtype=np.int64))
        if not set(val.tolist()) <= set(outer_train.tolist()):
            raise ValueError(f"fold {k}: validation dış train'in alt kümesi değil.")
        fit = np.setdiff1d(outer_train, val)
        if set(g[fit].tolist()) & set(g[val].tolist()):
            raise ValueError(f"fold {k}: fit ve validation yakın-tekrar grupları ayrık değil.")
        if set(g[outer_train].tolist()) & set(g[outer_val].tolist()):
            raise ValueError(f"fold {k}: dış train ve dış val yakın-tekrar grupları ayrık değil.")
        for part, idx, want in (("val", val, n_val), ("fit", fit, n_fit)):
            if want is not None and len(idx) != int(want[str(k)]):
                raise ValueError(f"fold {k}: {part} boyutu {len(idx)}, dosyada {want[str(k)]}.")
            if len(np.unique(y[idx])) != n_classes:
                raise ValueError(f"fold {k}: {part} her sınıfı içermiyor ({n_classes} sınıf).")
        plans.append(FoldPlan(k, outer_train, outer_val, fit, val))
    return plans


def plans_from_fold_file(
    train: pd.DataFrame, ref: dict, folds: dict, path: Path = SPLIT_PATH
) -> list[FoldPlan]:
    """Ortamdan bağımsız fold planı: dış = ref fold_id[0], iç = fold dosyası (make_cv yok)."""
    check_fold_file(folds, ref)
    return build_plans(
        train, ref["fold_id0"], folds["val_row_ids"], path, folds["n_val"], folds["n_fit"]
    )


def setup_folds(
    train: pd.DataFrame, ref: dict, folds: dict, names: list[str], path: Path = SPLIT_PATH
) -> list[FoldPlan]:
    """Eğitimden önce: train parmak izi (row_id sırası, etiket, metin) = fold dosyası; fold
    dosyası = char_lr referansı; plan dosyalardan kurulur. Not: plan fold_id[0]'dan kurulduğu
    için check_reference'ın fold karşılaştırması burada yapı gereği sağlanır; bağımsız değeri
    split/etiket adı/row_id/y_true denetimleridir. Dış fold'lara bağımsız güvence build_plans'in
    dış train/val grup ayrıklığı denetimidir."""
    check_fingerprint(train, folds)
    plans = plans_from_fold_file(train, ref, folds, path)
    check_reference(train, plans, ref, names)
    return plans


def folds_info(folds: dict) -> dict:
    """Koşu json'larına yazılan fold kaynağı."""
    return {
        "file": f"outputs/splits/{folds['_file']}",
        "sha256": folds["_sha256"],
        "source": FOLD_SOURCE,
        "train_fingerprint": folds["train_fingerprint"],
    }


def label_maps(names: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """id2label / label2id: indeks = ClassLabel tamsayısı (data.label_names() sırası)."""
    id2label = dict(enumerate(names))
    return id2label, {n: i for i, n in id2label.items()}


def check_labels(train: pd.DataFrame, names: list[str]) -> None:
    """label tamsayısı ile category adı names sırasıyla tutarlı mı (label2id[category] == label)."""
    if len(names) != FT_CONFIG["num_labels"]:
        raise ValueError(f"{len(names)} sınıf, FT_CONFIG num_labels={FT_CONFIG['num_labels']}")
    _, label2id = label_maps(names)
    if "category" in train:
        enc = train["category"].map(label2id).to_numpy()
        if not np.array_equal(enc, train["label"].to_numpy()):
            raise ValueError("category -> label2id kodlaması train['label'] ile aynı değil.")


# --------------------------------------------------------------------------- metrik, kurallar (saf)
def oof_metrics(y, pred, names: list[str]) -> dict:
    labels = list(range(len(names)))
    pc = f1_score(y, pred, average=None, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": dict(zip(names, pc.tolist(), strict=True)),
    }


def compute_metrics(eval_pred) -> dict:
    """Trainer her değerlendirmede çağırır: logit -> argmax -> macro-F1 (7 sınıf) ve accuracy.
    Dönen anahtarlar Trainer'da `eval_` önekini alır (metric_for_best_model = macro_f1)."""
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.asarray(logits).argmax(-1)
    y = np.asarray(eval_pred.label_ids)
    labels = list(range(FT_CONFIG["num_labels"]))
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float((pred == y).mean()),
    }


def is_collapsed(best_val_macro_f1: float) -> bool:
    """RULES['collapse']: best-epoch val macro-F1 < 0.5."""
    return best_val_macro_f1 < FT_CONFIG["collapse_threshold"]


def select_used(
    runs: dict[tuple[int, int], dict], slots
) -> tuple[dict[tuple[int, int], dict], list[tuple[int, int]], list[tuple[int, int]]]:
    """Her (fold, seed) yuvası için kullanılan koşu (RULES['collapse']).

    Orijinal koşu çöktüyse seed + 100 koşusu kullanılır; o da çökse bile (kural tek yeniden koşu
    der). Döner: (kullanılan koşular, eksik (fold, seed) listesi, çöken yuvalar).
    """
    off = FT_CONFIG["collapse_reseed_offset"]
    used, missing, collapsed = {}, [], []
    for k, s in slots:
        r = runs.get((k, s))
        if r is None:
            missing.append((k, s))
            continue
        if r["collapsed"]:
            collapsed.append((k, s))
            r = runs.get((k, s + off))
            if r is None:
                missing.append((k, s + off))
                continue
        used[(k, s)] = r
    return used, missing, collapsed


def e_final(best_epochs) -> int:
    """E_final = best epoch'ların medyanı (RULES['final_model']).

    15 koşuda (tek sayı) medyan bir gözlemdir, yani tam sayıdır. Çift sayıda (yalnız smoke)
    alt medyan alınır; 15 koşuda ikisi aynıdır.
    """
    xs = [int(e) for e in best_epochs]
    if not xs:
        raise ValueError("best epoch yok")
    if any(not 1 <= e <= FT_CONFIG["num_train_epochs"] for e in xs):
        raise ValueError(f"best epoch 1..{FT_CONFIG['num_train_epochs']} dışında: {xs}")
    return int(statistics.median_low(xs))


def val_curve(log_history: list[dict]) -> list[dict]:
    """Trainer.state.log_history -> epoch başına train_loss ve val macro-F1 / accuracy / loss."""
    rows: dict[int, dict] = {}
    for e in log_history:
        if "epoch" not in e:
            continue
        ep = int(round(e["epoch"]))
        if "eval_macro_f1" in e:
            rows.setdefault(ep, {"epoch": ep}).update(
                val_macro_f1=e["eval_macro_f1"],
                val_accuracy=e.get("eval_accuracy"),
                val_loss=e.get("eval_loss"),
            )
        elif "loss" in e:
            rows.setdefault(ep, {"epoch": ep})["train_loss"] = e["loss"]
    return [rows[k] for k in sorted(rows)]


def best_epoch(curve: list[dict]) -> tuple[int, float]:
    """İlk argmax: Trainer da best checkpoint'i yalnız kesin iyileşmede (>) günceller."""
    ev = [r for r in curve if "val_macro_f1" in r]
    if not ev:
        raise ValueError("val eğrisinde değerlendirme yok")
    b = max(ev, key=lambda r: r["val_macro_f1"])  # max eşitlikte ilk elemanı döndürür
    return int(b["epoch"]), float(b["val_macro_f1"])


# --------------------------------------------------------------------------- koşu kayıtları
def _collect(out_dir: Path, pattern: re.Pattern, glob: str, key, kind: str) -> dict:
    """Desene uyan json'lar: anahtar -> kayıt. Aynı anahtarda birden çok dosya varsa HATA
    (eski/yinelenen koşu sessizce seçilmez; fazlası elle superseded/ altına taşınmalı)."""
    out: dict = {}
    for f in sorted(Path(out_dir).glob(glob)):
        m = pattern.fullmatch(f.name)
        if not m:
            continue
        k = key(m)
        if k in out:
            raise ValueError(
                f"{kind} {k} için birden çok json: {out[k]['_file']} ve {f.name}. Biri "
                "superseded/ altına taşınmadan devam edilmez."
            )
        out[k] = json.loads(f.read_text(encoding="utf-8")) | {"_file": f.name}
    return out


def find_runs(out_dir: Path, smoke: bool = False) -> dict[tuple[int, int], dict]:
    """out_dir'deki CV koşu json'ları: (fold, seed) -> kayıt. Yinelenen anahtar hata verir.
    Smoke ve gerçek koşular karıştırılmaz."""
    runs = _collect(
        out_dir, RUN_RE, f"phase4_{NAME}_f*_s*_*.json", lambda m: (int(m[1]), int(m[2])), "koşu"
    )
    for rec in runs.values():
        if bool(rec.get("smoke")) != smoke:
            raise ValueError(f"{rec['_file']}: smoke={rec.get('smoke')}, bu çağrı smoke={smoke}")
    return runs


def find_finals(out_dir: Path) -> dict[int, dict]:
    """out_dir'deki final json'ları: seed -> kayıt. Aynı seed için birden çok json hata verir."""
    return _collect(
        out_dir, FINAL_RE, f"phase4_{NAME}_final_s*_*.json", lambda m: int(m[1]), "final seed"
    )


def check_config(rec: dict) -> None:
    """Kayıt bu koddaki dondurulmuş FT_CONFIG ve model revision'ıyla mı üretildi? Değilse hata."""
    if rec.get("ft_config") != report.to_jsonable(FT_CONFIG):
        raise ValueError(f"{rec.get('_file')}: ft_config bu koddaki FT_CONFIG ile aynı değil.")
    if rec.get("revision") != FT_CONFIG["revision"]:
        raise ValueError(f"{rec.get('_file')}: model revision {rec.get('revision')} farklı.")


BEST_RELOAD_TOL = 1e-3  # best checkpoint yeniden değerlendirmesi: fp16/CUDA gürültüsü payı


def check_best_checkpoint(rec: dict) -> None:
    """load_best_model_at_end doğrulaması: Trainer'ın best_metric'i val eğrisinin maksimumuna
    eşit ve yüklenen modelin val macro-F1'i ona 1e-3 içinde yakın olmalı; değilse hata."""
    best, tb, rl = (
        rec["best_val_macro_f1"],
        rec.get("trainer_best_metric"),
        rec.get("val_macro_f1_reloaded"),
    )
    if tb is None or abs(tb - best) > 1e-12:
        raise ValueError(f"{rec.get('_file')}: trainer best_metric {tb} != val eğrisi max {best}")
    if rl is None or abs(rl - best) > BEST_RELOAD_TOL:
        raise ValueError(
            f"{rec.get('_file')}: yüklenen modelin val macro-F1'i {rl}, best {best} "
            f"(fark > {BEST_RELOAD_TOL}); best checkpoint yüklenmemiş olabilir."
        )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


# --------------------------------------------------------------------------- Hugging Face parçaları
def require_device(smoke: bool) -> str:
    """GPU kapısı (RULES['colab']): --smoke dışında CUDA yoksa eğitim başlamaz."""
    if smoke:
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU yok: Faz 4 eğitimi yalnız GPU'da koşar (yerelde yalnız --smoke)."
        )
    return "cuda"


def env_info(device: str) -> dict:
    import platform

    import accelerate
    import datasets
    import scipy
    import sklearn
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "sklearn": sklearn.__version__,  # bilgi: fold'lar make_cv'den değil dosyadan okunur
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }


def load_tokenizer():
    """Revision'a sabitlenmiş BERTurk tokenizer'ı (WordPiece, cased)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FT_CONFIG["model_id"], revision=FT_CONFIG["revision"])
    check_tokenizer_revision(tok)
    return tok


def check_tokenizer_revision(tok) -> dict:
    """Yüklenen tokenizer sabit revision'dan mı? Tokenizer modelden farklı olarak commit hash'i
    her sürümde saklamaz (5.x'te yok). Bu yüzden: (1) varsa _commit_hash == revision; (2) o
    revision'ın vocab.txt'si HF önbelleğinde snapshots/<revision>/ altında; (3) tokenizer'ın
    sözlüğü bu dosyayla token -> id düzeyinde birebir aynı. Biri tutmazsa ValueError."""
    import hashlib

    from huggingface_hub import try_to_load_from_cache

    rev = FT_CONFIG["revision"]
    got = getattr(tok, "_commit_hash", None) or tok.init_kwargs.get("_commit_hash")
    if got is not None and got != rev:
        raise ValueError(f"tokenizer revision {got}, beklenen {rev}")
    path = try_to_load_from_cache(FT_CONFIG["model_id"], "vocab.txt", revision=rev)
    if not isinstance(path, str) or Path(path).parent.name != rev:
        raise ValueError(f"vocab.txt @ {rev} HF önbelleğinde bulunamadı: {path}")
    raw = Path(path).read_bytes()
    vocab = {t: i for i, t in enumerate(raw.decode("utf-8").splitlines())}
    if tok.get_vocab() != vocab:
        raise ValueError(f"tokenizer sözlüğü {rev} revision'ının vocab.txt'siyle aynı değil")
    return {
        "commit": rev,
        "vocab_size": len(vocab),
        "vocab_sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_model(names: list[str], smoke_model: str | None, vocab_size: int):
    """encoder + [CLS] pooler + dropout + Linear(768 -> 7). Başlık rastgele başlatılır: çağırmadan
    önce set_seed(seed). smoke_model='tiny': aynı tokenizer, rastgele 2 katmanlı küçük BERT."""
    from transformers import AutoModelForSequenceClassification

    id2label, label2id = label_maps(names)
    if smoke_model == "tiny":
        from transformers import BertConfig

        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=FT_CONFIG["max_length"],
            num_labels=len(names),
            id2label=id2label,
            label2id=label2id,
        )
        return AutoModelForSequenceClassification.from_config(cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        FT_CONFIG["model_id"],
        revision=FT_CONFIG["revision"],
        num_labels=len(names),
        id2label=id2label,
        label2id=label2id,
    )
    got = getattr(model.config, "_commit_hash", None)
    if got is not None and got != FT_CONFIG["revision"]:
        raise ValueError(f"beklenen revision {FT_CONFIG['revision']}, yüklenen {got}")
    return model


def to_dataset(df: pd.DataFrame, tok):
    """DataFrame -> datasets.Dataset; tokenize: baştan kesme, max_length 512, padding YOK
    (DataCollatorWithPadding her batch'i kendi en uzun örneğine doldurur)."""
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True), preserve_index=False)

    def enc(batch):
        return tok(batch["text"], truncation=True, max_length=FT_CONFIG["max_length"])

    return ds.map(enc, batched=True, remove_columns=["text"])


def training_args(output_dir: Path, seed: int, smoke: bool, final: bool = False):
    """TrainingArguments, FT_CONFIG'ten. transformers 4.45+ ve 5.x uyumlu.

    Sürüm farkı: 4.x'te `warmup_ratio` vardır; 5.x'te kaldırıldı ve `warmup_steps` < 1 bir
    float oran olarak yorumlanır. İkisi de ceil(0.1 · toplam adım) warmup verir.
    """
    from transformers import TrainingArguments

    c = FT_CONFIG
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = {
        "output_dir": str(output_dir),
        "seed": seed,
        "learning_rate": c["learning_rate"],
        "per_device_train_batch_size": c["per_device_train_batch_size"],
        "gradient_accumulation_steps": c["gradient_accumulation_steps"],
        "per_device_eval_batch_size": c["per_device_eval_batch_size"],
        "num_train_epochs": SMOKE_EPOCHS if smoke else c["num_train_epochs"],
        "lr_scheduler_type": c["lr_scheduler_type"],
        "weight_decay": c["weight_decay"],
        "max_grad_norm": c["max_grad_norm"],
        "optim": c["optim"],
        "fp16": bool(c["fp16"]) and not smoke,
        "use_cpu": smoke,
        "dataloader_pin_memory": not smoke,
        "logging_strategy": "epoch",
        "report_to": "none",
        "save_only_model": True,  # optimizer durumu yazılmaz (koşu içinden devam yok)
        "save_total_limit": 1,  # load_best_model_at_end ile en iyi + son checkpoint kalır
    }
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = c["warmup_ratio"]
    else:
        kw["warmup_steps"] = c["warmup_ratio"]
    if final:  # validation yok: değerlendirme, checkpoint ve best-model yükleme yok
        kw |= {"eval_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False}
    else:
        kw |= {
            "eval_strategy": c["eval_strategy"],
            "save_strategy": c["save_strategy"],
            "load_best_model_at_end": c["load_best_model_at_end"],
            "metric_for_best_model": c["metric_for_best_model"],
            "greater_is_better": c["greater_is_better"],
        }
    return TrainingArguments(**kw)


def make_trainer(model, args, train_ds, eval_ds, tok, callbacks):
    """Trainer; tokenizer 4.46+'da `processing_class`, 4.45'te `tokenizer` argümanıyla verilir."""
    from transformers import DataCollatorWithPadding, Trainer

    # Gradyan birikimi eşdeğerliği (FT_CONFIG: 8 x 2 = batch 16): transformers 5.x, forward'ı
    # **kwargs alan modelin kaybı num_items_in_batch ile kendisinin ölçeklediğini varsayar ve
    # kaybı birikim adımına BÖLMEZ. BertForSequenceClassification ise düz CrossEntropyLoss()
    # ortalaması döndürür: bölünmezse gradyan 2 mikro-batch ortalamasının TOPLAMI olur ve
    # max_grad_norm=1.0 fiilen 0.5'e iner. False: Trainer kaybı birikim adımına böler (4.x gibi).
    model.accepts_loss_kwargs = False
    kw = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": DataCollatorWithPadding(tok),
        "compute_metrics": compute_metrics if eval_ds is not None else None,
        "callbacks": callbacks,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kw["processing_class"] = tok
    else:
        kw["tokenizer"] = tok
    trainer = Trainer(**kw)
    # Sürüm koruması: bu nitelik varsa (4.46+) False olmalı; değilse kayıp birikim adımına
    # bölünmez ve 8 x 2 birikim batch 16'ya eşdeğer olmaz (yeni bir transformers davranışı).
    if getattr(trainer, "model_accepts_loss_kwargs", False) is not False:
        raise RuntimeError(
            "Trainer.model_accepts_loss_kwargs False değil: gradyan birikimi kaybı birikim "
            "adımına bölmeyebilir (FT_CONFIG etkin batch 16 eşdeğerliği bozulur)."
        )
    return trainer


def stop_at_epoch_callback(n_epochs: int):
    """Final modeller: schedule 5 epoch üzerinden kurulur, eğitim n_epochs sonunda durur."""
    from transformers import TrainerCallback

    class StopAtEpoch(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch is not None and int(round(state.epoch)) >= n_epochs:
                control.should_training_stop = True
            return control

    return StopAtEpoch()


def _schedule(trainer) -> dict:
    max_steps = int(trainer.state.max_steps)
    return {
        "num_train_epochs": trainer.args.num_train_epochs,
        "max_steps": max_steps,
        "warmup_steps": int(trainer.args.get_warmup_steps(max_steps)),
        "global_step": int(trainer.state.global_step),
    }


def _free() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- eğitim
ENV_KEYS = ("transformers", "torch", "gpu")


def env_key(env: dict) -> dict:
    """Koşular arasında aynı olması gereken ortam alanları."""
    return {k: (env or {}).get(k) for k in ENV_KEYS}


def check_env(recs, env: dict) -> None:
    """Kayıtların env'i (transformers, torch, gpu adı) verilen ortamla aynı mı? Değilse hata
    (dosya adı ve değerlerle). Yarım kalmış bir CV başka sürümle/GPU'yla sürdürülmez."""
    want = env_key(env)
    for r in recs:
        got = env_key(r.get("env"))
        if got != want:
            diff = {k: (got[k], want[k]) for k in ENV_KEYS if got[k] != want[k]}
            raise ValueError(f"{r.get('_file')}: ortam farklı (kayıt, şimdiki): {diff}")


def phase4_records(out_dir: Path) -> list[dict]:
    """out_dir kökündeki koşu ve final json'ları (env taşıyan kayıtlar)."""
    recs = []
    for pat in (f"phase4_{NAME}_f*_s*_*.json", f"phase4_{NAME}_final_s*_*.json"):
        for f in sorted(Path(out_dir).glob(pat)):
            if RUN_RE.fullmatch(f.name) or FINAL_RE.fullmatch(f.name):
                recs.append(json.loads(f.read_text(encoding="utf-8")) | {"_file": f.name})
    return recs


def load_checked_train(ref_dir: Path, folds_file: Path) -> dict:
    """Yalnız-train veri + etiket doğrulaması + fold planı (make_cv ÇALIŞTIRILMAZ): dış fold'lar
    char_lr fold_id[0], iç validation fold dosyası; train parmak izi dosyayla aynı olmalı."""
    from ttc4900.data import label_names, load_train

    train = load_train()  # test satırları belleğe hiç alınmaz
    names = label_names()
    if names != load_split()["label_names"]:
        raise ValueError("data.label_names() split dosyasındaki label_names ile aynı değil.")
    check_labels(train, names)
    ref = load_reference(ref_dir)
    folds = load_fold_file(folds_file)
    plans = setup_folds(train, ref, folds, names)
    return {"train": train, "names": names, "ref": ref, "folds": folds, "plans": plans}


def _context(args) -> dict:
    """Ortak kurulum: GPU kapısı, ortam tekliği, yalnız-train veri, etiket ve fold doğrulamaları,
    tokenizer."""
    smoke = args.smoke
    device = require_device(smoke)
    out_dir, ckpt_dir, model_dir = resolve_dirs(args)
    env = env_info(device)
    check_env(phase4_records(out_dir), env)  # eğitimden önce: farklı ortamda devam yok
    d = load_checked_train(args.ref_dir, args.folds_file)
    train, names, ref, folds, plans = (d[k] for k in ("train", "names", "ref", "folds", "plans"))
    tok = load_tokenizer()  # revision denetimi içeride (check_tokenizer_revision)
    print(
        f"[kurulum] {len(train)} train satırı; parmak izi {folds['_file']} ile, dış fold'lar ve "
        f"etiketler {ref['file']} ile doğrulandı (make_cv çalıştırılmadı); tokenizer @ "
        f"{FT_CONFIG['revision'][:8]} doğrulandı; cihaz {device}",
        flush=True,
    )
    return {
        "smoke": smoke,
        "smoke_model": args.smoke_model if smoke else None,
        "device": device,
        "env": env,
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "model_dir": model_dir,
        "train": train,
        "names": names,
        "plans": plans,
        "ref": ref,
        "folds": folds_info(folds),
        "tok": tok,
        "tokenizer_check": check_tokenizer_revision(tok),
    }


def _base_record(ctx: dict, started) -> dict:
    return {
        "phase": 4,
        "timestamp": started.isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": SPLIT_NAME,
        "data_revision": DATASET_REVISION,
        "model_id": FT_CONFIG["model_id"]
        if ctx["smoke_model"] != "tiny"
        else "tiny (smoke: rastgele 2 katmanlı BERT, BERTurk tokenizer'ı)",
        "revision": FT_CONFIG["revision"],
        "smoke": ctx["smoke"],
        "smoke_model": ctx["smoke_model"],
        "label_names": ctx["names"],
        "env": ctx["env"],
        "tokenizer_check": ctx["tokenizer_check"],
        "folds": ctx.get("folds"),
        "ft_config": FT_CONFIG,
    }


FAILED_ATTEMPTS = "phase4_failed_attempts.jsonl"


def record_failed_attempt(out_dir: Path, info: dict, error: BaseException, state: dict) -> Path:
    """Başarısız bir eğitim denemesini out_dir/phase4_failed_attempts.jsonl'e bir satır olarak
    ekler (fold, seed, zaman, git_commit, hata, varsa val eğrisi). İz kaybolmasın diye; koşu
    sayılmaz ve hiçbir seçimde kullanılmaz."""
    curve = None
    tr = state.get("trainer")
    if tr is not None:
        try:
            curve = val_curve(tr.state.log_history)
        except Exception:  # noqa: BLE001 - iz kaydı hatayı maskelememeli
            curve = None
    row = info | {
        "time": report.now().isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "error": f"{type(error).__name__}: {error}",
        "val_curve": curve,
    }
    path = Path(out_dir) / FAILED_ATTEMPTS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report.to_jsonable(row), ensure_ascii=False) + "\n")
    return path


def read_failed_attempts(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / FAILED_ATTEMPTS
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def train_cv_run(ctx: dict, k: int, seed: int, reseed_of: int | None = None) -> dict:
    """_train_cv_run + başarısız denemenin izi (istisna kaydedilip yeniden fırlatılır)."""
    state: dict = {}
    try:
        return _train_cv_run(ctx, k, seed, reseed_of, state)
    except Exception as e:
        info = {"kind": "cv", "fold": k, "seed": seed, "reseed_of": reseed_of,
                "smoke": ctx["smoke"]}  # fmt: skip
        record_failed_attempt(ctx["out_dir"], info, e, state)
        raise


def _train_cv_run(ctx: dict, k: int, seed: int, reseed_of: int | None, state: dict) -> dict:
    """Bir CV koşusu: fit'te eğit, val'de early stopping, outer_val'de OOF tahmini.

    Çıktı: OOF npz (row_id, y_true, y_pred, logits, fold_id) ve en son json (json'un varlığı =
    koşu tamamlandı; yeniden başlatmada atlanır). Checkpoint'ler koşu bitince silinir.
    """
    from transformers import EarlyStoppingCallback, set_seed

    c, train, names, tok = FT_CONFIG, ctx["train"], ctx["names"], ctx["tok"]
    p = smoke_subset(ctx["plans"][k]) if ctx["smoke"] else ctx["plans"][k]
    started, t0 = report.now(), time.perf_counter()
    ds_fit, ds_val, ds_oof = (to_dataset(train.iloc[i], tok) for i in (p.fit, p.val, p.outer_val))
    set_seed(seed)  # model Trainer'dan önce kurulur: başlığın başlangıcı da bu seed'e bağlı
    model = load_model(names, ctx["smoke_model"], len(tok))
    run_dir = Path(ctx["ckpt_dir"]) / f"f{k}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    es = EarlyStoppingCallback(early_stopping_patience=c["early_stopping_patience"])
    args = training_args(run_dir, seed, ctx["smoke"])
    # eval_dataset = validation parçası (dış train içinden); OOF parçası Trainer'a verilmez
    trainer = make_trainer(model, args, ds_fit, ds_val, tok, [es])
    state["trainer"] = trainer  # hata olursa val eğrisi iz kaydına girer
    out = trainer.train()
    curve = val_curve(trainer.state.log_history)
    best_ep, best_f1 = best_epoch(curve)
    schedule = _schedule(trainer)
    trainer_best = trainer.state.best_metric
    reloaded = float(trainer.evaluate()["eval_macro_f1"])  # best checkpoint yüklendi mi
    name = f"phase4_{NAME}_f{k}_s{seed}"
    check_best_checkpoint(  # tutmazsa hiçbir dosya yazılmadan durur
        {
            "_file": name,
            "best_val_macro_f1": best_f1,
            "trainer_best_metric": trainer_best,
            "val_macro_f1_reloaded": reloaded,
        }
    )
    pred = trainer.predict(ds_oof)  # yalnız tahmin: eğitim ve seçim bitti
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    logits = np.asarray(logits, dtype=np.float32)
    y_pred = logits.argmax(-1).astype(np.int64)
    y_true = train["label"].to_numpy()[p.outer_val]
    row_id = train["row_id"].to_numpy()
    preds_file = f"preds/cv_{name}_{started:%Y%m%d-%H%M%S}.npz"
    out_dir = Path(ctx["out_dir"])
    _save_npz(
        out_dir / preds_file,
        row_id=row_id[p.outer_val],
        y_true=y_true,
        y_pred=y_pred,
        logits=logits,
        fold_id=np.full(len(p.outer_val), k, dtype=np.int64),
    )
    epochs_run = max(r["epoch"] for r in curve)
    obj = _base_record(ctx, started) | {
        "experiment": f"phase4_{NAME}_cv_run",
        "fold": k,
        "seed": seed,
        "slot_seed": seed if reseed_of is None else reseed_of,
        "reseed_of": reseed_of,
        "n_fit": len(p.fit),
        "n_val": len(p.val),
        "n_outer_train": len(p.outer_train),
        "n_oof": len(p.outer_val),
        "val_row_ids": sorted(int(i) for i in row_id[p.val]),
        "fit_row_ids_sha256": row_ids_hash(np.sort(row_id[p.fit])),
        "eval_dataset": (
            "validation = dış train içinden make_cv(tr_df, n_splits=5, random_state=42) fold 0 "
            "(yerelde üretilip outputs/splits/phase4_folds.json'dan okundu); dış val (OOF) ve "
            "test eğitimde/seçimde kullanılmadı"
        ),
        "val_curve": curve,
        "best_epoch": best_ep,
        "best_val_macro_f1": best_f1,
        "trainer_best_metric": trainer_best,
        "val_macro_f1_reloaded": reloaded,
        "epochs_run": epochs_run,
        "stopped_early": epochs_run < schedule["num_train_epochs"],
        "collapsed": is_collapsed(best_f1),
        "collapse_rule": RULES["collapse"],
        "train_loss_mean": float(out.training_loss),
        "schedule": schedule,
        "oof": oof_metrics(y_true, y_pred, names),
        "preds_file": preds_file,
        "fold_reference": {
            "file": ctx["ref"]["file"],
            "preds_file": ctx["ref"]["preds_file"],
            "check": f"dış fold'lar = {REFERENCE} fold_id[0], etiketler = y_true (doğrulandı)",
        },
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "test": None,
    }
    path = report.write_metrics(obj, name, out_dir, when=started)  # en son: varlığı = tamam
    del trainer, model, pred
    _free()
    shutil.rmtree(run_dir, ignore_errors=True)
    return obj | {"_file": path.name}


def _print_run(r: dict) -> None:
    flag = "  ÇÖKTÜ" if r["collapsed"] else ""
    print(
        f"  {r['_file']}: best epoch {r['best_epoch']} val F1 {r['best_val_macro_f1']:.4f} "
        f"OOF F1 {r['oof']['macro_f1']:.4f} ({r['runtime_seconds']:.0f} sn){flag}",
        flush=True,
    )


def cmd_cv(args) -> None:
    ctx = _context(args)
    off = FT_CONFIG["collapse_reseed_offset"]
    for k in args.folds:
        for s in args.seeds:
            runs = find_runs(ctx["out_dir"], ctx["smoke"])
            rec = runs.get((k, s))
            if rec is None:
                print(f"[cv] fold {k} seed {s}: eğitiliyor", flush=True)
                rec = train_cv_run(ctx, k, s)
                _print_run(rec)
            else:
                check_config(rec)  # eski konfigürasyonla yazılmış koşu atlanmaz: hata
                print(f"[cv] fold {k} seed {s}: {rec['_file']} var, atlandı", flush=True)
            if not rec["collapsed"]:
                continue
            if (k, s + off) in runs:
                check_config(runs[(k, s + off)])
                print(f"[cv] fold {k} seed {s + off} (yeniden koşu): var, atlandı", flush=True)
                continue
            print(
                f"[cv] fold {k} seed {s} çöktü (best val F1 < {FT_CONFIG['collapse_threshold']});"
                f" seed {s + off} ile bir kez yeniden koşuluyor",
                flush=True,
            )
            _print_run(train_cv_run(ctx, k, s + off, reseed_of=s))


def final_meta(
    seed: int, reseed_of: int | None, in_sample_macro_f1: float, model_dir: Path
) -> dict:
    """FINAL_RULES['final_collapse'] alanları (saf): slot, çöküş ve birincil model.

    Orijinal eğitim (reseed_of None): slot = seed; çöktüyse slot'un kullanılan modeli seed+offset
    olacaktır. Yeniden eğitim: slot = reseed_of; çöküp çökmemesinden bağımsız kullanılan odur.
    primary_for_test yalnız slot PRIMARY_SEED'in KULLANILAN modelinde True'dur.
    """
    off = FT_CONFIG["collapse_reseed_offset"]
    collapsed = is_collapsed(in_sample_macro_f1)
    slot = seed if reseed_of is None else reseed_of
    if reseed_of is not None and seed != reseed_of + off:
        raise ValueError(f"yeniden eğitim seed'i {seed}, beklenen {reseed_of + off}")
    used_seed = slot + off if (reseed_of is None and collapsed) else seed
    is_used = used_seed == seed
    return {
        "slot_seed": slot,
        "reseed_of": reseed_of,
        "collapsed": collapsed,
        "reseed_seed": slot + off if (reseed_of is None and collapsed) else None,
        "reseed_also_collapsed": collapsed if reseed_of is not None else None,
        "used_for_slot": is_used,
        "primary_for_test": slot == PRIMARY_SEED and is_used,
        "primary_model_dir": str(Path(model_dir) / str(used_seed))
        if slot == PRIMARY_SEED
        else None,
    }


def train_final(ctx: dict, seed: int, e_fin: int, used: dict, reseed_of: int | None = None) -> dict:
    """_train_final + başarısız denemenin izi (istisna kaydedilip yeniden fırlatılır)."""
    state: dict = {}
    try:
        return _train_final(ctx, seed, e_fin, used, reseed_of, state)
    except Exception as e:
        info = {"kind": "final", "fold": None, "seed": seed, "reseed_of": reseed_of,
                "e_final": e_fin, "smoke": ctx["smoke"]}  # fmt: skip
        record_failed_attempt(ctx["out_dir"], info, e, state)
        raise


def weights_sha256(model_dir: Path) -> dict[str, str]:
    """Kaydedilen ağırlık dosyalarının sha256'sı (model kimliği): dosya adı -> özet."""
    import hashlib

    files = sorted(Path(model_dir).glob("*.safetensors")) or sorted(Path(model_dir).glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"{model_dir}: ağırlık dosyası (*.safetensors / *.bin) yok")
    out = {}
    for f in files:
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[f.name] = h.hexdigest()
    return out


def check_model_dir_free(mdir: Path) -> None:
    """Final model dizini var ve boş değilse HATA: kaydedilmiş bir model üzerine yazılmaz."""
    mdir = Path(mdir)
    if mdir.exists() and any(mdir.iterdir()):
        raise FileExistsError(
            f"{mdir} zaten var ve boş değil; final model üzerine yazılmaz. Json'u olmayan bir "
            "model ise (yarım kalmış koşu) elle taşıyın/silin."
        )


def _train_final(
    ctx: dict, seed: int, e_fin: int, used: dict, reseed_of: int | None, state: dict
) -> dict:
    """Final model: tüm train, validation yok; schedule 5 epoch üzerinden, E_final'da durur.
    reseed_of: FINAL_RULES['final_collapse'] yeniden eğitimiyse slot seed'i (seed = slot+100)."""
    from transformers import set_seed

    mdir = Path(ctx["model_dir"]) / str(seed)
    check_model_dir_free(mdir)  # eğitimden önce: üzerine yazma yok
    train, tok = ctx["train"], ctx["tok"]
    idx = np.arange(len(train))
    if ctx["smoke"]:
        n = sum(SMOKE_SIZES.values())
        idx = np.sort(np.random.default_rng(0).choice(idx, size=n, replace=False))
    started, t0 = report.now(), time.perf_counter()
    ds = to_dataset(train.iloc[idx], tok)
    set_seed(seed)
    model = load_model(ctx["names"], ctx["smoke_model"], len(tok))
    run_dir = Path(ctx["ckpt_dir"]) / f"final_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    args = training_args(run_dir, seed, ctx["smoke"], final=True)
    trainer = make_trainer(model, args, ds, None, tok, [stop_at_epoch_callback(e_fin)])
    state["trainer"] = trainer
    out = trainer.train()
    curve = val_curve(trainer.state.log_history)
    epochs_run = max(r["epoch"] for r in curve)
    if epochs_run != e_fin:
        raise RuntimeError(f"final eğitim {epochs_run} epoch koştu, E_final={e_fin}")
    check_model_dir_free(mdir)
    mdir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    sha = weights_sha256(mdir)  # model kimliği: teste taşınan dosya bu özetle eşleşmeli
    # FINAL_RULES['final_collapse'] ölçütü: eğitildiği train satırlarında in-sample macro-F1
    pred = trainer.predict(ds)
    lg = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    ins = oof_metrics(train["label"].to_numpy()[idx], np.asarray(lg).argmax(-1), ctx["names"])
    obj = _base_record(ctx, started) | {
        "experiment": f"phase4_{NAME}_final",
        "seed": seed,
        "n_train": len(idx),
        "validation": None,
        "e_final": e_fin,
        "e_final_rule": RULES["final_model"],
        "e_final_sources": e_final_sources(used),
        "epochs_run": epochs_run,
        "train_loss_curve": curve,
        "train_loss_mean": float(out.training_loss),
        "train_loss_last_epoch": curve[-1].get("train_loss"),
        "train_in_sample_macro_f1": ins["macro_f1"],
        "train_in_sample_accuracy": ins["accuracy"],
        "train_in_sample_note": (
            "Final model eğitildiği train satırlarında (eval modu, dropout kapalı) "
            "değerlendirildi; genelleme ölçüsü değildir. Çöküş ölçütü olarak kullanılır: kural "
            "FINAL_RULES['final_collapse']."
        ),
        "final_collapse_rule": FINAL_RULES["final_collapse"],
        **final_meta(seed, reseed_of, ins["macro_f1"], ctx["model_dir"]),
        "schedule": _schedule(trainer),
        "model_dir": str(mdir),
        "weights_sha256": sha,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "test": None,
    }
    path = report.write_metrics(obj, f"phase4_{NAME}_final_s{seed}", ctx["out_dir"], when=started)
    del trainer, model, pred
    _free()
    shutil.rmtree(run_dir, ignore_errors=True)
    return obj | {"_file": path.name}


def e_final_sources(used: dict) -> list[dict]:
    """E_final'ın kaynağı: her (fold, seed) yuvasında kullanılan koşu (çöküş sonrası)."""
    return [
        {"fold": k, "slot_seed": s, "seed": r["seed"], "file": r["_file"],
         "best_epoch": r["best_epoch"]}
        for (k, s), r in sorted(used.items())
    ]  # fmt: skip


def check_final(rec: dict, e_fin: int, used: dict, smoke: bool, model_dir=None) -> None:
    """Var olan final json'u bu CV koşularından, bu konfigürasyondan ve FINAL_RULES'a tutarlı
    alanlarla mı yazılmış? Değilse hata. model_dir verilirse primary_model_dir de denetlenir."""
    check_config(rec)
    f = rec.get("_file")
    if rec.get("final_collapse_rule") != FINAL_RULES["final_collapse"]:
        raise ValueError(f"{f}: final_collapse_rule bu koddaki FINAL_RULES metniyle aynı değil")
    if bool(rec.get("smoke")) != smoke:
        raise ValueError(f"{f}: smoke={rec.get('smoke')}, bu çağrı smoke={smoke}")
    if rec.get("e_final") != e_fin:
        raise ValueError(f"{f}: e_final {rec.get('e_final')}, CV'den hesaplanan {e_fin}")
    if rec.get("e_final_sources") != e_final_sources(used):
        raise ValueError(f"{f}: e_final_sources kullanılan CV koşularıyla aynı değil")
    want = final_meta(
        rec["seed"], rec.get("reseed_of"), rec["train_in_sample_macro_f1"], model_dir or "."
    )
    pmd, wmd = rec.get("primary_model_dir"), want.pop("primary_model_dir")
    if model_dir is None:  # summarize: kayıt başka işletim sisteminde yazılmış olabilir
        pmd = pmd and re.split(r"[\\/]", pmd)[-1]
        wmd = wmd and Path(wmd).name
    bad = {k: (rec.get(k), v) for k, v in want.items() if rec.get(k) != v}
    if pmd != wmd:
        bad["primary_model_dir"] = (rec.get("primary_model_dir"), wmd)
    if bad:
        raise ValueError(f"{f}: FINAL_RULES alanları tutarsız (kayıt, beklenen): {bad}")


def select_finals(finals: dict[int, dict], slots) -> tuple[dict[int, dict], list[int], list[int]]:
    """Slot (FT_CONFIG seed'i) başına kullanılan final model (FINAL_RULES['final_collapse']).
    Döner: (kullanılan, eksik seed'ler, çöken slotlar). select_used ile aynı mantık."""
    used, missing, collapsed = select_used(
        {(0, s): r for s, r in finals.items()}, [(0, s) for s in slots]
    )
    return {s: r for (_, s), r in used.items()}, [s for _, s in missing], [s for _, s in collapsed]


def _print_final(r: dict) -> None:
    flag = "  ÇÖKTÜ (FINAL_RULES)" if r["collapsed"] else ""
    tag = f" (slot {r['slot_seed']} yeniden eğitimi)" if r.get("reseed_of") is not None else ""
    print(
        f"  {r['_file']}{tag}: {r['epochs_run']} epoch, son epoch eğitim kaybı "
        f"{r['train_loss_last_epoch']}, train in-sample macro-F1 "
        f"{r['train_in_sample_macro_f1']:.4f}, model {r['model_dir']}, birincil "
        f"{r['primary_for_test']} ({r['runtime_seconds']:.0f} sn){flag}",
        flush=True,
    )


def cmd_final(args) -> None:
    ctx = _context(args)
    runs = find_runs(ctx["out_dir"], ctx["smoke"])
    if ctx["smoke"]:
        slots = sorted(k for k, r in runs.items() if r.get("reseed_of") is None)
    else:
        slots = [(k, s) for k in range(FT_CONFIG["cv_n_splits"]) for s in FT_CONFIG["seeds"]]
    used, missing, _ = select_used(runs, slots)
    if missing or not used:
        raise SystemExit(f"E_final için CV koşuları eksik (fold, seed): {missing or 'hiç yok'}")
    for r in used.values():
        check_config(r)
        check_best_checkpoint(r)
    e_fin = e_final([r["best_epoch"] for r in used.values()])
    print(f"[final] E_final = {e_fin} ({len(used)} koşunun best epoch medyanı)", flush=True)
    off = FT_CONFIG["collapse_reseed_offset"]
    finals = find_finals(ctx["out_dir"])  # yinelenen seed json'u hata verir
    md = ctx["model_dir"]
    for s in args.seeds:
        rec = finals.get(s)
        if rec is None:
            print(f"[final] seed {s}: eğitiliyor", flush=True)
            rec = train_final(ctx, s, e_fin, used)
            _print_final(rec)
        else:
            check_final(rec, e_fin, used, ctx["smoke"], md)
            print(f"[final] seed {s}: {rec['_file']} var, atlandı", flush=True)
        if not rec["collapsed"]:
            continue
        if s + off in finals:  # yeniden eğitim yalnız bir kez
            check_final(finals[s + off], e_fin, used, ctx["smoke"], md)
            print(f"[final] seed {s + off} (yeniden eğitim): var, atlandı", flush=True)
            continue
        print(
            f"[final] seed {s} çöktü (train in-sample macro-F1 < "
            f"{FT_CONFIG['collapse_threshold']}); FINAL_RULES['final_collapse']: aynı E_final ile "
            f"seed {s + off} bir kez yeniden eğitiliyor",
            flush=True,
        )
        _print_final(train_final(ctx, s + off, e_fin, used, reseed_of=s))


# --------------------------------------------------------------------------- özet (CPU)
def sigma_ci_factors(df: int, level: float = 0.95) -> list[float]:
    """σ için %level aralığın s'ye oranı (normal varsayımı): (df·s²/σ²) ~ χ²_df ⇒
    [sqrt(df / χ²_{1-a/2, df}), sqrt(df / χ²_{a/2, df})]."""
    a = 1 - level
    return [
        float(np.sqrt(df / stats.chi2.ppf(1 - a / 2, df))),
        float(np.sqrt(df / stats.chi2.ppf(a / 2, df))),
    ]


def c4_factor(n: int) -> float:
    """E[s] = c4(n)·σ (normal örneklem, ddof=1): c4 = sqrt(2/(n-1)) Γ(n/2) / Γ((n-1)/2)."""
    import math

    return math.sqrt(2 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))


def check_run_symmetry(runs: dict[tuple[int, int], dict]) -> None:
    """CV koşu kümesinin RULES['collapse'] ile tutarlılığı: FT_CONFIG seed'li koşular yeniden
    koşu değildir; seed+offset koşusu yalnız orijinali çökmüş bir slot için ve reseed_of = slot
    ile vardır. Json içindeki fold/seed dosya adıyla aynı olmalı. Değilse hata."""
    seeds, off = set(FT_CONFIG["seeds"]), FT_CONFIG["collapse_reseed_offset"]
    for (k, sd), r in sorted(runs.items()):
        f = r.get("_file")
        if r.get("fold") != k or r.get("seed") != sd:
            raise ValueError(
                f"{f}: json fold/seed ({r.get('fold')}, {r.get('seed')}) adla uyuşmuyor"
            )
        if sd in seeds:
            if r.get("reseed_of") is not None:
                raise ValueError(f"{f}: FT_CONFIG seed'li koşuda reseed_of={r.get('reseed_of')}")
            continue
        base = sd - off
        if base not in seeds or r.get("reseed_of") != base:
            raise ValueError(
                f"{f}: seed {sd} / reseed_of {r.get('reseed_of')} bir slotla uyuşmuyor"
            )
        orig = runs.get((k, base))
        if orig is None or not orig["collapsed"]:
            raise ValueError(
                f"{f}: yeniden koşu var ama ({k}, {base}) orijinali yok ya da çökmemiş"
            )


def _check_run_rows(r: dict, plan: FoldPlan, ref: dict, pos: dict, out_dir: Path):
    """Bir koşunun OOF npz'i ve fit/val parçaları planla ve referansla tutarlı mı? (z, idx)."""
    k = r["fold"]
    if r["split"] != ref["split"] or r["label_names"] != ref["label_names"]:
        raise ValueError(f"{r['_file']}: split/etiketler {REFERENCE} ile aynı değil")
    z = _load_npz(Path(out_dir) / r["preds_file"])
    idx = np.array([pos[int(i)] for i in z["row_id"]], dtype=np.int64)
    if not np.array_equal(np.sort(idx), np.flatnonzero(ref["fold_id0"] == k)):
        raise ValueError(f"{r['_file']}: OOF satırları {REFERENCE} fold {k} ile aynı değil")
    if not np.array_equal(ref["y_true"][idx], z["y_true"]):
        raise ValueError(f"{r['_file']}: y_true {REFERENCE} ile aynı değil")
    if r["val_row_ids"] != sorted(int(i) for i in ref["row_id"][plan.val]):
        raise ValueError(f"{r['_file']}: val_row_ids fold {k} planıyla aynı değil")
    if r["fit_row_ids_sha256"] != row_ids_hash(np.sort(ref["row_id"][plan.fit])):
        raise ValueError(f"{r['_file']}: fit satırları (sha256) fold {k} planıyla aynı değil")
    if set(r["val_row_ids"]) & {int(i) for i in z["row_id"]}:
        raise ValueError(f"{r['_file']}: validation ile OOF satırları kesişiyor")
    return z, idx


def superseded_files(out_dir: Path) -> list[str]:
    """out_dir alt dizinlerindeki (ör. superseded/) phase4_* dosyaları: özette listelenir, hata
    değildir ve hiçbir hesaba girmez."""
    root = Path(out_dir)
    return sorted(
        f.relative_to(root).as_posix() for f in root.rglob("phase4_*") if f.parent != root
    )


def summarize(
    out_dir: Path = METRICS_DIR,
    ref_dir: Path = METRICS_DIR,
    when=None,
    split_path: Path = SPLIT_PATH,
    folds_path: Path = FOLDS_PATH,
):
    """15 koşuyu toplar. Yazar: phase4_summary_<ts>.json ve seed 42 OOF'u
    (phase4_berturk_ft_<ts>.json + preds npz; şekil 1 x n, char_lr repeat 0 sırası ve fold'ları).
    Ağ ve GPU gerekmez. Döner: (özet yolu, OOF json yolu)."""
    c = FT_CONFIG
    ref = load_reference(ref_dir)
    names = ref["label_names"]
    folds, seeds = list(range(c["cv_n_splits"])), list(c["seeds"])
    runs = find_runs(out_dir, smoke=False)  # yinelenen (fold, seed) json'u hata verir
    used, missing, collapsed = select_used(runs, [(k, s) for k in folds for s in seeds])
    if missing:
        raise SystemExit(f"Eksik CV koşuları (fold, seed): {missing}")
    check_run_symmetry(runs)
    for r in runs.values():  # kullanılmayan (çöken) koşular da aynı konfigürasyondan olmalı
        check_config(r)
        check_best_checkpoint(r)
    finals = find_finals(out_dir)  # final json'ları varsa CV ile çapraz denetlenir
    env_recs = [*runs.values(), *finals.values()]  # ortam tekliği: CV + final, gpu adı dahil
    check_env(env_recs, env_recs[0]["env"])
    # fit/val parçaları: aynı fold planı eğitimdeki gibi dosyalardan (make_cv'siz) yeniden kurulur
    # ve TÜM koşu json'larıyla (çöken orijinaller dahil) karşılaştırılır
    plan_train = pd.DataFrame({"row_id": ref["row_id"], "label": ref["y_true"]})
    fold_file = load_fold_file(folds_path)
    plans = plans_from_fold_file(plan_train, ref, fold_file, split_path)
    check_reference(plan_train, plans, ref, names)
    for r in runs.values():  # her koşu bu fold dosyasıyla eğitilmiş olmalı
        if (r.get("folds") or {}).get("sha256") != fold_file["_sha256"]:
            raise ValueError(
                f"{r['_file']}: fold dosyası sha256'sı {fold_file['_file']} ile aynı değil"
            )
    pos = {int(r): i for i, r in enumerate(ref["row_id"])}
    rows_of = {key: _check_run_rows(r, plans[key[0]], ref, pos, out_dir) for key, r in runs.items()}
    used_keys = {id(r): key for key, r in runs.items()}
    n = len(ref["row_id"])
    f1 = np.zeros((len(folds), len(seeds)))
    acc = np.zeros_like(f1)
    per_class = []
    y_pred0 = np.full((1, n), -1, dtype=np.int64)
    fold0 = np.full((1, n), -1, dtype=np.int64)
    logits0 = np.full((n, len(names)), np.nan, dtype=np.float32)
    run_rows = []
    for (k, s), r in sorted(used.items()):
        z, idx = rows_of[used_keys[id(r)]]
        m = oof_metrics(z["y_true"], z["y_pred"], names)
        if abs(m["macro_f1"] - r["oof"]["macro_f1"]) > 1e-9:
            raise ValueError(f"{r['_file']}: json ve npz macro-F1 farklı")
        j = seeds.index(s)
        f1[k, j], acc[k, j] = m["macro_f1"], m["accuracy"]
        per_class.append([m["per_class_f1"][nm] for nm in names])
        if s == PRIMARY_SEED:
            y_pred0[0, idx], fold0[0, idx], logits0[idx] = z["y_pred"], k, z["logits"]
        run_rows.append(
            {
                "fold": k,
                "slot_seed": s,
                "seed": r["seed"],
                "file": r["_file"],
                "oof_macro_f1": m["macro_f1"],
                "oof_accuracy": m["accuracy"],
                "best_epoch": r["best_epoch"],
                "best_val_macro_f1": r["best_val_macro_f1"],
                "epochs_run": r["epochs_run"],
                "n_fit": r["n_fit"],
                "n_val": r["n_val"],
                "n_oof": r["n_oof"],
                "git_commit": r["git_commit"],
            }
        )
    if (y_pred0 < 0).any() or not np.array_equal(fold0[0], ref["fold_id0"]):
        raise ValueError(f"seed {PRIMARY_SEED} OOF'u tüm satırları char_lr fold'larıyla örtmüyor")

    fold_mean = f1.mean(axis=1)  # fold skoru = seed ortalaması
    ref_f1 = np.array(ref["macro_f1_folds0"])
    d = fold_mean - ref_f1
    ratio = float(np.mean(np.array(ref["n_val0"]) / np.array(ref["n_train0"])))
    ratio_fit = float(np.mean([r["n_oof"] / r["n_fit"] for r in run_rows]))
    nb = nadeau_bengio(d, ratio) | {"fold_corr": float(np.corrcoef(fold_mean, ref_f1)[0, 1])}
    e_fin = e_final([r["best_epoch"] for r in used.values()])
    finals_summary = summarize_finals(finals, e_fin, used)
    when = when or report.now()
    commits = sorted({r["git_commit"] for r in run_rows})
    j0 = seeds.index(PRIMARY_SEED)

    preds_file = f"preds/cv_phase4_{NAME}_{when:%Y%m%d-%H%M%S}.npz"
    _save_npz(
        Path(out_dir) / preds_file,
        row_id=ref["row_id"],
        y_true=ref["y_true"],
        y_pred=y_pred0,
        fold_id=fold0,
        logits=logits0,
    )
    oof_obj = {
        "experiment": f"phase4_{NAME}",
        "phase": 4,
        "timestamp": when.isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "split": ref["split"],
        "model_id": c["model_id"],
        "revision": c["revision"],
        "seed": PRIMARY_SEED,
        "label_names": names,
        "cv": {
            "scheme": (
                f"make_cv {c['cv_n_splits']} fold, random_state {c['cv_random_state']} "
                f"({REFERENCE} repeat 0 ile aynı fold'lar; fold_id[0] + phase4_folds.json'dan "
                f"okundu); seed {PRIMARY_SEED}"
            ),
            "n_folds": len(folds),
            "n_repeats": 1,
            "macro_f1_folds": f1[:, j0].tolist(),
            "macro_f1_mean": float(f1[:, j0].mean()),
            "accuracy_folds": acc[:, j0].tolist(),
            "pooled_oof": oof_metrics(ref["y_true"], y_pred0[0], names),
        },
        "preds_file": preds_file,
        "sources": [r["file"] for r in run_rows if r["slot_seed"] == PRIMARY_SEED],
        "reseeded_folds": [k for k, s in collapsed if s == PRIMARY_SEED],
        "reseeded_folds_note": (
            f"Bu fold'larda seed {PRIMARY_SEED} koşusu çöktü (RULES['collapse']); o fold'ların "
            f"OOF'u seed {PRIMARY_SEED + c['collapse_reseed_offset']} koşusundandır."
        ),
        "source_commits": commits,
        "fold_reference": {"file": ref["file"], "preds_file": ref["preds_file"]},
        "fold_source": folds_info(fold_file),
        "notes": (
            f"Seed {PRIMARY_SEED} CV koşularının OOF tahminleri (her fold kendi modelinden; final "
            "model değil). paired_test --power birincil çift (berturk_ft - char_lr) için okur. "
            "Logitler fp16 autocast altında hesaplandı, float32 saklandı."
        ),
        "test": None,
    }
    p_oof = report.write_metrics(oof_obj, f"phase4_{NAME}", out_dir, when=when)

    seed_std = f1.std(axis=1, ddof=1)
    best_eps = [int(r["best_epoch"]) for r in used.values()]
    max_ep = c["num_train_epochs"]
    df_seed = len(seeds) - 1
    sigma_ci = sigma_ci_factors(df_seed)
    df_pool = len(folds) * df_seed  # 5 · 2 = 10
    pooled_seed_std = float(np.sqrt(np.mean(seed_std**2)))  # sqrt(ortalama s²)
    sigma_ci_pool = sigma_ci_factors(df_pool)
    c4 = c4_factor(len(seeds))
    epochs_lt_max = [int(r["epochs_run"]) < max_ep for r in used.values()]
    eff = c["effective_batch_size"]
    upd_cv = [
        r["schedule"]["max_steps"] / r["schedule"]["num_train_epochs"]
        for r in used.values()
        if "schedule" in r
    ]
    upd_final = [
        f["schedule"]["max_steps"] / f["schedule"]["num_train_epochs"]
        for f in finals.values()
        if "schedule" in f
    ]
    if not upd_cv:
        upd_cv = [r["n_fit"] / eff for r in run_rows]
    if not upd_final:
        upd_final = [len(ref["row_id"]) / eff]
    collapsed_detail = []
    for k, s in collapsed:
        o, rr = runs[(k, s)], runs[(k, s + c["collapse_reseed_offset"])]
        collapsed_detail.append(
            {
                "fold": k,
                "seed": s,
                "reseed": s + c["collapse_reseed_offset"],
                "original": {
                    "file": o["_file"],
                    "best_val_macro_f1": o["best_val_macro_f1"],
                    "oof_macro_f1": o["oof"]["macro_f1"],
                },
                "reseed_run": {
                    "file": rr["_file"],
                    "best_val_macro_f1": rr["best_val_macro_f1"],
                    "oof_macro_f1": rr["oof"]["macro_f1"],
                },
                "reseed_also_collapsed": bool(rr["collapsed"]),
            }
        )
    summary = {
        "experiment": "phase4_summary",
        "phase": 4,
        "timestamp": when.isoformat(timespec="seconds"),
        "git_commit": report.git_commit(),
        "fold_source": folds_info(fold_file),
        "split": ref["split"],
        "model_id": c["model_id"],
        "revision": c["revision"],
        "label_names": names,
        "cv": {
            "seeds": seeds,
            "macro_f1_by_fold_seed": f1.tolist(),
            "macro_f1_folds": fold_mean.tolist(),
            "macro_f1_mean": float(fold_mean.mean()),
            "macro_f1_nb_ci95_ratio_char_lr": nb_mean_ci(fold_mean, ratio),
            "ratio_char_lr": ratio,
            "macro_f1_nb_ci95_ratio_bert_fit": nb_mean_ci(fold_mean, ratio_fit),
            "ratio_bert_fit": ratio_fit,
            "macro_f1_fold_std": float(fold_mean.std(ddof=1)),
            "macro_f1_seed_std_by_fold": seed_std.tolist(),
            "macro_f1_seed_std_mean": float(seed_std.mean()),
            "macro_f1_seed_std_pooled": pooled_seed_std,
            "macro_f1_seed_std_pooled_df": df_pool,
            "macro_f1_seed_std_pooled_sigma_ci95": [
                pooled_seed_std * sigma_ci_pool[0],
                pooled_seed_std * sigma_ci_pool[1],
            ],
            "macro_f1_by_seed": dict(zip(map(str, seeds), f1.mean(axis=0).tolist(), strict=True)),
            "macro_f1_by_seed_std": float(f1.mean(axis=0).std(ddof=1)),
            "accuracy_folds": acc.mean(axis=1).tolist(),
            "accuracy_mean": float(acc.mean()),
            "per_class_f1_mean": dict(zip(names, np.mean(per_class, axis=0).tolist(), strict=True)),
        },
        "runs": run_rows,
        "collapsed_slots": collapsed_detail,
        "e_final": e_fin,
        "best_epochs": [r["best_epoch"] for r in run_rows],
        "best_epoch_distribution": {str(e): best_eps.count(e) for e in range(1, max_ep + 1)},
        "n_best_at_max_epoch": best_eps.count(max_ep),
        "frac_epochs_run_lt_max": float(np.mean(epochs_lt_max)),
        "env": env_key(env_recs[0]["env"]),
        "finals": finals_summary,
        "final_rules": FINAL_RULES,
        "superseded_files": superseded_files(out_dir),
        "failed_attempts": read_failed_attempts(out_dir),
        "vs_reference": {
            "a": NAME,
            "b": REFERENCE,
            "reference_file": ref["file"],
            "reference_macro_f1_folds_repeat0": ref_f1.tolist(),
            "reference_macro_f1_mean_repeat0": float(ref_f1.mean()),
            "d_folds": d.tolist(),
            "nadeau_bengio": nb,
            "val_train_ratio": ratio,
            "val_train_ratio_rationale": (
                f"{REFERENCE}'nin kullandığı oran: repeat 0 dış fold'larında n_val/n_train "
                f"(ortalama {np.mean(ref['n_val0']):.0f}/{np.mean(ref['n_train0']):.0f} = "
                f"{ratio:.3f}). Faz 2 ve 3'teki char_lr karşılaştırmalarıyla aynı tanım. "
                "Düzeltme terimi, dış bölümlemenin yarattığı fold'lar arası train örtüşmesini "
                "(korelasyonu) yaklaşık modeller; iki prosedür aynı dış bölümlemeyi paylaşır."
            ),
            "sensitivity_bert_fit_ratio": {
                "val_train_ratio": ratio_fit,
                "nadeau_bengio": nadeau_bengio(d, ratio_fit),
                "note": (
                    "BERT'in gerçek eğitim boyutuyla oran (n_oof / n_fit); SE daha büyük, daha "
                    "muhafazakâr. Yalnız duyarlılık; birincil değer val_train_ratio'dur."
                ),
            },
            "inference": "keşifsel (tek karşılaştırma, Holm yok); test iddiası Faz 5'te",
        },
        "notes": {
            "cv_comparison": RULES["cv_comparison"],
            "input_text": RULES["input_text"],
            "training_size": (
                f"CV'de BERT ortalama {np.mean([r['n_fit'] for r in run_rows]):.0f} satırla "
                f"(validation ayrıldı), {REFERENCE} {int(np.mean(ref['n_train0']))} satırla "
                "eğitildi; karşılaştırma BERT için muhafazakâr."
            ),
            "fold_std": "5 fold skorunun (seed ortalaması) std'si (ddof=1); CI değil",
            "seed_std": (
                f"aynı fold'da {len(seeds)} seed'in std'si (ddof=1); eğitim (seed) "
                f"değişkenliği. Her biri {df_seed} serbestlik dereceli: normal varsayımıyla σ için "
                f"%95 aralık kabaca [{sigma_ci[0]:.2f}·s, {sigma_ci[1]:.1f}·s]; tek fold'un seed "
                "std'si çok belirsizdir."
            ),
            "seed_std_sigma_ci95_factors": sigma_ci,
            "macro_f1_seed_std_mean": (
                f"fold başına s'lerin ortalaması; E[s] = c4·σ, c4({len(seeds)}) ≈ {c4:.3f}: σ'yı "
                "aşağı yanlı tahmin eder. Havuzlanmış değer tercih edilir."
            ),
            "macro_f1_seed_std_pooled": (
                f"sqrt(ortalama s²), df = {len(folds)}·{df_seed} = {df_pool} (fold'lar arası eşit "
                "seed varyansı varsayımı). σ için %95 aralık (normal varsayımı, χ²): "
                f"[{sigma_ci_pool[0]:.2f}·s_p, {sigma_ci_pool[1]:.2f}·s_p] -> "
                "macro_f1_seed_std_pooled_sigma_ci95."
            ),
            "macro_f1_by_seed_std": (
                "seed başına 5-fold ortalamalarının std'si (ddof=1). ~σ_seed/√5 ölçeği yalnız "
                "seed'in fold'lar arası ORTAK bir ana etkisi yoksa geçerlidir. set_seed başlığı "
                "her fold'da aynı başlattığı (ve veri sırasını aynı RNG'den çektiği) için ana "
                "etki olasıdır; o durumda ≈ √(σ²_ana + σ²_etk/5). Tek koşunun seed değişkenliği "
                "değildir."
            ),
            "macro_f1_nb_ci95": (
                "BERT kendi CV ortalaması (fold skoru = seed ortalaması, J=5) için Nadeau-Bengio "
                "düzeltmeli %95 CI iki oranla: ratio_char_lr (n_val/n_train = 726/2904 = 0.25; "
                "char_lr ile birincil karşılaştırmanın tanımı) ve ratio_bert_fit (n_oof/n_fit, "
                "~0.31; BERT'in gerçek eğitim boyutu, daha geniş)."
            ),
            "frac_epochs_run_lt_max": (
                f"epochs_run < {max_ep} olan kullanılan koşuların oranı. Erken durmanın tamamını "
                f"saymaz: patience {c['early_stopping_patience']} ile best_epoch = "
                f"{max_ep - c['early_stopping_patience']} olan koşular {max_ep}. epoch'ta durur ve "
                f"epochs_run = {max_ep} görünür."
            ),
            "best_epoch_censoring": (
                f"n_best_at_max_epoch: best epoch'u üst sınır {max_ep}'te olan koşu sayısı; bu "
                "koşularda daha uzun eğitim daha iyi olabilirdi (sağdan sansür). E_final de "
                f"{max_ep}'i geçemez. 15 koşuda n_best_at_max_epoch ≤ 7 iken medyan (E_final, 8. "
                "sıra istatistiği) sansürden etkilenmez."
            ),
            "e_final_units": (
                f"E_final epoch biriminde aktarılır: CV'de epoch başına ~{np.mean(upd_cv):.0f} "
                f"güncelleme (~n_fit/16), final'de ~{np.mean(upd_final):.0f} (3630/16). Aynı "
                "epoch sayısı finalde daha çok güncelleme demektir; lr eğrisinin biçimi (5 epoch "
                "üzerinden) aynıdır."
            ),
            "reproducibility": (
                "fp16 ve CUDA çekirdekleri (cuDNN, atomik toplamlar) nedeniyle aynı seed bit "
                "düzeyinde tekrarlanamaz; 'seed 42 final modeli' kaydedilen dosyadır (model_dir), "
                "yeniden eğitim değil."
            ),
            "collapse": RULES["collapse"],
            "dirty_commits": any(cm.endswith("-dirty") for cm in commits),
        },
        "source_commits": commits,
        "primary_oof_file": p_oof.name,
        "rules": RULES,
        "ft_config": FT_CONFIG,
        "test": None,
    }
    p_sum = report.write_metrics(summary, "phase4_summary", out_dir, when=when)
    return p_sum, p_oof


def _final_brief(f: dict | None) -> dict | None:
    if f is None:
        return None
    return {
        "file": f["_file"],
        "seed": f["seed"],
        "collapsed": f["collapsed"],
        "train_in_sample_macro_f1": f["train_in_sample_macro_f1"],
        "train_in_sample_accuracy": f.get("train_in_sample_accuracy"),
        "train_loss_last_epoch": f.get("train_loss_last_epoch"),
        "epochs_run": f["epochs_run"],
        "model_dir": f["model_dir"],
        "weights_sha256": f.get("weights_sha256"),
    }


def summarize_finals(finals: dict[int, dict], e_fin: int, used: dict) -> dict:
    """Final json'larını CV ve FINAL_RULES['final_collapse'] ile çapraz denetler; slot başına
    kullanılan modeli ve birincil test modelini döndürür. Tutarsızlıkta ValueError."""
    off = FT_CONFIG["collapse_reseed_offset"]
    seeds = list(FT_CONFIG["seeds"])
    for rec in finals.values():
        check_final(rec, e_fin, used, smoke=False)
        if rec.get("reseed_of") is None and rec["seed"] not in seeds:
            raise ValueError(f"{rec['_file']}: seed {rec['seed']} bir FT_CONFIG seed'i değil")
        if rec.get("reseed_of") is not None:
            orig = finals.get(rec["reseed_of"])
            if orig is None or not orig["collapsed"]:
                raise ValueError(
                    f"{rec['_file']}: yeniden eğitim var ama slot {rec['reseed_of']} orijinali "
                    "yok ya da çökmemiş (FINAL_RULES yalnız çöküşte yeniden eğitir)"
                )
    slots = [s for s in seeds if s in finals]
    chosen, missing, collapsed = select_finals(finals, slots)
    if missing:
        raise ValueError(f"Çöken final slotlarının yeniden eğitimi eksik (seed): {missing}")
    primaries = [r for r in finals.values() if r["primary_for_test"]]
    want = [chosen[PRIMARY_SEED]] if PRIMARY_SEED in chosen else []
    if [r["_file"] for r in primaries] != [r["_file"] for r in want]:
        raise ValueError(
            f"primary_for_test tutarsız: işaretli {[r['_file'] for r in primaries]}, "
            f"beklenen {[r['_file'] for r in want]}"
        )
    out = {
        "rule": FINAL_RULES["final_collapse"],
        "slots": {},
        "collapsed_slots": collapsed,
        "missing_slots": [s for s in seeds if s not in finals],
    }
    for s in slots:
        o, rr = finals[s], finals.get(s + off) if finals[s]["collapsed"] else None
        u = chosen[s]
        out["slots"][str(s)] = {
            "used_seed": u["seed"],
            "used_file": u["_file"],
            "used_model_dir": u["model_dir"],
            "used_train_in_sample_macro_f1": u["train_in_sample_macro_f1"],
            "collapsed": o["collapsed"],
            "original": _final_brief(o),
            "reseed": _final_brief(rr),
            "reseed_also_collapsed": bool(rr["collapsed"]) if rr is not None else None,
            "primary_for_test": u["primary_for_test"],
        }
    if want:
        p = want[0]
        out |= {
            "primary_slot": PRIMARY_SEED,
            "primary_seed": p["seed"],
            "primary_file": p["_file"],
            "primary_model_dir": p["model_dir"],
            "primary_model_sha256": p.get("weights_sha256"),
        }
    return out


def cmd_summarize(args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else METRICS_DIR
    p_sum, p_oof = summarize(out_dir, args.ref_dir, folds_path=args.folds_file)
    s = json.loads(p_sum.read_text(encoding="utf-8"))
    cv, vs = s["cv"], s["vs_reference"]
    nb = vs["nadeau_bengio"]
    print(p_sum)
    print(p_oof)
    print(
        f"{NAME}: macro-F1 {cv['macro_f1_mean']:.4f}  fold std {cv['macro_f1_fold_std']:.4f}  "
        f"seed std (ort.) {cv['macro_f1_seed_std_mean']:.4f}  E_final {s['e_final']}"
    )
    pairs = zip(cv["macro_f1_folds"], vs["reference_macro_f1_folds_repeat0"], strict=True)
    for k, (a, b) in enumerate(pairs):
        print(f"  fold {k}: {NAME} {a:.4f}  {REFERENCE} {b:.4f}  d {a - b:+.4f}")
    print(
        f"  d ort. {nb['mean_diff']:+.4f}  N-B %95 CI [{nb['ci95'][0]:+.4f}, {nb['ci95'][1]:+.4f}]"
        f"  p_NB {nb['p']:.3g} (oran {vs['val_train_ratio']:.3f})"
    )
    if s["collapsed_slots"]:
        print("  çöken koşular:", [(c["fold"], c["seed"]) for c in s["collapsed_slots"]])
    fin = s["finals"]
    for sd, v in fin["slots"].items():
        print(f"  final slot {sd}: kullanılan seed {v['used_seed']} ({v['used_file']}), çöktü "
              f"{v['collapsed']}, birincil {v['primary_for_test']}")  # fmt: skip
    if "primary_model_dir" in fin:
        print(f"  birincil test modeli: seed {fin['primary_seed']} -> {fin['primary_model_dir']}")


def cmd_folds(args) -> None:
    """YEREL, bir kez: iç validation fold'larını make_cv ile üretir ve FOLDS_PATH'e yazar (var
    olanın üzerine yazmaz). Dış fold'lar char_lr fold_id[0] ile doğrulanır; sklearn >= 1.8 gerekir
    (char_lr fold'ları bu davranışla üretildi; eski sürümde check_reference hata verir)."""
    from ttc4900.data import label_names, load_train

    train = load_train()
    names = label_names()
    check_labels(train, names)
    obj = fold_file_dict(train, load_reference(args.ref_dir), names)
    out = write_fold_file(obj, args.folds_file)
    print(out, obj["n_val"], obj["generated_with"])


def cmd_verify(args) -> None:
    """Eğitimsiz doğrulama (GPU gerekmez, make_cv çalıştırılmaz): load_train() parmak izi = fold
    dosyası, fold dosyası = char_lr referansı, plan kurulur ve gruplar ayrık. Hata = eğitim yok."""
    d = load_checked_train(args.ref_dir, args.folds_file)
    fp = train_fingerprint(d["train"])
    print(f"[verify] {d['folds']['_file']} sha256 {d['folds']['_sha256'][:12]}; referans "
          f"{d['ref']['preds_file']}")  # fmt: skip
    print(f"[verify] train parmak izi (dosyayla aynı): {fp}")
    for p in d["plans"]:
        print(f"  fold {p.fold}: fit {len(p.fit)}  val {len(p.val)}  oof {len(p.outer_val)}")
    print("[verify] tamam: fold'lar dosyalardan kuruldu, make_cv çalıştırılmadı.")


# --------------------------------------------------------------------------- CLI
def resolve_dirs(args) -> tuple[Path, Path, Path]:
    """(out_dir, ckpt_dir, model_dir). Smoke gerçek outputs/metrics'e yazmaz."""
    out, ckpt, model = args.out_dir, args.ckpt_dir, getattr(args, "model_dir", None)
    if args.smoke:
        out = Path(out) if out else SMOKE_DIR / "metrics"
        if out.resolve() == METRICS_DIR.resolve():
            raise SystemExit("--smoke gerçek outputs/metrics dizinine yazmaz.")
        model = Path(model or SMOKE_DIR / "models")
        if model.resolve() == DEFAULT_MODEL_DIR.resolve():
            raise SystemExit("--smoke gerçek final model dizinine (outputs/models) yazmaz.")
        return out, Path(ckpt or SMOKE_DIR / "ckpt"), model
    return (
        Path(out or METRICS_DIR),
        Path(ckpt or DEFAULT_CKPT_DIR),
        Path(model or DEFAULT_MODEL_DIR),
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m ttc4900.finetune", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument(
            "--out-dir",
            type=Path,
            default=None,
            help="metrik json/npz dizini (varsayılan outputs/metrics; Colab: Drive)",
        )
        p.add_argument(
            "--ref-dir",
            type=Path,
            default=METRICS_DIR,
            help=f"{REFERENCE} referans json/npz dizini (repo içi outputs/metrics)",
        )
        p.add_argument(
            "--folds-file",
            type=Path,
            default=FOLDS_PATH,
            help="iç validation fold dosyası (yerelde 'folds' ile bir kez üretilir, commit'lenir)",
        )

    for name in ("cv", "final"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--ckpt-dir", type=Path, default=None, help="geçici checkpoint dizini")
        p.add_argument("--seeds", type=int, nargs="+", default=list(FT_CONFIG["seeds"]))
        p.add_argument("--smoke", action="store_true", help="<=64 satır, 1 epoch, CPU, fp16 yok")
        p.add_argument("--smoke-model", choices=SMOKE_MODELS, default="tiny")
        if name == "cv":
            folds = list(range(FT_CONFIG["cv_n_splits"]))
            p.add_argument("--folds", type=int, nargs="+", default=folds)
        else:
            p.add_argument(
                "--model-dir", type=Path, default=None, help="final modeller: <model-dir>/<seed>/"
            )
    common(sub.add_parser("summarize"))
    common(sub.add_parser("folds", help="YEREL, bir kez: fold dosyasını make_cv ile üretir"))
    common(sub.add_parser("verify", help="eğitimsiz: train parmak izi + fold dosyası doğrulaması"))
    return ap


def main(argv=None) -> None:
    ap = _parser()
    args = ap.parse_args(argv)
    if args.cmd in ("cv", "final"):
        extra = set(args.seeds) - set(FT_CONFIG["seeds"])
        if extra:
            ap.error(f"seed'ler FT_CONFIG['seeds'] içinden olmalı; fazla: {sorted(extra)}")
        if args.cmd == "cv" and not set(args.folds) <= set(range(FT_CONFIG["cv_n_splits"])):
            ap.error(f"fold'lar 0..{FT_CONFIG['cv_n_splits'] - 1} olmalı")
    cmds = {"cv": cmd_cv, "final": cmd_final, "summarize": cmd_summarize, "folds": cmd_folds,
            "verify": cmd_verify}  # fmt: skip
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()

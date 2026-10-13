"""Faz 4: BERTurk fine-tuning. Dondurulmuş konfigürasyon (FT_CONFIG) ve kurallar (RULES).

Bu dosyadaki değerler EĞİTİMDEN ÖNCE ayrı bir commit'le donduruldu ("Faz 4a"). Ayar taraması
yapılmaz; CV ya da test sonucuna göre hiçbir değer değiştirilmez (tek istisna RULES["collapse"]).
Eğitim kodu (Trainer, Colab CLI'ı) Faz 4b'de bu dosyaya eklenir; FT_CONFIG ve RULES ona dokunmaz.

Sezgi: Faz 3'te encoder donmuştu, öğrenilen yalnızca φ(x) üstündeki LR'ydi (konveks problem).
Burada AutoModelForSequenceClassification = encoder + [CLS] pooler + dropout + Linear(768 -> 7);
kayıp L(θ, W, b) = (1/n) Σ_i -log softmax(W φ_θ(x_i) + b)_{y_i} ve TÜM θ (~110M) güncellenir.
Problem konveks değildir, sonuç başlangıca/veri sırasına/dropout'a bağlıdır: çok seed gerekir.

İnsan okuması: docs/faz4_protokol.md. Test yöntemi: ttc4900.test_plan.TEST_PLAN.
"""

from __future__ import annotations

from ttc4900.data import RANDOM_STATE
from ttc4900.embeddings import ENCODERS
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

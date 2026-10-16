# Faz 4 protokolü: BERTurk fine-tuning

Bu kurallar **eğitimden önce** ayrı bir commit'le donduruldu ("Faz 4a"). Kaynakları:
- `src/ttc4900/finetune.py`: `FT_CONFIG`, `RULES`, `LITERATURE`
- Test yöntemi: `src/ttc4900/test_plan.py` (`TEST_PLAN`, ayrıntısı `docs/test_protokol.md`)

Ayar taraması yapılmaz. CV ya da test sonucuna göre hiçbir değer değiştirilmez. Önceden
tanımlı iki istisna vardır:
- "Çöken koşu" (`RULES["collapse"]`, Faz 4a): CV koşuları için.
- "Çöken final model" (`FINAL_RULES["final_collapse"]`, Faz 4a eki): final modeller için.
  Faz 4a'dan sonra, ama **tüm eğitimden (CV dahil) önce** ayrı bir commit'le donduruldu.

## Model
- `dbmdz/bert-base-turkish-cased` @ `b6e1de16c983e0f2c70664591ea3f22810072608`. Faz 3'teki
  `berturk_mean` ile aynı model ve revision.
- `AutoModelForSequenceClassification(num_labels=7)`: encoder + [CLS] pooler + dropout +
  Linear(768→7).
- Faz 3'ten farkı: orada yalnızca donmuş temsilin üstündeki LR öğreniliyordu. Burada ~110M
  parametrenin hepsi güncellenir. Bu problem konveks değildir, bu yüzden birden fazla seed gerekir.

## Dondurulan hiperparametreler
| Parametre | Değer | Gerekçe |
|---|---|---|
| lr | 2e-5 | Devlin vd. 2019 aralığı {2,3,5}e-5. Sun vd. 2019 2e-5 kullanır. Mosbach vd. 2021 küçük veride küçük lr önerir |
| Etkin batch | 16 (8 × birikim 2) | Devlin {16, 32}. 512 token fp16'da T4 belleği için 8'lik mikro-batch kullanılır. BERT LayerNorm kullandığı için birikim eşdeğerdir |
| Epoch | En fazla 5, early stopping patience 2 (val macro-F1) | Devlin 2–4 önerir, Mosbach küçük veride daha uzun eğitim. 5 + ES bu ikisi arasında uzlaşma |
| Schedule | warmup_ratio 0.1, linear decay | Sun vd. 2019 |
| weight_decay / grad norm | 0.01 / 1.0, AdamW (torch) | BERT standardı |
| max_length | 512, baştan kesme, dinamik padding | Faz 3 ile aynı. Sun'da head+tail biraz daha iyi, ama tutarlılık için head |
| fp16 | True | T4 |
| Seed | {42, 43, 44} | Dodge vd. 2020. CLAUDE.md çoklu seed ister |

**Çöken koşu (Mosbach 2021; CV koşuları için):**
- En iyi epoch'taki val macro-F1 < 0.5 ise koşu çökmüş sayılır.
- Koşu seed + 100 ile **bir kez** yeniden koşulur. İki koşu da raporlanır, yeniden koşu
  kullanılır.

## CV
- Dış fold'lar `make_cv(train, n_splits=5, random_state=42)` ile kurulur. Bunlar char_lr nested
  CV'sinin 1. tekrarıyla aynı fold'lardır; eşleşme kodda npz'den doğrulanır.
- 5 fold × 3 seed = 15 koşu.
- **Fold kaynağı (Faz 4b düzeltmesi, eğitimden önce; hiperparametre ve kural değişmedi):**
  - `StratifiedGroupKFold(shuffle=True)` aynı girdi ve `random_state` ile sklearn < 1.8 ve
    ≥ 1.8'de farklı fold'lar üretir. Aynı row_id/etiket girdisiyle denendi: 1.6.1 ve 1.7.2'de
    3630 satırın 2919'unun dış fold'u farklı; 1.8.0 ve 1.9.1'de char_lr ile birebir aynı. Colab'da
    sklearn 1.6.1 olduğu için ilk denemede `check_reference` eğitimden önce durdu.
  - Bu yüzden eğitim ortamında (`cv`, `final`, `--smoke`) ve `summarize`'da `make_cv`
    çalıştırılmaz. Dış fold'lar char_lr npz'indeki `fold_id[0]`'dan okunur. İç validation
    parçaları yerelde (sklearn 1.9.1) yukarıdaki `make_cv` çağrılarıyla **bir kez** üretilip
    `outputs/splits/phase4_folds.json`'a yazıldı (`finetune folds`; üzerine yazılmaz).
  - Fold'ların kendisi değişmedi: Faz 4a'daki tanımın, char_lr ile aynı sklearn davranışındaki
    çıktısıdır. Dosyadan kurulan plan yerelde `make_cv` planıyla birebir aynıdır (test).
  - Eğitimden önce: `load_train()`'in parmak izi (row_id sırası, etiketler, metinlerin sha256'sı)
    dosyadakiyle, dosyadaki char_lr npz adı ve `fold_id[0]` sha256'sı referansla aynı olmalı. Val
    dış train'in alt kümesi, fit/val ve dış train/val yakın-tekrar grupları ayrık olmalı. Biri
    tutmazsa eğitim başlamaz. Colab sürümleriyle (Python 3.13, datasets 4.8.5, pandas 2.2.3)
    yerelde denendi: parmak izi aynı.
  - Val ve fit boyutları dosyadaki `n_val` / `n_fit` ile aynı olmalı ve ikisi de 7 sınıfın
    hepsini içermeli.
  - Her koşu json'u ve özet, fold dosyasının içerik sha256'sını taşır (`folds` / `fold_source`;
    kanonik JSON üzerinden, satır sonundan bağımsız). `summarize` her CV koşusunun bu sha'yla
    eğitildiğini denetler.
  - Fold dosyasının içerik sha256'sı eğitimden önce sabitlendi: `9c0c31e8f57c0a34821f39a34c13958871c0f4df570dcfb7ef08b8e94c5cdb4b`
    (`tests/test_finetune.py::test_fold_file_content_pinned`).
- **Validation:** Her dış fold'un train kısmında `make_cv(tr_df, n_splits=5, random_state=42)`
  çağrılır ve ilk fold validation olur (~581 val / ~2323 fit).
  - Trainer'ın `eval_dataset`'i her zaman bu parçadır.
  - Testle doğrulananlar:
    - Fit ile val'in grup id'leri ayrık.
    - Val, dış train'in alt kümesi.
    - Val ile dış val kesişmiyor.
    - Val ile test_ids kesişmiyor (yalnız id düzeyinde).
- **Fold skoru:** 3 seed'in ortalaması. Seed std'si ayrı raporlanır.
- **char_lr ile karşılaştırma (keşifsel):** d_j = BERT_j − char_lr_j (repeat 0), Nadeau-Bengio
  ile (J=5, df=4). Karşılaştırılan şey prosedürlerdir: char_lr'de iç ayarlama var, BERT'te yok.
- **Eğitim boyutu notu:** CV'de BERT ~2323 satırla, char_lr 2904 satırla eğitilir. Bu yüzden CV
  karşılaştırması BERT için muhafazakârdır. Final modellerde ikisi de 3630 satırla eğitilir.

## Final model ve teste taşıma
- **Final model:**
  - Tüm train (3630 satır) ile, validation olmadan eğitilir.
  - E_final = 15 CV koşusunun en iyi epoch'larının medyanı.
  - Schedule CV ile aynıdır (5 epoch üzerinden warmup + decay) ve eğitim E_final'da durur.
- **Çöken final model (Faz 4a eki, tüm eğitimden önce donduruldu; `FINAL_RULES`):**
  - Final modelde validation olmadığı için yukarıdaki çöküş kuralı uygulanamaz. Yerine:
    final modelin tüm train (3630) üzerindeki in-sample macro-F1'i < 0.5 ise model çökmüş sayılır.
  - Aynı E_final ile seed + 100 kullanılarak **bir kez** yeniden eğitilir. İki model de
    raporlanır, yeniden eğitilen kullanılır (o da çökerse yine o kullanılır ve işaretlenir).
  - Seed 42 çökerse teste taşınan birincil model seed 142 modelidir (`TEST_PLAN
    ['primary_seed_rule']`).
  - Ölçüt yalnızca train'dir; test ya da CV skoru kullanılmaz. Eşik CV kuralıyla aynıdır
    (çöken model sabit sınıf tahmin eder, macro-F1 ≈ 0.04). In-sample F1 iyimser olduğu için
    aynı eşik finalde CV'dekinden gevşektir; kural önceden kayıtlı olduğu için değiştirilmez.
  - Faz 5 birincil modeli seed numarasından değil, `phase4_summary`'deki
    `finals.primary_model_dir` alanından okur. Testten önce ağırlıkların sha256'sını
    (`primary_model_sha256`) doğrular ve train in-sample macro-F1'ini `load_train()` ile yeniden
    hesaplayarak kaydı teyit eder.
- **Teste taşınan model:** seed 42 final modeli (yukarıdaki istisna dışında).
  - CV sonucundan bağımsızdır: char_lr'yi geçip geçmediğine bakılmaz.
  - Seed 43 ve 44'ün test skorları betimseldir (p yok).
- **Birincil aile:** m=1 (`berturk_ft − char_lr`).
  - Kullanılan yöntem: exact McNemar, grup bootstrap ve üç koşullu karar kuralı
    (`docs/test_protokol.md`).
- **`berturk_ft − bge_m3`:** Yalnızca betimsel CI ile raporlanır; p hesaplanmaz. Karar Faz 5a'da,
  test açılmadan `test_plan.DESCRIPTIVE`'e yazıldı. İkincil aile m=4 olarak kaldı.
- Test sonucuna göre konfigürasyon, epoch ya da seed değiştirilmez.

## Girdi asimetrisi
Ham metin kullanılır. BERTurk tokenizer'ı U+0091–U+0094 karakterlerini siler, char_lr ise
bunları görür (`docs/shortcut_learning.md`). Bu asimetri char_lr lehinedir.

## Çıktılar
- **Her CV koşusu:**
  - `phase4_berturk_ft_f{k}_s{seed}_<ts>.json`: val eğrisi, en iyi epoch, OOF metrikleri.
  - OOF npz: row_id, y_true, y_pred, **logits**, fold_id. Logitler Faz 6'da temperature
    scaling için saklanır.
- **Özet:** `phase4_summary_<ts>.json`.
- **Seed 42 OOF'u:** `phase4_berturk_ft_<ts>.json` + npz (şekil 1 × 3630). Birincil güç analizi
  (`paired_test --power`) bu dosyayı okur.

## Colab
1. Kod, Faz 4b commit'ine checkout edilir.
2. Paket yükseltmesi yapılmaz: requirements'taki eğitim paketleri kuruluysa olduğu gibi
   kullanılır, yalnız eksik olan kurulur (pip'in kurulu bir paketi değiştirmeyeceği önce
   `--dry-run` ile doğrulanır). Torch kurulmaz. Sürümler yazdırılır ve metrik dizinine
   `phase4_colab_env_<ts>.json` olarak kaydedilir.
3. `torch.cuda.is_available()` kontrol edilir. GPU yoksa eğitim başlamaz. Ardından
   `finetune verify` (eğitimsiz) train parmak izini ve fold dosyasını doğrular; hata = eğitim yok.
4. Metrikler Drive'a yazılır ve koşu kopsa kaldığı yerden devam eder. Checkpoint'ler geçici
   diske yazılır ve silinir. Final modeller Drive'da kalır, git'e girmez.
5. Metrikler zip olarak yerele indirilir. `summarize` ve commit yerelde yapılır.

**T4 süre tahmini:** CV ~2.5–3.5 saat, final ~30 dakika. Smoke testinden sonra gerçek hızla
güncellenir.

## Test seti
Faz 4 test setine dokunmaz. Veri yalnızca `load_train()` ile okunur.

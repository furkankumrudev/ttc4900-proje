# Faz 4 protokolü: BERTurk fine-tuning

Bu kurallar **eğitimden önce** ayrı bir commit'le donduruldu ("Faz 4a"). Kaynakları:
- `src/ttc4900/finetune.py`: `FT_CONFIG`, `RULES`, `LITERATURE`
- Test yöntemi: `src/ttc4900/test_plan.py` (`TEST_PLAN`, ayrıntısı `docs/test_protokol.md`)

Ayar taraması yapılmaz. CV ya da test sonucuna göre hiçbir değer değiştirilmez. Tek önceden
tanımlı istisna aşağıdaki "çöken koşu" kuralıdır.

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

**Çöken koşu (Mosbach 2021):**
- En iyi epoch'taki val macro-F1 < 0.5 ise koşu çökmüş sayılır.
- Koşu seed + 100 ile **bir kez** yeniden koşulur. İki koşu da raporlanır.

## CV
- Dış fold'lar `make_cv(train, n_splits=5, random_state=42)` ile kurulur. Bunlar char_lr nested
  CV'sinin 1. tekrarıyla aynı fold'lardır; eşleşme kodda npz'den doğrulanır.
- 5 fold × 3 seed = 15 koşu.
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
- **Teste taşınan model:** seed 42 final modeli.
  - CV sonucundan bağımsızdır: char_lr'yi geçip geçmediğine bakılmaz.
  - Seed 43 ve 44'ün test skorları betimseldir (p yok).
- **Birincil aile:** m=1 (`berturk_ft − char_lr`).
  - Kullanılan yöntem: exact McNemar, grup bootstrap ve üç koşullu karar kuralı
    (`docs/test_protokol.md`).
- **`berturk_ft − bge_m3`:** Öneri, yalnızca betimsel CI ile raporlanması. Son karar, test
  açılmadan önceki son commit'te `FAMILIES`'e yazılır.
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
2. Torch yeniden kurulmaz: requirements'tan torch satırı atlanır.
3. `torch.cuda.is_available()` kontrol edilir. GPU yoksa eğitim başlamaz.
4. Metrikler Drive'a yazılır ve koşu kopsa kaldığı yerden devam eder. Checkpoint'ler geçici
   diske yazılır ve silinir. Final modeller Drive'da kalır, git'e girmez.
5. Metrikler zip olarak yerele indirilir. `summarize` ve commit yerelde yapılır.

**T4 süre tahmini:** CV ~2.5–3.5 saat, final ~30 dakika. Smoke testinden sonra gerçek hızla
güncellenir.

## Test seti
Faz 4 test setine dokunmaz. Veri yalnızca `load_train()` ile okunur.

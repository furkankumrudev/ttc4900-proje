# Faz 5 protokolü: final test değerlendirmesi

**Kaynak:** `src/ttc4900/final_eval.py` (`RULES`, `IN_SAMPLE_TOL`, `V1_FROZEN`) ve
`src/ttc4900/test_plan.py` (`TEST_PLAN`, `FAMILIES`, `DESCRIPTIVE`, `SENTENCES`). Bu belge ile
kaynak çelişirse kaynak geçerlidir. Test yöntemi: `docs/test_protokol.md`.

Hepsi **test seti açılmadan**, "test öncesi son commit"te donduruldu. Test sonucuna göre hiçbiri
değişmez.

## Adaylar
| Ad | Kaynak | Test için ne yapılır |
|---|---|---|
| `char_lr`, `char_lr_c1clean`, `word_lr`, `word_svc`, `word_cnb` | Faz 2 json'u, `final.best_params` | Tüm train'de bir kez fit |
| `bge_m3` | Faz 3 json'u (`C=0.001`) + train embedding önbelleği | Tüm train'de bir kez fit; test embedding'i bir kez üretilir |
| `berturk_ft` (seed 42, birincil), `berturk_ft_s43`, `berturk_ft_s44` | Faz 4 final ağırlıkları (`outputs/models/<seed>`) | Yeniden eğitilmez; yalnız tahmin |

## Yeniden eğitim kuralı (`RULES['refit']`)
- Klasik modeller ve bge_m3 + LR, **tüm train (split_v2, 3630 satır)** üzerinde, dondurulmuş
  `final.best_params` ile **bir kez** fit edilir.
- Testten önce ya da sonra **hiçbir yeniden seçim/ayar yapılmaz**: GridSearchCV, `make_cv`, iç CV
  ya da başka bir arama yok.
- Pipeline'ın sabit ayarları `baselines.SPECS` / `frozen.official_spec` ile aynıdır.
  `best_params`'ın her anahtarı spec'in grid'inde, değeri grid'deki bir değerdir; değilse kod
  durur.
- Testler (`tests/test_final_eval.py`):
  - Modülde arama/CV aracı yok.
  - Parametreler json'la aynı.
  - Tek fit, tüm train satırlarıyla.
  - Grid dışı değer reddedilir.

Gerçek kayıtlardaki donmuş değerler:

| Model | best_params |
|---|---|
| char_lr | C=100, char_wb (2,4) |
| char_lr_c1clean | C=100, char_wb (2,4) |
| word_lr | C=100, (1,1) |
| word_svc | C=1, (1,1) |
| word_cnb | alpha=0.1, (1,1) |
| bge_m3 | C=0.001 |

## Adımlar ve sıra
1. **`verify`** (yalnız train, test açılmaz). Her BERTurk final modeli için:
   - Ağırlık sha256'sı Faz 4 final json'undaki `weights_sha256` ile **tam eşit** olmalı.
   - `config.id2label`, split `label_names` ile aynı olmalı.
   - Yerel tokenizer sabit revision'ın `vocab.txt`'siyle aynı olmalı ve tüm train metinlerinde
     Hub revision tokenizer'ıyla aynı input_ids'i üretmeli.
   - Train in-sample macro-F1 yerel CPU'da (fp32, eval) yeniden hesaplanır.
     **|yeni − kayıt| ≤ 0.005** ve yeni ≥ 0.5 olmalı.
     - Kayıt Colab T4'te fp16 ile üretildi; tolerans bu sayısal farkı karşılar (yaklaşık 18
       satır).
   - Birincil model seed numarasından değil, `phase4_summary.finals.primary_model_dir`'den
     okunur. Bu Colab yoludur; son klasör adı `outputs/models/`'e eşlenir.
   - Model dizinindeki **tüm** dosyaların (ağırlık, config, tokenizer) sha256'sı kaydedilir.
   - Biri tutmazsa test **açılmaz**.
   - Çıktı: `phase5_model_check_<ts>.json`.
2. **`predict`** (testi **bir kez** açar). Ön koşullar:
   - Çalışma ağacı temiz.
   - Temiz bir commit'te geçmiş bir `model_check` kaydı var.
   - O commit'ten beri `src/`, `requirements.txt`, `outputs/metrics`, `outputs/splits` ve
     `outputs/dedup` değişmemiş. Muafiyet yalnızca **yeni eklenen** Faz 5 çıktılarına (`phase5_*`
     json'ları, `preds/test_*.npz`) tanınır; değiştirilen ya da silinen bir Faz 5 json'u kapıyı
     tetikler.
   - Bu yollarda izlenmeyen bir girdi dosyası yok.
   - Ağırlık sha256'ları ve model dizinindeki tüm dosyaların sha256'sı verify'dakiyle aynı.

   Fit'ler test metni belleğe alınmadan önce yapılır.
3. **`evaluate`**: yalnız npz'leri okur, metin okumaz.
4. **`v1-diagnosis`**: ana sonuçlardan **sonra** çalışır (aşağıda).

`load_train_test` yalnız `final_eval`'in `predict` ve `v1-diagnosis` adımlarında çağrılır
(`eda.py`'deki Faz 1 kullanımı tarihîdir). Bir AST testi bunu denetler.

## Kayıtlar: write-once
- Her aday için `outputs/metrics/preds/test_<model>.npz` dosyası yazılır.
  - İçerik: `row_id`, `y_true`, `y_pred`, `scores` ve `meta`.
  - `scores`: BERT'te logit, LR/CNB'de `predict_proba`, SVC'de `decision_function`. Faz 6
    kalibrasyonu için saklanır.
  - Yazma yöntemi: geçici dosya + `os.link`. Hedef varsa hata verir, üzerine yazmaz.
- `predict` yeniden çalıştırılırsa var olan npz yeniden üretilmez. Yalnızca satırları ve önceki
  manifest'teki sha256'sı denetlenir; eksik modeller üretilir.
- Manifest `phase5_test_preds_<ts>.json` şunları içerir:
  - Her dosyanın sha256'sı.
  - Kaynak json'lar.
  - Ortam sürümleri: python, torch, transformers, tokenizers, safetensors, sklearn, numpy,
    scipy, CPU, thread sayısı.
  - Model başına süre.
- bge_m3 test embedding'i ayrı bir önbellekte durur:
  - Dosya: `outputs/cache/emb_bge_m3_TEST_5617a9f6_512.npz`.
  - Meta: `side: "test"` ve `test_row_ids_sha256`.
  - `load_embeddings` (train yolu) `side: "test"` dosyasını reddeder.
  - Revision, max_length ve pooling Faz 3 ile aynıdır.

## Değerlendirme
- Bütün modeller aynı grup bootstrap ağırlıklarıyla ölçülür (B = 10000, seed 42; `split_v2`
  `components`).
- Birincil ve ikincil aile `paired_test.compare_family` ile değerlendirilir: exact McNemar → Holm
  → karar kuralı.
- Betimsel çiftler aynı replikatlardan fark ve CI ile verilir; p hesaplanmaz.
- BERT seed özeti: 3 seed'in test macro-F1'i, ortalama ± std (ddof=1).
- Her model için accuracy ve macro-F1 (ikisi de aynı w'lerden grup bootstrap CI ile) ve sınıf
  bazlı F1 raporlanır.
- Her çift için accuracy farkının grup bootstrap CI'ı **betimsel** olarak eklenir. Bu CI karar
  kuralına girmez. McNemar (accuracy) yönünü grup bağımlılığını hesaba katarak gösterir.
- Çıktı: `phase5_test_<ts>.json`.

### Cümle şablonları (`test_plan.SENTENCES`)
Sonuç dili test görülmeden sabitlendi. Kod `decide()` çıktısına göre bir şablon seçer ve yalnızca
sayıları doldurur.

| Durum | Şablonun özü |
|---|---|
| ayırt edildi | "{a}, test macro-F1'de {b}'den daha yüksek/düşük olarak ayırt edildi: fark, CI, n10/n01, ham ve Holm p, aile ve m. Üç koşul da sağlandı." |
| ayırt edilemedi (yön çelişkisi) | "McNemar (accuracy) <model adı> lehine, macro-F1 CI'ı ters yönde; iki ölçüt farklı büyüklükleri ölçtüğü için yön hakkında karar verilmez." |
| ayırt edilemedi | "Fark, CI, Holm p; sağlanan ve sağlanmayan koşullar tek tek (CI 0'ı içeriyorsa yön tutarlılığı 'değerlendirilemez'); eşdeğerlik iddiası değildir; test öncesi güç, α başına etiketli (ikincilde Holm'un en kötü adımı α/4 ve en iyi adımı α): OOF farkında güç, accuracy MDE, kaba macro-F1 MDE." |
| betimsel | "a − b (betimsel, p yok): skorlar, fark, CI; çıkarım için kullanılmaz." |
| seed özeti | "Üç değer tek tek; ortalama ± std ve aralık; ortalamanın güven aralığı değildir, birincil modeli de içerir; birincil karar yalnız seed 42 modeline dayanır." |
| seed tutarlılığı (birincil cümleye eklenir) | "Birincil farkın CI işareti; seed 43/44 − char_lr için fark, CI ve aynı yön / 0'ı içeriyor / ters yön. Betimsel, karara girmez." |
| sınırlılıklar | "McNemar grup bağımlılığını yok sayar (p iyimser olabilir); CI'lar marjinal; sınıf F1'leri CI'sız ve gürültülü; test fp32 CPU, CV fp16 Colab." |
| birincil koşul (birincil cümleye eklenir) | "İddia bu tek eğitilmiş modele koşuludur; seed belirsizliği ölçülmez." |
| CV–test notu | "CV'den farklı çıkması tek başına anomali değildir (eğitim boyutu, kazananın laneti)." |

## split_v1 sızıntı teşhisi (ayrı adım, `RULES['v1_diagnosis']`)
- **Ne zaman:** ana sonuçlar (`phase5_test` json'u) yazılıp **commit'lendikten sonra**, temiz
  commit'te. Kod bu json yoksa, json commit'lenmemişse, ağaç kirliyse ya da izlenmeyen girdi
  dosyası varsa çalışmaz. Sonuç ana tablolara girmez.
- **Model:** `char_lr`, dondurulmuş ayarlarla (**C=100, char_wb, ngram (2,4)**).
  - **Ayarlama yok.** Değerlerin Faz 2 json'uyla aynı olduğu kodda denetlenir.
  - split_v1 train'inde (3630 satır) fit edilir, split_v1 test'inde (908 satır) tahmin edilir.
- **Bileşenler:** `split_v2.json` içindeki `components` alanı (TF-IDF kosinüsü > 0.8 grafının
  bağlı bileşenleri). v1 ve v2 aynı temiz kümeyi böler; kod bunu denetler.
- **Sızmış satır:** bileşeninde en az bir v1 train satırı olan v1 test satırı. Zincirler de
  (test–test–train) sayılır.
- **Birincil teşhis büyüklüğü:** accuracy(sızmış) − accuracy(temiz).
  - Belirsizlik: %95 yüzdelik **grup bootstrap** CI. Bütün v1 test bileşenleri ortak
    w ~ Multinomial(G, 1/G) ile örneklenir (B = 10000, seed 42; `paired_test.bootstrap_weights`).
  - Bir bileşen iki alt kümeye bölünmez. Paydası 0 olan replikatlar düşer ve sayısı raporlanır.
  - p hesaplanmaz.
- **macro-F1 farkı verilmez.** Sızmış alt küme yaklaşık 100 satırdır ve 7 sınıfa dağılır. Bunun
  yerine alt küme başına n, grup sayısı, accuracy ve sınıf sayıları verilir.
- **Yan yana:** v1 test genel macro-F1 ve accuracy, aynı w'lerden alınan grup bootstrap CI'larıyla
  verilir. Yanına v2 test char_lr macro-F1'i ve CI'ı konur.
  - Test kümeleri farklıdır: kısmen örtüşürler ve eğitim bileşimleri de farklıdır.
  - char_lr'nin ayarları split_v2 train'inde seçildi ve bu train v1 test satırlarının çoğunu
    içerir. Bu yüzden v1 genel skoru tamamen görülmemiş veride ayarlanmış bir modelin skoru
    değildir (grid dar olduğu için etki küçük beklenir).
  - p yok. v1 − v2 farkı sızıntı etkisi olarak **sunulmaz**.
- **Sınırlılık:**
  - Sızmış alt küme rastgele değildir; konu ve sınıf bileşimi farklı olabilir. Bu yüzden fark
    yalnızca ezberlemeye atfedilemez.
  - "Temiz" alt küme sızıntıdan arınmış sayılmaz; benzerliği ≤ 0.8 olan yeniden yazımlar
    bulunabilir.
- Çıktı: `phase5_v1_leakage_<ts>.json` (`"split": "split_v1"`) ve write-once
  `test_v1_char_lr.npz` (`leaked` ve `groups` dahil).

## Commit düzeni (`RULES['commits']`)
1. Faz 5a: aile kararı (`27e3513`).
2. Faz 5b: değerlendirme kodu, testler, şablonlar ve kurallar.
3. **Test öncesi son commit:** temiz commit'te üretilen `phase5_model_check` json'u.
4. Test koşusu (`predict` + `evaluate`): tahmin npz'leri, manifest ve sonuç json'u **yorumsuz,
   ayrı bir commit**.
5. Yorum ve belge güncellemeleri: sonraki commit.
6. split_v1 teşhisi: kendi koşusu ve commit'i.

## Süre tahmini (yerel CPU)
Ölçüm: BERT için 32 train metninde 0.21 sn/metin. bge-m3 için Faz 3 ölçümü 0.50 sn/metin.

| Adım | Süre |
|---|---|
| `verify` (3 × 3630 metin) | ≈ 40 dk |
| `predict`: BERT (3 × 908 metin) | ≈ 10 dk |
| `predict`: bge-m3 test embedding'i | ≈ 11 dk |
| `predict`: klasik fit'ler | birkaç dk |
| `evaluate` | < 1 dk |

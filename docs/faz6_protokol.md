# Faz 6 protokolü: olasılık kalibrasyonu

**Kaynak:** `src/ttc4900/calibration.py` (`CALIB_RULES`, `CALIB_MODELS`, `SENTENCES`).
Komutlar: `python -m ttc4900.calibration train` (6b), `python -m ttc4900.calibration test` (6c). Bu belge ile kaynak çelişirse kaynak geçerlidir. Hepsi 6a commit'inde, **hiçbir
kalibrasyon sayısı hesaplanmadan** dondurulur ve sonuçlara göre değiştirilmez.

## Soru
Modellerin verdiği olasılıklar, doğru olma sıklığıyla uyuşuyor mu? Uyuşmuyorsa, tek parametreli
bir düzeltme (temperature scaling) bunu train içinde ve testte iyileştiriyor mu?

**İki model, iki ayrı soru.** Bu faz bir model karşılaştırması değildir. Faz 5'in sınıflandırma sonuçları (accuracy,
macro-F1, McNemar) değişmez, çünkü temperature scaling argmax'ı korur. Kod bunu her model için
denetler.

## Modeller (`CALIB_MODELS`)
| Model | Rol | Logit kaynağı |
|---|---|---|
| `berturk_ft` (seed 42) | **birincil** | BERT logit'leri |
| `char_lr` | **birincil** | log(predict_proba) |
| `berturk_ft_s43`, `berturk_ft_s44` | betimsel (karar yok) | BERT logit'leri |

- word_lr, word_svc, word_cnb, char_lr_c1clean ve bge_m3 kapsam dışıdır.

## Yöntem: temperature scaling
- **Sezgi:** aşırı emin bir model logit'lerini büyük üretir. Bütün logit'leri tek bir T > 0'a
  bölmek olasılıkları yumuşatır (T > 1) ya da sertleştirir (T < 1). Sıralama değişmediği için
  tahmin edilen sınıf da değişmez.
- **Formül:** p̂(k | x) = softmax(z(x) / T)_k. T, kalibrasyon kümesinde ortalama negatif
  log-olabilirliği (NLL) minimize eder: T̂ = argmin_T −(1/n) Σᵢ log p̂(yᵢ | xᵢ).
- **Kod:**
  - `scipy.optimize.minimize_scalar`, log T üzerinde, sınırlar [log 0.05, log 20], `bounded`
    yöntem, xatol 1e-6.
  - Optimum bir sınıra düşerse (log T'de 1e-3 içinde) sonuç kaydedilir ve o model için karar
    "değerlendirilemez" olur. 6b'de bu, final T'yi ve beş cross-fit T'sini kapsar.
- **char_lr'nin logit'i:**
  - Çok sınıflı LR'de log p_k = z_k − logsumexp(z). Bütün sınıflardan aynı sabit çıktığı için
    softmax(log p / T) = softmax(z / T).
  - Bu yüzden test npz'sindeki `predict_proba`'nın log'u kullanılır.
  - Alt taşmayla p = 0 çıkarsa `np.finfo(float).tiny`'ye kırpılır. Kırpılan hücre sayısı
    raporlanır.

## Metrikler
| Metrik | Tanım | Rol |
|---|---|---|
| **NLL** | −(1/n) Σ log p̂(yᵢ) | **birincil** (T'nin optimize ettiği uygun skor kuralı) |
| Brier | (1/n) Σᵢ Σ_k (p̂_k − 1[yᵢ = k])² | ikincil, betimsel |
| ECE (top-label) | Σ_m (\|B_m\| / n) · \|acc(B_m) − conf(B_m)\|; conf = max_k p̂_k; 15 eşit genişlikte bin (0, 1/15], … | ikincil, betimsel |
| ECE, 10 ve 20 bin | aynı tanım | bin duyarlılığı, betimsel |
| Reliability diyagramı | 15 bin; bin başına conf ve acc, satır sayısıyla | figür |

- **ΔNLL = NLL(önce, T = 1) − NLL(sonra); pozitif = iyileşme.** Bütün metriklerde Δ = önce −
  sonra alınır (hepsinde küçük = iyi).
- ECE yanlı bir kestiricidir ve bin seçimine duyarlıdır. Karar yalnızca NLL'ye dayanır.

## 6b: train içinde (ana kanıt)
Yalnızca train OOF tahminleri kullanılır. Test npz'leri okunmaz, `load_train_test` çağrılmaz.
Bunu bir AST testi denetler.

### OOF logit'leri
- **BERT:** Faz 4 CV koşularının dış fold logit'leri.
  - seed 42: `cv_phase4_berturk_ft_20261015-214140.npz` (3630 satır).
  - seed 43/44: kendi 5 fold npz'leri birleştirilir.
  - Logit'ler early stopping'in kullandığı iç validation'dan değil, dış fold'dan gelir. Bu
    yüzden T fit'ine seçim sızıntısı girmez.
- **char_lr:** OOF olasılıkları yeniden üretilir.
  - Dış fold'lar char_lr npz'sindeki `fold_id[0]`'dan okunur (BERT ile aynı fold'lar). `make_cv`
    çalıştırılmaz.
  - Her fold'da donmuş `final.best_params` (C=100, char_wb (2,4)) ile `load_train()` verisinde
    fit edilir; ayar yapılmaz.
  - Kaydedilen değer `decision_function`'dır.
  - Sınırlılık: bu parametreler tüm train'de seçildi, held-out fold da buna dahildi. Grid dar
    olduğu için etkinin küçük olması beklenir.

### Birincil 6b büyüklüğü: cross-fit ΔNLL
- Her dış fold j için T̂₋ⱼ, diğer 4 fold'un OOF logit'lerinde fit edilir ve fold j'ye uygulanır.
  Böylece her satırın kalibre olasılığı, o satırı görmemiş bir T ile hesaplanır.
- ΔNLL = NLL(önce, T = 1) − NLL(sonra, cross-fit T). Pozitif = iyileşme.
- **CI:** train grupları (split_v2 `components`, `make_cv` ile aynı) üzerinde grup bootstrap,
  B = 10000, `np.random.default_rng(42)`, yüzdelik %95.
  - Bootstrap T̂₋ⱼ'leri sabit tutar. Yani CI, T'nin kestirim belirsizliğini içermez. Bu belirsizlik
    ayrıca raporlanır.
- **Karar (her birincil model için ayrı):**
  - *"Train içinde iyileşme gösterildi"* ⇔ ΔNLL'nin %95 CI'ı tamamen 0'ın üstünde.
  - Değilse *"train içinde iyileşme gösterilemedi"*.
  - p hesaplanmaz. İki model ayrı ayrı değerlendirilir; CI'lar marjinaldir.
- Brier ve ECE (15/10/20 bin) aynı cross-fit olasılıklarla, önce/sonra ve CI'larıyla betimsel
  verilir.

### Final T (teste uygulanacak)
- T̂_final, modelin tüm 3630 OOF satırında fit edilir.
- Belirsizliği için train grupları üzerinde grup bootstrap uygulanır: her replikatta T yeniden
  fit edilir, B = 10000, seed 42, %95 yüzdelik CI.
- T̂_final 6b json'una yazılır. 6c yalnızca bu değeri okur.

### T'nin 15 CV koşusundaki dağılımı (betimsel, BERT)
- 15 koşunun (5 fold × 3 seed) her birinde T, yalnızca o koşunun kendi 726 OOF satırında ayrı
  fit edilir.
- Raporlananlar:
  - 5 × 3 tablo.
  - Genel ortalama, std (ddof=1), min ve max.
  - Seed başına (fold'lar üzerinden) ve fold başına (seed'ler üzerinden) ortalama ve std.
  - Her koşunun best epoch'u yan yana verilir (Faz 4 json'larından).
- Amaç, T'nin eğitim koşusuna ne kadar duyarlı olduğunu göstermektir; uyuşmazlık sınırlılığı
  için bağlam sağlar. Karar kuralına girmez.
- Bu T'ler kendi verisinde fit ve ölçüldüğü için in-sample'dır; performans iddiası için
  kullanılmaz.
- char_lr için 5 fold'un T'si aynı biçimde betimsel verilir (tek tekrar).

## 6c: test (teyit, tek sefer)
- **Ön koşullar:**
  - Çalışma ağacı temiz.
  - 6a ve 6b commit'lenmiş.
  - 6b json'u izleniyor. sha256'sı kaydedilir.
- **Girdi:** yalnızca `outputs/metrics/preds/test_<model>.npz` içindeki `scores` okunur. Test
  metni okunmaz, model yeniden koşulmaz.
  - npz sha256'ları Faz 5 sonuç json'undakiyle aynı olmalıdır.
- **Uygulama:** T̂_final değiştirilmeden uygulanır. Testte hiçbir şey fit edilmez.
- **Birincil test büyüklüğü:** ΔNLL_test = NLL(önce, T = 1) − NLL(sonra, T̂_final); pozitif =
  iyileşme.
  - CI: test grupları üzerinde grup bootstrap, **Faz 5'teki aynı ağırlıklarla** (B = 10000,
    seed 42, `test_groups`, `paired_test.bootstrap_weights`).
  - Kod, accuracy CI'larının Faz 5 sonuç json'undakiyle birebir aynı çıktığını denetler.
    Çıkmazsa durur.
  - Kayıtlı skorların argmax'ı Faz 5'in `y_pred`'iyle aynı olmalıdır.
- **Karar (teyit):**
  - *"Testte teyit edildi"* ⇔ ΔNLL_test'in %95 CI'ı tamamen 0'ın üstünde.
  - Değilse *"testte teyit edilemedi"*.
  - Ana kanıt 6b'dir. Test sonucu 6b'nin kararını değiştirmez; iki sonuç birlikte raporlanır.
- **Betimsel:**
  - Brier ve ECE (15/10/20 bin), önce/sonra ve CI'larıyla.
  - Reliability diyagramları.
  - Seed 43/44 aynı biçimde, karar verilmeden raporlanır.
  - Modeller arası farklar: aşağıdaki "Modeller arası" bölümü.
- **Kayıt:** testin bu kullanımı tek seferliktir.
  - Çıktı: `phase6_test_calibration_<ts>.json`. Hedef dosya varsa üzerine yazılmaz.
  - Yorumsuz ayrı bir commit'e girer.

## Modeller arası (`CALIB_RULES['between_models']`, yalnız betimsel)
- `berturk_ft − char_lr` için 6b'de ve 6c'de şunlar verilir:
  - Kalibre NLL, Brier ve ECE farkı.
  - ΔNLL farkı.
  - 6b'de ayrıca T̂_final farkı.
- Hepsi aynı w'lerden fark ve %95 CI ile verilir. p, karar ve "X, Y'den daha iyi kalibre"
  iddiası yoktur.

## Sınırlılıklar (önceden yazıldı)
- **Model uyuşmazlığı (BERT):**
  - T, CV modellerinin OOF logit'lerinden öğrenilir. Bu modeller ~2323 satırla, early stopping
    ile eğitildi; best epoch 1–5.
  - T ise farklı bir modele uygulanır: final model 3630 satırla, sabit 4 epoch eğitildi ve
    validation yok.
  - Final modelin güven düzeyi farklı olabilir; yönü bilinmiyor. Yeniden eğitim yapılmaz.
  - 15 koşunun T dağılımı, bu duyarlılığın ölçeği için bağlamdır. 6c, uyuşmazlığa rağmen T'nin
    teste taşınıp taşınmadığını gösterir.
- **char_lr:** OOF modelleri 2904 satırla, final model 3630 satırla eğitildi. Parametreler tüm
  train'de seçildi.
- **Sayısal ortam:** OOF logit'leri Colab'da fp16 ile (float32 olarak kaydedildi), test
  logit'leri CPU'da fp32 ile üretildi.
- **Bootstrap:** CI'lar sabit T'ye koşulludur. T belirsizliği ayrı verilir.
- **Dağılım:** kalibrasyon yalnızca bu veri dağılımı için geçerlidir (tek derleme,
  kısayol artefaktları; `docs/shortcut_learning.md`). Başka kaynaklardan gelen metinlerde (demoya
  yazılan metinler dahil) olasılıklar kalibre olmayabilir.

## Demo ve belge dili (`CALIB_RULES['demo_language']`)
- Demoda BERTurk'ün kullanılması bir dağıtım tercihidir. Gerekçe, Faz 5'te önceden belirlenmiş
  birincil model olmasıdır. Testte char_lr'den ayırt edilemedi; "char_lr'den iyi" iddiası
  yapılmaz.
- Gösterilen olasılıklar için "bu veri dağılımında kalibre edildi" yazılır. Kısayol notları
  (bozuk tırnaklar, ajans künyeleri, dönem adları) demo açıklamasında yer alır.

## Cümle şablonları (`SENTENCES`)
| Durum | Şablonun özü |
|---|---|
| 6b gösterildi | "{model}: train içinde cross-fit ΔNLL +x [lo, hi]; iyileşme gösterildi. T̂_final = t [lo, hi]." |
| 6b gösterilemedi | "{model}: ΔNLL x [lo, hi], CI 0'ı içeriyor ya da 0'ın altında; iyileşme gösterilemedi. Bu, modelin kalibre olduğunu kanıtlamaz." |
| değerlendirilemez | "{model}: T optimizasyonu sınıra düştü (T = t); karar verilmedi." |
| 6c teyit | "{model}: testte ΔNLL +x [lo, hi]; teyit edildi (ana kanıt 6b)." |
| 6c teyit edilemedi | "{model}: testte ΔNLL x [lo, hi]; teyit edilemedi; olası nedenler: model uyuşmazlığı, düşük güç. Bu, testte etki olmadığı anlamına gelmez. 6b kararı değişmez." |
| betimsel | "{a} − {b} (betimsel, p yok): metrik, fark, CI." |

## Değişiklik notu (6b'den sonra, 6c'den önce; `CALIB_RULES['change_log']`)
6b sonuçları görüldükten sonra, test açılmadan, kullanıcı onayıyla üç değişiklik yapıldı:
1. **"Teyit edilemedi" şablonu.** Eski şablon nedeni önceden atfediyordu ("T farklı modellerden
   öğrenildi"). Yeni şablon "olası nedenler: model uyuşmazlığı, düşük güç" diyor ve "etki yok"
   okumasını açıkça dışlıyor. Karar mantığı (`decide`) değişmedi.
2. **Yorum kuralı** (`interpretation`): "teyit edilemedi" etki olmadığı anlamına gelmez. Test
   908 satır, train'in dörtte biri; CI'lar daha geniş.
3. **Fold başına cross-fit ΔNLL** (`python -m ttc4900.calibration folds`): **sonradan,
   betimsel**, hiçbir karara girmez.
   - Her fold'a 6b'nin cross-fit T'si uygulanır. Kod, T'leri 6b json'uyla denetler.
   - CI, fold'un grupları üzerinde grup bootstrap'iyle hesaplanır.
   - Fold'un kendi in-sample T'si yan yana verilir.
   - Fold ΔNLL'lerinin n ağırlıklı ortalaması 6b toplamına eşit olmalıdır; kod bunu da
     denetler.

## Commit düzeni
1. **6a:** `CALIB_RULES`, `calibration.py` kodu, testler ve bu belge. Hiçbir kalibrasyon sayısı
   hesaplanmamış olur.
   - Testler: sentetik veride T'nin bilinen değeri bulması, argmax'ın korunması, cross-fit'te
     held-out fold'un kendi T fit'ine girmemesi, ECE'nin elle hesaplanmış bir örnekle tutması,
     fit yolunda test erişimi olmaması (AST), 6c ön koşulları.
2. **6b:** temiz commit'te koşu; json, char_lr OOF skor npz'si (write-once) ve reliability
   figürü, yorumsuz.
3. **6c:** temiz commit'te test koşusu, json ve figürler, yorumsuz.
4. **Yorum:** `docs/faz6_sonuclar.md` ve README güncellemesi, stats-reviewer ve
   leakage-reviewer denetimiyle.

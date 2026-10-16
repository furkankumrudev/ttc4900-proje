# Faz 6 sonuçları: olasılık kalibrasyonu

**Kaynaklar** (hepsi `outputs/metrics/`):
- 6b, train içi (ana kanıt): `phase6_train_calibration_20261016-224639.json`.
- Fold dökümü (sonradan, betimsel): `phase6_train_folds_20261016-230928.json`.
- 6c, test (teyit, tek sefer): `phase6_test_calibration_20261016-230938.json`.
- Figürler: `outputs/figures/phase6_train_reliability.png`, `phase6_test_reliability.png`.
- Protokol: `docs/faz6_protokol.md`, `src/ttc4900/calibration.py` (`CALIB_RULES`).

Bu belgedeki sayılar bu json'lardan alındı. Yorum, sonuç json'ları commit'lendikten sonra ve
ayrı bir commit'te yazıldı.

## Sıra
| Commit | Ne | Test açık mı |
|---|---|---|
| `a1bf407` | 6a: kurallar, kod, testler (hiçbir kalibrasyon sayısı hesaplanmadan) | hayır |
| `a668dea` | 6b sonuçları, yorumsuz | hayır |
| `d4098b5` | 6b sonuçları görüldükten sonra: "teyit edilemedi" şablonu nedeni atfetmiyor; fold dökümü kodu eklendi (karar mantığı değişmedi) | hayır |
| `27f2b68` | Fold dökümü sonuçları, yorumsuz | hayır |
| `2b157f8` | 6c test sonuçları, yorumsuz | **evet** (test skorları, bir kez) |

6c'de testte hiçbir şey fit edilmedi; T̂_final 6b json'undan okundu. Kod iki şeyi denetledi:
bootstrap ağırlıklarının Faz 5'tekilerle aynı olduğunu ve kayıtlı test skorlarının argmax'ının
Faz 5 tahminleriyle aynı olduğunu. Temperature scaling argmax'ı korur, bu yüzden Faz 5'in
accuracy, macro-F1 ve McNemar sonuçları değişmez.

## Yöntem (kısaca)
- **Sezgi:** aşırı emin bir model büyük logit üretir. Bütün logit'leri tek bir T > 0'a bölmek
  olasılıkları yumuşatır (T > 1) ya da sertleştirir (T < 1).
- **Formül:** p̂ = softmax(z / T). T, train OOF tahminlerinde ortalama NLL'yi minimize ederek
  bulunur.
- **Büyüklük:** ΔNLL = NLL(önce, T = 1) − NLL(sonra). Pozitif değer iyileşme demektir.
- **Karar:**
  - 6b'de "gösterildi" ⇔ cross-fit ΔNLL'nin %95 grup bootstrap CI'ı tamamen 0'ın üstünde.
  - 6c'de "teyit edildi" ⇔ test ΔNLL'sinin CI'ı tamamen 0'ın üstünde (Faz 5 ile aynı
    ağırlıklar).
- **Kapsam:** iki birincil model iki ayrı sorudur. Modeller arası farklar yalnızca betimseldir.
- **CI'lar** T sabit tutularak hesaplandı. T'nin kendi belirsizliği ayrıca verildi.

## Ana sonuç
| Model | T̂_final [%95 CI] | 6b: cross-fit ΔNLL [%95 CI] | 6b karar | 6c: test ΔNLL [%95 CI] | 6c karar |
|---|---|---|---|---|---|
| **berturk_ft** (seed 42) | 1.283 [1.239, 1.325] | +0.0200 [+0.0123, +0.0282] | **gösterildi** | +0.0509 [+0.0329, +0.0707] | **teyit edildi** |
| **char_lr** | 0.916 [0.868, 0.964] | +0.0014 [−0.0008, +0.0035] | **gösterilemedi** | −0.0016 [−0.0062, +0.0025] | **teyit edilemedi** |

### berturk_ft: gösterildi ve testte teyit edildi
**Kapsam:** "teyit edildi" bu tek eğitilmiş BERTurk koşusu (seed 42) ve bu test seti için
geçerlidir.
- Seed 43'te train içi kazanç gösterilemedi.
- 15 CV koşusunda T'nin std'si 0.25. Farklı bir eğitim koşusu için T kayabilir.
- Aşağıdaki fold dökümünde üç (koşu, fold) biriminde kalibrasyon NLL'yi kötüleştirdi.

- **T̂_final = 1.283 > 1:** OOF tahminlerinde model aşırı emindi. Logit'leri 1.28'e bölmek
  olasılıkları yumuşatır.
- **Train içinde (6b):** cross-fit NLL 0.3227'den 0.3027'ye indi.
- **Testte (6c, önceden kaydedilmiş, tek sefer):** NLL 0.3380'den 0.2870'e indi. Karar kuralı
  bunu "teyit edildi" olarak sınıflandırdı. T = 1.28 ile ölçekleme, testte aşırı güveni azalttı.

| Betimsel metrik | 6b önce → sonra | 6c önce → sonra | 6c Δ [%95 CI] |
|---|---|---|---|
| Brier | 0.1388 → 0.1340 | 0.1272 → 0.1231 | +0.0041 [+0.0023, +0.0061] |
| ECE (15 bin) | 0.048 → 0.015 | 0.058 → 0.041 | +0.016 [+0.010, +0.020] |

- **Aşırı güven azaldı ama tamamen gitmedi.** Testte kalibrasyondan sonra ECE 0.041.
  - En kalabalık güven bin'inde (828 satır) ortalama güven 0.984, accuracy 0.954. Kalibrasyondan
    önce aynı bin'de 0.996'ya karşı 0.945 idi.
  - Kalan ECE büyük ölçüde bu bin'den geliyor: ECE'ye katkısı 828/908 × 0.030 ≈ 0.027, yani
    0.041'in yaklaşık üçte ikisi.
- **ECE uyarısı:** ECE sonlu örneklemde yukarı yanlıdır ve bin seçimine duyarlıdır. Testte 10,
  15 ve 20 bin için sonra-değerleri 0.037, 0.041 ve 0.043. Karar yalnızca NLL'ye dayanır.

### char_lr: gösterilemedi, testte teyit edilemedi
- "Teyit edilemedi" etki olmadığı anlamına gelmez. Testte 908 satır var, train'in dörtte biri;
  CI'lar daha geniş. Olası nedenler: model uyuşmazlığı ve düşük güç. Ana kanıt 6b'dir ve 6b de
  bir iyileşme gösteremedi.
- CI'ların üst sınırları da bilgi taşıyor: +0.0035 (train) ve +0.0025 (test). Bu sınırlar
  BERTurk'ün kazancının (+0.020 / +0.051) çok altında, yani büyük bir kazanç beklenmiyor.
- **Başlangıçta zaten iyi kalibreydi:**
  - T = 1'de ECE küçüktü: train OOF'ta 0.020 [0.015, 0.030], testte 0.025 [0.021, 0.046].
    ECE yanlı bir kestirici olduğu için, iyi kalibre bir modelde bile sıfırdan uzak çıkması
    beklenir.
  - T̂ = 0.916 hafif bir çekingenlik gösteriyor; CI [0.868, 0.964] 1'i dışlıyor. Bu küçük
    sapma NLL'de ayırt edilebilir bir kazanca dönüşmüyor.
  - T'nin 1'den farklı olup olmadığı ile NLL'nin iyileşip iyileşmediği farklı sorulardır;
    ΔNLL kuralı daha muhafazakârdır.
- **Betimsel:** testte Brier ve ECE değişimlerinin CI'ları 0'ı içeriyor.
  - Brier: −0.0000 [−0.0012, +0.0011].
  - ECE değişimi bin sayısına göre işaret değiştiriyor: 15 bin +0.004, 10 bin −0.007, 20 bin
    +0.001. Hepsinin CI'ı 0'ı içeriyor.
  - 6b'de üç bin sayısında da ECE hafifçe arttı; CI'lar 0'ı içeriyor.

## Modeller arası (yalnız betimsel; sıralama yok)
İki model iki ayrı sorudur. Aşağıdaki farklar p'siz, karar kuralı dışında verildi.
"Daha iyi kalibre" iddiası yapılmaz.

| berturk_ft − char_lr | 6b fark [%95 CI] | 6c fark [%95 CI] |
|---|---|---|
| NLL (sonra) | +0.0072 [−0.0162, +0.0310] | +0.0035 [−0.0495, +0.0575] |
| Brier (sonra) | −0.0109 [−0.0222, +0.0006] | −0.0097 [−0.0320, +0.0132] |
| ECE15 (sonra) | −0.0083 [−0.0178, +0.0013] | +0.0205 [−0.0021, +0.0307] |
| ΔNLL (kalibrasyon kazancı) | +0.0185 [+0.0096, +0.0282] | +0.0526 [+0.0322, +0.0748] |

- Kalibrasyondan sonraki metriklerin hiçbirinde fark ayırt edilemedi.
- BERT'in kalibrasyon kazancı daha büyük. Bu, BERT'in T̂'sinin 1'den daha uzak olmasıyla
  (başlangıçta daha aşırı emin) tutarlıdır. Kalibrasyondan sonra daha iyi olduğu anlamına
  gelmez.
- **Doğruluk ile olasılık güvenilirliği farklı özelliklerdir (betimsel):**
  - T = 1'de test NLL'si: char_lr 0.282, BERT 0.338. Bu fark için CI hesaplanmadı.
  - Accuracy nokta tahminleri ters sırada: char_lr 0.913, BERT 0.927. Faz 5'te bu fark ayırt
    edilemedi.
  - Bu sayılar sıralama için değil, iki özelliğin ayrı ölçüldüğünü göstermek için verildi. İki
    modelin ne accuracy'si ne de kalibrasyonu birbirinden ayırt edildi.
- T değerleri modeller arasında karşılaştırılamaz: BERT logit'i ile LR'nin decision function'ı
  farklı ölçeklerdedir. Anlamlı olan yalnızca T'nin 1'e göre yönüdür: BERT'te > 1 (aşırı emin),
  char_lr'de < 1 (hafif çekingen).

## Seed'ler (betimsel)
| Seed | T̂ | 6b cross-fit ΔNLL [%95 CI] | 6c test ΔNLL [%95 CI] |
|---|---|---|---|
| 42 (birincil) | 1.283 | +0.0200 [+0.0123, +0.0282] | +0.0509 [+0.0329, +0.0707] |
| 43 | 1.210 | −0.0025 [−0.0081, +0.0036] | +0.0386 [+0.0253, +0.0529] |
| 44 | 1.296 | +0.0125 [+0.0042, +0.0212] | +0.0484 [+0.0309, +0.0680] |

- Karar yalnızca seed 42 içindir. Seed 43 ve 44 için karar verilmedi.
- Train içi kazanç seed'e göre değişiyor: s43'te CI 0'ı içeriyor ve nokta tahmini negatif.
  Testte s43 ve s44'ün ΔNLL'si de pozitif.
- **Bu satırlar bağımsız tekrar değildir.** Aynı test satırları ve aynı train fold'ları
  kullanıldı; bu yüzden sayılmaz ve birleştirilmez.

## Fold heterojenliği (sonradan, betimsel)
Fold dökümü 6b sonuçları görüldükten sonra kullanıcı onayıyla eklendi ve karara girmez. Her
fold'a 6b'nin cross-fit T'si, yani diğer 4 fold'da fit edilen T, uygulandı.

| Koşu | Kendi in-sample T'si | Uygulanan cross-fit T | Best epoch | Fold ΔNLL [%95 CI] |
|---|---|---|---|---|
| s42, fold 1 | 1.00 | 1.36 | 2 | −0.038 [−0.051, −0.025] |
| s43, fold 1 | 0.75 | 1.36 | 1 | −0.109 [−0.119, −0.099] |
| s44, fold 3 | 0.86 | 1.45 | 1 | −0.112 [−0.126, −0.096] |

- **Negatif fold'lar:** üç seed'in her birinde bir fold'un ΔNLL'si negatif. Bu fold'ların
  ortak özelliği, koşunun erken durması ve kendi T'sinin 1 civarında ya da altında olması. Bu
  koşular aşırı emin değildi; diğer fold'lardan öğrenilen T > 1 onları gereğinden fazla
  yumuşattı.
- **Diğer fold'lar:** ΔNLL pozitif (+0.020 ile +0.062 arası). Tek istisna s43 fold 3: +0.003,
  CI 0'ı içeriyor.
- **s43'ün negatif 6b toplamı** tek bir fold'dan (fold 1, −0.109) geliyor.
- **char_lr:** fold ΔNLL'leri −0.003 ile +0.004 arasında. Kendi T'leri 0.85–0.99.

## Sonradan gözlemlenen hipotezler ve gelecek çalışma
Aşağıdakiler 6b ve 6c sonuçları görüldükten sonra fark edildi. Önceden kaydedilmedi ve test
edilmedi. **Testte başka bir T denenmedi**; T̂_final dondurulmuş kurala göre kullanıldı.

**1. T, best epoch ile artıyor görünüyor (15 BERT CV koşusu).**

| Best epoch | Koşu | T ortalaması | Aralık |
|---|---|---|---|
| 1 | 2 | 0.81 | 0.75–0.86 |
| 2 | 2 | 1.06 | 1.00–1.13 |
| 3 | 3 | 1.31 | 1.26–1.34 |
| 4 | 5 | 1.43 | 1.31–1.55 |
| 5 | 3 | 1.49 | 1.41–1.58 |

- Sezgisi: daha uzun eğitilen model train verisine daha çok uyar ve daha emin hale gelir.
- Best epoch fold ile kısmen karışıktır: epoch ≤ 2 olan dört koşunun hepsi fold 1 ve 3'ten.
  Yine de her fold içinde, üç seed boyunca T best epoch ile birlikte artıyor. Örneğin fold
  3'te epoch 1, 2, 3 için T = 0.86, 1.13, 1.34.
- 15 koşu bağımsız değildir (fold ve seed paylaşılıyor). Bu yüzden korelasyon ya da p
  hesaplanmadı.

**2. T̂_final final modeli eksik düzeltiyor olabilir.**
- Final model sabit 4 epoch eğitildi. Best epoch'u 4 olan koşuların T ortalaması 1.43,
  kullanılan T̂_final ise 1.283. T̂_final, farklı epoch'larda durmuş koşuların birleşik
  OOF'undan öğrenildi.
- Bu hipotezle çelişmeyen ama onu ayırt etmeyen gözlemler:
  - T = 1'de ECE testte 0.058 [0.044, 0.076], train OOF'ta 0.048 [0.041, 0.058]. CI'lar
    örtüşüyor. ECE'nin yanlılığı küçük örneklemde (908 satır) daha büyüktür.
  - Kalibrasyondan sonra en kalabalık bin'de güven ile accuracy: testte 0.984 ve 0.954 (fark
    0.030), train OOF'ta 0.972 ve 0.966 (fark 0.006).
  - Bunlar tek test setinde, tek modelde ve sonradan yapılmış gözlemlerdir; kanıt sayılmaz.
    Tek bir skaler T bin düzeyindeki sapmaları zaten düzeltemez.
- Ama final model farklı bir veri boyutuyla (3630 satır) ve farklı bir lr takvimiyle eğitildi.
  Önceden yazılan sınırlılık geçerliliğini koruyor: final modelin güven düzeyindeki farkın
  yönü önceden bilinmiyordu.

**Gelecek çalışma önerisi** (bu projede yapılmadı):
- Final modelle aynı epoch sayısında eğitilmiş modellerin OOF logit'lerinden T öğrenmek.
- Ya da final model eğitilirken train'den, `make_cv` fold'larından biriyle bir kalibrasyon
  parçası ayırmak.
- Bu soru, bu test seti yeniden kullanılmadan, yeni bir değerlendirme verisiyle sınanmalıdır.

## Demo notu (Gradio)
- Demo, seed 42 BERTurk modelini kullanacak ve olasılıkları T = 1.283 ile kalibre edilmiş
  olarak gösterecek.
- BERTurk'ü seçmek bir dağıtım tercihidir. Gerekçe, Faz 5'te önceden belirlenmiş birincil model
  olmasıdır. Testte char_lr'den ayırt edilemedi; "char_lr'den iyi" iddiası yapılmaz.
- Demo açıklamasında yer alacak sınırlılıklar:
  - **Kalan kalibrasyon hatası:** testte kalibrasyondan sonra ECE 0.041. En yüksek güven
    bandında model hâlâ biraz aşırı emin (güven 0.98, accuracy 0.95).
  - **Veri dağılımı:** kalibrasyon yalnızca bu veri dağılımı için geçerli (tek kaynak, tek
    dönem). Demoya yazılan metinler farklı bir dağılımdan gelebilir; olasılıklar orada kalibre
    olmayabilir.
  - **Kısayollar:** bozuk tırnaklar, ajans künyeleri ve dönem adları tahmini etkileyebilir
    (`docs/shortcut_learning.md`).

## Sınırlılıklar
- **Model uyuşmazlığı (BERT):** T, ~2323 satırla early stopping ile eğitilen CV modellerinden
  öğrenildi ve 3630 satırla, sabit 4 epoch eğitilen final modele uygulandı.
- **T'nin CI'ı yalnız veri örneklemesi belirsizliğidir.** Aynı OOF modelleri için hesaplandı.
  Eğitim koşusundan gelen belirsizlik çok daha büyük: 15 koşunun T std'si 0.25.
- **Cross-fit tamamen temiz değil.** Bir satırın T'si o satırı değerlendirmeden fit edildi.
  Ama diğer fold'ların modelleri bu satırla eğitilmiş olabilir. Skaler bir parametre için etkinin
  küçük olması beklenir.
- **Diğer:**
  - OOF logit'leri Colab'da fp16 ile, test logit'leri CPU'da fp32 ile üretildi.
  - Top-label ECE yalnızca en yüksek sınıfın güvenini ölçer.
  - Tek parametreli T, sınıfa ya da güven bandına göre değişen sapmaları düzeltemez.

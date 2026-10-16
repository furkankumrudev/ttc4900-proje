# Faz 5 sonuçları: final test değerlendirmesi

**Kaynaklar** (hepsi `outputs/metrics/`):
- Ana sonuçlar: `phase5_test_20261016-172356.json`. Tahmin manifesti:
  `phase5_test_preds_20261016-171044.json`.
- split_v1 teşhisi: `phase5_v1_leakage_20261016-220137.json`.
- Keşifsel (sonradan) çift: `phase5_exploratory_20261016-220512.json`.
- Test öncesi güç analizi: `test_power_20261015-215451.json`.
- CV sayıları: `phase2_summary_20261012-080148.json`, `phase2_char_lr_c1clean_20261012-075141.json`,
  `phase3_summary_20261012-095539.json`, `phase4_summary_20261015-214140.json` (ayrıntı:
  `docs/faz4_sonuclar.md`).
- Protokoller: `docs/test_protokol.md` (yöntem, karar kuralı, güç), `docs/faz5_protokol.md`
  (adımlar, write-once kayıtlar).

Bu belgedeki bütün sayılar bu json'lardan alındı. Yorum, sonuç json'ları commit'lendikten sonra
ve ayrı bir commit'te yazıldı.

## Sıra: neler testten önce sabitlendi
| Commit | Ne | Test açık mı |
|---|---|---|
| `1587e1d` | Birincil çiftin güç analizi (gerçek BERT OOF'u) | hayır |
| `27e3513` | Aileler kesin: birincil m=1, ikincil m=4, betimsel çiftler | hayır |
| `30d9019` | Değerlendirme kodu, karar kuralı, sonuç cümlesi şablonları | hayır |
| `b50f134` | Model doğrulaması (sha256 + in-sample F1), test öncesi son commit | hayır |
| `5c5ef48` | Test tahminleri ve sonuç json'u, yorumsuz | **açıldı** (bir kez) |
| `7a428c6` | CLI çıktısı UTF-8 (yalnız konsol yazdırma; sonuçları etkilemez) | test sonrası |
| `c3a521e` | split_v1 teşhisi, yorumsuz | test sonrası |
| `9774c34` | Keşifsel çift adımının kodu (çift burada, sonuçlar görüldükten sonra tanımlandı) | test sonrası |
| `0fac774` | Keşifsel çift sonucu, yorumsuz | test sonrası |

Güç analizi, test tahminlerinden yaklaşık 19 saat önce commit'lendi. Bu yüzden aşağıdaki güç ve
MDE değerleri test sonucuna bakılarak seçilmiş olamaz.

## Özet
- **Birincil karşılaştırma (BERTurk fine-tuning − char_lr): ayırt edilemedi.**
  - Fark +0.0159 macro-F1, %95 grup bootstrap CI [−0.0004, +0.0328].
  - Exact McNemar n10/n01 = 38/25, p = 0.13.
  - Karar kuralının üç koşulundan hiçbiri sağlanmadı.
- **İkincil aile (m = 4): dört karşılaştırmanın hepsi ayırt edilemedi.**
- Test öncesi güç analizi bu sonucu olası gösteriyordu: birincil çiftte güç 0.30, MDE ≈ 0.025.
  Gözlenen fark (+0.0159) testten önce hesaplanan MDE'nin altında ve CI'ın içinde; gerçek
  farkın büyüklüğü bilinmiyor.
- "Ayırt edilemedi", farkın sıfır olduğu anlamına gelmez. Eşdeğerlik marjı önceden
  tanımlanmadığı için bu veriden "iki model aynı" sonucu çıkmaz.

## Test skorları (n = 908, 850 grup)
CI'lar %95, split_v2 yakın-tekrar bileşenleri üzerinden grup bootstrap'iyle hesaplandı
(B = 10000, seed 42; bütün modeller aynı ağırlıklarla). CI'lar marjinaldir, eşzamanlı değildir.

| Model | Test macro-F1 [%95 CI] | Test accuracy [%95 CI] | CV macro-F1 (ort. ± std) |
|---|---|---|---|
| **berturk_ft** (seed 42, birincil) | **0.9291** [0.9114, 0.9451] | 0.9273 [0.9095, 0.9440] | 0.9188 ± 0.0098 ᵃ |
| berturk_ft_s43 | 0.9278 [0.9102, 0.9437] | 0.9262 [0.9084, 0.9429] | 0.9166 ± 0.0120 ᵃ |
| berturk_ft_s44 | 0.9250 [0.9067, 0.9417] | 0.9229 [0.9043, 0.9404] | 0.9152 ± 0.0155 ᵃ |
| **char_lr** (birincil baseline) | **0.9132** [0.8931, 0.9308] | 0.9130 [0.8933, 0.9310] | 0.9054 ± 0.0107 ᵇ |
| word_svc | 0.9123 [0.8925, 0.9299] | 0.9119 [0.8926, 0.9301] | 0.9017 ± 0.0094 ᵇ |
| word_lr | 0.9092 [0.8894, 0.9271] | 0.9086 [0.8890, 0.9270] | 0.8977 ± 0.0100 ᵇ |
| bge_m3 (+ LR) | 0.9058 [0.8851, 0.9245] | 0.9053 [0.8850, 0.9244] | 0.9072 ± 0.0094 ᵇ |
| word_cnb | 0.9043 [0.8841, 0.9225] | 0.9042 [0.8845, 0.9226] | 0.8948 ± 0.0120 ᵇ |
| char_lr_c1clean (betimsel) | 0.9037 [0.8829, 0.9223] | 0.9031 [0.8826, 0.9222] | 0.9009 ± 0.0115 ᵇ |

ᵃ Faz 4 CV: her satır bir seed'in 5 dış fold'u; std fold'lar arasıdır (ddof=1).
ᵇ Faz 2/3 nested CV: 5 fold × 3 tekrar = 15 fold; std 15 fold arasıdır.

Std'ler CI değildir ve iki satır grubunda farklı sayıda fold'dan hesaplanmıştır. Tek model
CI'larının yarı genişliği ≈ ±0.018'dir. Bu yüzden bu tablodaki sıralama tek başına bir karşılaştırma
sonucu değildir. Karşılaştırmalar eşleştirilmiş testlerle aşağıda verildi.

## Birincil karşılaştırma: berturk_ft − char_lr
Karar kuralı (`docs/test_protokol.md`): *ayırt edildi* ⇔ Holm p < 0.05, macro-F1 farkının CI'ı
0'ı dışlıyor ve sign(n10 − n01) CI'ın işaretiyle aynı.

| Büyüklük | Değer |
|---|---|
| macro-F1: berturk_ft / char_lr | 0.9291 / 0.9132 |
| Fark (macro-F1) | +0.0159, %95 CI [−0.0004, +0.0328], bootstrap SE 0.0085 |
| Fark (accuracy, betimsel) | +0.0143, %95 CI [−0.0022, +0.0317] |
| McNemar n10 / n01 | 38 / 25 (63 uyuşmayan satır) |
| Exact McNemar p (m = 1, Holm = ham) | 0.13 |
| Koşullar | p < 0.05: **hayır**. CI 0'ı dışlıyor: **hayır**. Yön tutarlılığı: CI 0'ı içerdiği için değerlendirilemez |
| **Karar** | **ayırt edilemedi** |

**Test öncesi güç (yalnız train OOF'tan, `1587e1d`):**
- α = 0.05'te accuracy MDE 0.025, kaba macro-F1 MDE 0.025.
  - Yani 908 test satırıyla ancak yaklaşık 2.5 puanlık bir fark %80 güçle yakalanabilirdi.
- Gerçek fark seed 42 OOF accuracy farkına (+0.0135) eşitse güç 0.30.
  - Karar kuralı iki koşulu birlikte istediği için ortak güç bundan da düşüktür.
- Gözlenen test farkı (+0.0159) ve CI'ı, testten önce hesaplanan MDE (0.025) ile birlikte
  okunmalı: bu test, 2.5 puandan küçük gerçek farkları %80'in altında güçle yakalar. Gerçek
  farkın büyüklüğü bilinmiyor; CI hem ≈ 0'lık hem ≈ 3.3 puanlık bir farkla uyumlu.

**İddianın kapsamı:**
- Karar tek bir eğitilmiş modele (seed 42) koşulludur. McNemar ve bootstrap test örneklemesinin
  belirsizliğini ölçer, eğitimden (seed'den) gelen belirsizliği ölçmez.
- McNemar test satırlarını bağımsız sayar. Testte 908 satırın 108'i çok satırlı yakın-tekrar
  gruplarında. Bu yüzden p iyimser olabilir; grup bootstrap CI'ı bu bağımlılığı hesaba katar.

## İkincil aile (m = 4, Holm)
Hiçbirinde koşulların hiçbiri sağlanmadı.

| Çift (a − b) | Fark macro-F1 [%95 CI] | n10 / n01 | Ham p | Holm p | Karar | Test öncesi güç (α/4 ; α) |
|---|---|---|---|---|---|---|
| char_lr − word_lr | +0.0040 [−0.0086, +0.0162] | 19 / 15 | 0.61 | 1 | ayırt edilemedi | 0.05 ; 0.15 |
| char_lr − word_svc | +0.0009 [−0.0116, +0.0129] | 17 / 16 | 1.00 | 1 | ayırt edilemedi | 0.02 ; 0.06 |
| char_lr − word_cnb | +0.0089 [−0.0066, +0.0241] | 30 / 22 | 0.33 | 1 | ayırt edilemedi | 0.08 ; 0.19 |
| bge_m3 − char_lr | −0.0074 [−0.0239, +0.0084] | 24 / 31 | 0.42 | 1 | ayırt edilemedi | 0.01 ; 0.04 |

- Güç, gerçek farkın OOF farkına eşit olduğu varsayımıyla hesaplandı. Holm'un en kötü adımı
  (α/4) ve en iyi adımı (α) gerçek gücün alt ve üst sınırıdır.
- Testten önce yazılan beklenti: bu ailede "ayırt edilemedi" beklenen sonuçtu. Dört CI'ın
  hepsi 0'ı içeriyor ve genişlikleri ≈ 2.5–3.2 puan.
- Klasik modeller arasında ve bge_m3 ile char_lr arasında bu testin ayırabileceği bir fark
  görülmedi. Bu, modellerin aynı olduğu anlamına gelmez: CI'lar 0'ın yanında birkaç puanlık
  farkları da içeriyor (ör. char_lr − word_cnb için +2.4, bge_m3 − char_lr için −2.4 puana
  kadar).

## Önceden kaydedilmiş betimsel çiftler (p yok)
Bu çiftler testten önce p'siz tanımlandı (`DESCRIPTIVE`) ve hiçbir Holm ailesine girmez. Fark
ve CI verilir; "ayırt edildi" ya da "ayırt edilemedi" kararı verilmez.

| Çift (a − b) | Fark macro-F1 [%95 CI] | Fark accuracy [%95 CI] |
|---|---|---|
| char_lr − char_lr_c1clean | +0.0095 [+0.0020, +0.0182] | +0.0099 [+0.0022, +0.0187] |
| berturk_ft − bge_m3 | +0.0233 [+0.0079, +0.0397] | +0.0220 [+0.0066, +0.0383] |
| berturk_ft_s43 − char_lr | +0.0146 [−0.0017, +0.0317] | +0.0132 [−0.0033, +0.0301] |
| berturk_ft_s44 − char_lr | +0.0118 [−0.0072, +0.0308] | +0.0099 [−0.0098, +0.0289] |

- `char_lr − char_lr_c1clean` betimsel bir gözlemdir: CI 0'ı dışlıyor ama marjinal ve
  düzeltmesizdir, çıkarım değildir. Yorumu `docs/shortcut_learning.md` → "Faz 5: test
  setinde"de.
- `berturk_ft − bge_m3`: fine-tuning edilmiş BERTurk ile donmuş bge-m3 embedding'i + LR
  arasındaki fark. Bu çift Faz 5a'da ikincil aileye alınmadı; bu yüzden CI'ın 0'ı dışlaması bir
  test sonucu olarak raporlanmaz.
- İki betimsel çiftte de CI'ın 0'ı dışlaması "ayırt edildi" anlamına gelmez.
- Seed 43 ve 44 satırları bir sonraki bölümde.

## Seed'ler ve yön tutarlılığı (betimsel, çıkarım değil)
**Test macro-F1, 3 seed:** 0.9291 (s42), 0.9278 (s43), 0.9250 (s44).
- Ortalama 0.9273 ± 0.0021 (std, ddof=1).
- Bu 3 eğitimin yayılımıdır, ortalamanın güven aralığı değildir. Std 3 değerden kestirildiği
  için çok kabadır.

**BERTurk − char_lr farkının yönü:**

| Düzey | BERTurk önde | Ayrıntı |
|---|---|---|
| CV, 5 dış fold (seed ortalaması) | 5 / 5 | d: +0.0042 … +0.0251 (`docs/faz4_sonuclar.md`) |
| CV, fold × seed | 14 / 15 | fold 0, seed 44: 0.8946 vs 0.8955 |
| CV, seed başına 5-fold ortalaması | 3 / 3 | 0.9188, 0.9166, 0.9152 vs char_lr 0.9054 |
| Test, 3 final model | 3 / 3 | +0.0159, +0.0146, +0.0118; üç CI de 0'ı içeriyor |

Bu tablo yalnızca betimseldir. Sayımlar bağımsız kanıt değildir:
- Üç test farkı aynı 908 satırda ve aynı char_lr tahminlerine karşı ölçüldü.
- CV fold'larının train kısımları örtüşür. Seed'ler aynı fold'ları ve aynı başlangıç
  ağırlıklarını (BERTurk checkpoint'i) paylaşır.
- Bu yüzden sayımlar bir işaret testine ya da birleşik bir p'ye dönüştürülemez. Yön tutarlılığı
  birincil kararı değiştirmez: birincil karar **ayırt edilemedi**.
- Üç seed'in test farkları 0.0118–0.0159 aralığında; birincil modelin (seed 42) farkı üçünün en
  büyüğü.
- Faz 4'teki CV karşılaştırması da (Nadeau-Bengio, p = 0.125) ayırt edememişti.

## CV'den teste: skorlar neden yükseldi
Dokuz modelin sekizinde test macro-F1'i CV ortalamasından yüksek:

| Model | CV | Test | Test − CV |
|---|---|---|---|
| berturk_ft (s42) | 0.9188 | 0.9291 | +0.0103 |
| berturk_ft_s43 | 0.9166 | 0.9278 | +0.0112 |
| berturk_ft_s44 | 0.9152 | 0.9250 | +0.0098 |
| char_lr | 0.9054 | 0.9132 | +0.0078 |
| char_lr_c1clean | 0.9009 | 0.9037 | +0.0028 |
| word_lr | 0.8977 | 0.9092 | +0.0115 |
| word_svc | 0.9017 | 0.9123 | +0.0105 |
| word_cnb | 0.8948 | 0.9043 | +0.0096 |
| bge_m3 | 0.9072 | 0.9058 | −0.0015 |

İki olası neden var. İkisi de ölçülerek ayrıştırılmadı.

1. **Final modeller daha çok veriyle eğitildi.**
   - CV'de her model, dış fold'un train kısmıyla eğitildi: klasik modeller ve bge_m3 ~2904
     satırla. BERTurk ~2323 satırla eğitildi, çünkü early stopping için ayrıca validation
     ayrıldı.
   - Final modeller 3630 satırın hepsiyle eğitildi. Bu, klasik modeller için %25, BERTurk için
     %56 daha fazla veri demek.
   - Öğrenme eğrisi hâlâ yükseliyorsa CV skoru, final modelin performansını aşağı yanlı
     tahmin eder. Öğrenme eğrileri ölçülmediği için bu etkinin büyüklüğü bilinmiyor.
2. **Test örneklem değişkenliği.**
   - Test tek bir 908 satırlık örneklemdir. Tek model CI'larının yarı genişliği ≈ ±0.018, yani
     ±0.01'lik bir sapma bu örneklemin olağan oynaması içinde.
   - Bütün modeller aynı test satırlarında ölçüldüğü için hataları pozitif korelasyonludur.
     CV fold'larına göre "kolay" bir test örneklemi bütün modelleri birlikte yukarı taşır.
     Sekiz modelin birlikte yükselmesi bu ortak bileşenle de uyumludur.
   - CV ortalaması ise 15 (BERT'te 5) fold'un ortalamasıdır. Daha az oynaması beklenir, ama
     fold'lar bağımsız olmadığı için bu ölçülmedi.

Yalnızca char_lr için ters yönde çalışan bir etki de var: char_lr dört klasik adayın CV
maksimumu olarak seçildi. Bu yüzden CV skoru kazananın laneti nedeniyle yukarı yanlıdır (Faz 2
tahmini +0.0027). Bu etki char_lr'nin test skorunu CV'nin altına çekmeli; char_lr'deki artış
bu etkiye rağmen oluştu. Diğer modeller seçimle teste taşınmadı.

bge_m3'ün hafif düşüşü (−0.0015) test örneklem değişkenliği ölçeğinde. Bu fark, bge_m3'ün
eğitim verisinden daha az yararlandığı anlamına gelmez; bunu gösterecek bir ölçüm yapılmadı.

Testten önce yazılan not da aynı şeyi söylüyordu (`TEST_PLAN['expectation_note']`): test
skorunun CV'den farklı çıkması tek başına bir anomali değildir.

## Keşifsel, sonradan: berturk_ft − char_lr_c1clean
**Bu karşılaştırma test sonuçları görüldükten sonra tanımlandı.** TEST_PLAN'da yoktur, hiçbir
Holm ailesine girmez, p hesaplanmadı ve karar kuralı uygulanmadı. Birincil sonucu değiştirmez.

| Çift | Fark macro-F1 [%95 CI] | Fark accuracy [%95 CI] |
|---|---|---|
| berturk_ft − char_lr_c1clean | +0.0254 [+0.0085, +0.0429] | +0.0242 [+0.0067, +0.0421] |

- **Neden bakıldı:** BERTurk tokenizer'ı U+0091–U+0094 karakterlerini siler, ham char_lr ise bu
  karakterleri görür. char_lr_c1clean bu karakterleri görmeyen char_lr'dir. Bu çift, farkı
  yalnızca bu asimetri kaldırılmış hâliyle betimler. Girdi yine de aynı değildir: BERTurk'ün
  512 token baştan kesmesi ve tokenizasyon farkları durur. (Kod ve json'daki not "girdi
  asimetrisini kaldırılmış hâliyle" der; kastedilen yalnız U+0091–U+0094'tür.)
- **Yeni bilgi sınırlı:** nokta tahmini, önceden kaydedilmiş iki farkın toplamıdır:
  (berturk_ft − char_lr) + (char_lr − char_lr_c1clean) = 0.0159 + 0.0095 = 0.0254. Bu çiftin
  eklediği tek şey, farkın aynı bootstrap ağırlıklarıyla hesaplanan CI'ıdır.
- **Neden kanıt sayılmaz:** referans model sonuçlar görüldükten sonra, daha düşük skorlu
  olduğu bilinerek seçildi. CI'ın 0'ı dışlaması bu yüzden bir çıkarım olarak okunmaz.
  Önceden kaydedilmiş birincil referans ham char_lr'dir ve o değişmez.

## split_v1 sızıntı teşhisi (`RULES['v1_diagnosis']`)
- **Amaç:** düz stratified split_v1'de, aynı haberin yeniden yazılmış sürümleri train ile test
  arasına bölünmüştü. Bu teşhis, bölünmenin skoru nasıl etkilediğini betimler.
- **Kurulum:** char_lr dondurulmuş ayarlarla (C=100, char_wb, (2,4)) kullanıldı, ayar yapılmadı.
  split_v1 train'inde (3630 satır) fit edildi, split_v1 test'inde (908 satır) tahmin edildi.
- **"Sızmış" satır:** yakın-tekrar bileşeninde (split_v2 `components`, kosinüs > 0.8) en az
  bir v1 train satırı bulunan v1 test satırı.

| Alt küme | n | Grup | Accuracy |
|---|---|---|---|
| Sızmış | 93 | 90 | 0.9677 |
| Temiz | 815 | 804 | 0.9104 |
| **Fark (sızmış − temiz)** | | | **+0.0573**, %95 grup bootstrap CI [+0.0132, +0.0942] |

- Bootstrap: B = 10000, seed 42, bileşen düzeyinde. Düşen replikat yok. p hesaplanmadı.
- **Okuma:**
  - Train'de yakın bir kopyası bulunan test satırlarında accuracy ≈ 5.7 puan daha yüksek.
  - CI geniş (≈ 1.3–9.4 puan), çünkü sızmış alt küme yalnızca 90 grup.
  - Bu, split_v2'ye geçme gerekçesiyle tutarlı bir betimsel bulgudur. Fark yalnızca
    ezberlemeye atfedilemez: sızmış alt küme rastgele seçilmedi ve sınıf bileşimi farklı.
    - Sızmış alt kümede sağlık 25, dünya 18, spor 3 satır var. Temiz alt kümede sınıflar
      101–126 arasında.
    - Sınıflar zorlukta farklıdır, bu yüzden bileşim farkı accuracy farkının bir kısmını
      açıklayabilir. Bu ayrıştırma yapılmadı.
  - "Temiz" alt küme sızıntıdan tamamen arınmış sayılmaz: benzerliği ≤ 0.8 olan yeniden
    yazımlar bulunabilir.
- **Yan yana (karşılaştırma değil):**
  - v1 test genel skoru: macro-F1 0.9179 [0.8992, 0.9347], accuracy 0.9163 [0.8977, 0.9338].
  - v2 test char_lr skoru: macro-F1 0.9132 [0.8931, 0.9308].
  - İki test kümesi farklıdır: kısmen örtüşürler ve eğitim bileşimleri de farklıdır. Ayrıca
    char_lr'nin ayarları v1 test satırlarının çoğunu içeren split_v2 train'inde seçildi.
  - Bu yüzden v1 − v2 farkı sızıntı etkisi olarak sunulmaz ve p hesaplanmaz.

## Sınıf bazlı F1 (betimsel, CI yok)
| Sınıf | berturk_ft | char_lr | char_lr_c1clean |
|---|---|---|---|
| siyaset | 0.949 | 0.940 | 0.928 |
| dünya | 0.884 | 0.887 | 0.885 |
| ekonomi | 0.875 | 0.884 | 0.852 |
| kültür | 0.973 | 0.930 | 0.925 |
| sağlık | 0.935 | 0.925 | 0.925 |
| spor | 0.981 | 0.965 | 0.965 |
| teknoloji | 0.906 | 0.862 | 0.846 |

Sınıf başına 114–138 test satırı var. Bu F1'lerin CI'ı hesaplanmadı ve sınıf başına birkaç
satırlık oynamaya duyarlılar. Sınıflar arasındaki farklar karşılaştırma sonucu olarak
okunmamalıdır.

## Sınırlılıklar
- **Tek test seti, tek bölme.** Bütün test sonuçları split_v2'nin tek bir test fold'una
  dayanır (908 satır, 850 grup).
- **p iyimser olabilir.** Exact McNemar test satırlarını bağımsız sayar ve test içindeki
  yakın-tekrar bağımlılığını yok sayar. Accuracy farkının grup bootstrap CI'ı bu yüzden
  betimsel olarak verildi.
- **CI'lar marjinaldir.** Eşzamanlı değildirler. Aileler ayrı α'da tutuldu; aileler arası
  toplam hata kontrolü yapılmadı.
- **Seed belirsizliği.** Birincil iddia tek eğitilmiş modele koşulludur. Seed değişkenliği
  yalnızca betimsel verildi (3 seed).
- **Sayısal ortam.** Test tahminleri yerel CPU'da fp32 ile, CV/OOF tahminleri Colab'da fp16
  ile üretildi. Sınırdaki satırlarda bu küçük bir oynama yaratabilir.
- **Kısayollar.** Skorlar bu veri kümesindeki kaynak artefaktlarını (bozuk tırnaklar, ajans
  künyeleri) da içerir. Başka kaynaklardan gelen haberler için iyimser olabilir; dış kaynakta
  ölçülmedi (`docs/shortcut_learning.md`).

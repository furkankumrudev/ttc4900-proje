# Faz 7 sonuçları: betimsel hata analizi

**Kaynaklar** (hepsi `outputs/metrics/`):
- 7b: `phase7_error_analysis_20261017-102723.json`; figür `outputs/figures/phase7_confusion.png`.
- Güven bantları (sonradan, betimsel): `phase7_conf_bands_20261017-104218.json`.
- Protokol: `docs/faz7_protokol.md`, `src/ttc4900/error_analysis.py` (`ERROR_RULES`).

**Bu analiz tamamen betimseldir.**
- p hesaplanmadı, karar kuralı yok.
- Hiçbir model, ayar ya da Faz 1–6 kuralı değişmedi. Faz 7'nin kendi kurallarına 7b'den sonra
  iki ekleme yapıldı: güven bantları adımı ve metin kuralı (`ERROR_RULES['change_log']`).
- Faz 5'in karşılaştırma sonuçları (birincil çift: ayırt edilemedi) ve Faz 6'nın kalibrasyon
  kararları olduğu gibi geçerli.
- Modeller arasında doğruluk ya da kalibrasyon sıralaması yapılmaz.
- Bulgular yalnızca BERTurk seed 42 final modeli ve char_lr için geçerlidir.

**Veri ve atıf.**
- Veri seti Hugging Face `savasy/ttc4900`, sabit revision `236e17b0…`.
- Dataset card'a göre veri Kemik grubundan (Yıldız Teknik Üniversitesi) alınmış; kaynağı Türk
  haber siteleri. Card'ın istediği atıf:
  - Kaggle sayfası: <https://www.kaggle.com/savasy/ttc4900>
  - Yıldırım, S. ve Yıldız, T. (2018). *A comparative analysis of text classification for
    Turkish language.* Pamukkale Univ Muh Bilim Derg 24(5), 879–886.
    doi:10.5505/pajes.2018.15931.
- **Lisans:** card'ın metadata'sında `license: unknown`, "Licensing Information" bölümü boş.
  Bu yüzden belgede haber metni alıntılanmaz. Örnekler, kişi ve kurum adları çıkarılmış genel
  konu etiketleri ve row_id ile verilir.

## Sıra
| Commit | Ne |
|---|---|
| `0be03ef` | 7a: kurallar, kod, testler, protokol (sayı yok) |
| `ce59df5` | 7b sonuçları, yorumsuz |
| `02ed923` | 7b'den sonra: güven bantları adımı (sonradan, betimsel) ve metin kuralı |
| `acf34a5` | Güven bantları sonucu, yorumsuz |

Analizin dayandığı kayıtlar doğrulandı:
- Test npz'lerinin sha256'ları Faz 5 manifest'iyle aynı.
- Skorların argmax'ı kayıtlı tahminlerle aynı.
- Bootstrap ağırlıkları Faz 5 ile birebir aynı: accuracy CI'ları tuttu.

**Çok hücre uyarısı.**
- Bu belgede her biri ayrı CI'lı onlarca oran var. CI'lar marjinaldir ve çoklu karşılaştırma
  için düzeltilmedi.
- Örneğin 6. bölümün sınıf içi tablosunda CI verilebilen 15 hücre var. Gerçek fark olmasa bile
  bunlardan yaklaşık 0.75'inin CI'ı şans eseri 0'ı dışlar.
- Hiçbir hücre karar olarak okunmamalı. n < 20 olan hücrelerde CI verilmedi.

## 1. Karışıklık matrisleri
Test 908 satır. Hata sayıları: BERT 66, char_lr 79. Bu iki sayının farkı Faz 5'te ayırt
edilemedi. Figür: `outputs/figures/phase7_confusion.png`.

| Gerçek sınıf (n) | BERT doğru | char_lr doğru |
|---|---|---|
| siyaset (138) | 130 | 132 |
| dünya (135) | 122 | 118 |
| ekonomi (138) | 123 | 122 |
| kültür (114) | 110 | 106 |
| sağlık (126) | 115 | 117 |
| spor (128) | 126 | 125 |
| teknoloji (129) | 116 | 109 |

Sınıf başına farklar test edilmedi; bu tablodan sıralama okunmaz. Sınıf başına 114–138 satır
var ve sayılar birkaç satırlık oynamaya duyarlıdır.

## 2. En çok karışan sınıf çiftleri
21 sırasız çiftin **tamamı**, simetrik karışma sayısına (a→b + b→a) göre sıralandı.

| Sıra | BERT | char_lr |
|---|---|---|
| 1 | dünya↔ekonomi 11 (4+7) | ekonomi↔teknoloji 13 (7+6) |
| 2 | ekonomi↔teknoloji 9 (4+5) | dünya↔teknoloji 8 (2+6) |
| 3 | siyaset↔dünya 7 (3+4) | = sağlık↔teknoloji 7 (2+5) |
| 4 | dünya↔teknoloji 6 | = siyaset↔dünya 7 (1+6) |
| 5–8 | = dünya↔sağlık, ekonomi↔sağlık, sağlık↔teknoloji, siyaset↔ekonomi: 5'er | dünya↔sağlık 6 |

"=" beraberlik demek; beraberlikler alfabetik sıraya göre dizildi.

- **Önceden adı konan çiftler:** ekonomi↔teknoloji BERT'te 2., char_lr'de 1. sırada.
  siyaset↔dünya BERT'te 3., char_lr'de 3.–4. sırada (beraber).
- **BERT'te birinci çift dünya↔ekonomi.** Bu çiftin önceden adı konmamıştı; 21 çiftin tamamı
  sıralandığı için görünür oldu.
- **Sıralar kararsız.** Hücreler küçük (4–13). BERT'te birinci ile ikinci arasındaki fark 2
  satır. 21 çiftin en büyüğü doğası gereği yüksek çıkar.

## 3. BERT ile char_lr'nin ayrıştığı satırlar
Faz 5'teki McNemar uyuşmazlıkları: n10 = 38 (BERT doğru, char_lr yanlış) ve n01 = 25 (tersi).
Bu bölüm sayımdır, CI yoktur. 25 ve 38 satırlık alt kümeler için yalnızca betimseldir.

| | n10 (38) | n01 (25) | Tüm test (908) |
|---|---|---|---|
| Gerçek sınıf | ekonomi 10, teknoloji 9, dünya 8, kültür 5, siyaset 3, spor 3, sağlık 0 | ekonomi 9, siyaset 5, dünya 4, sağlık 2, spor 2, teknoloji 2, kültür 1 | — |
| Yanılan modelin tahmini | char_lr: ekonomi 8, siyaset 6, teknoloji 6, dünya 5, kültür 5, sağlık 4, spor 4 | BERT: dünya 11, ekonomi 9, siyaset 3, spor 1, teknoloji 1 | — |
| C1 içeren satır | 20/38 = 0.53 [0.37, 0.69] | 10/25 = 0.40 [0.21, 0.61] | 307/908 = 0.34 [0.31, 0.37] |

- **Uzunluk:** ayrışan satırların karakter uzunluğu medyanı tüm testinkinden düşük: 1229 ve
  1337, testte 1638. Ama n01'in ortalaması (2201) testinkinden (1996) yüksek; dağılım çarpık.
  CI hesaplanmadı, yorum yapılmaz.
- Güven değerleri bu tablodan çıkarıldı: iki modelin olasılıkları farklı ölçeklerde, bu yüzden
  modeller arası karşılaştırılmaz.

## 4. Yüksek güvenli hatalar ve paydası
- **Kalibre güven:** max softmax(z / T). T = 1.283 (BERT), 0.916 (char_lr).
- Güven bantları adımı 7b sonuçları görüldükten sonra, betimsel olarak eklendi.
- **≥ 0.90 eşiği iki modelde farklı ölçekli olasılıklara uygulandı.** Bu yüzden bantların
  büyüklükleri, hata sayıları ve payları modeller arasında karşılaştırılmaz. Aşağıdaki iki
  tablo ayrı okunmalı; her biri yalnız kendi modelini betimler.

**BERT (seed 42)**

| | ≥ 0.90 | ≥ 0.95 |
|---|---|---|
| Banttaki tahmin | 848/908 = 0.934 [0.917, 0.950] | 815/908 = 0.898 [0.877, 0.918] |
| Bant içi doğruluk | 803/848 = 0.947 [0.931, 0.961] | 779/815 = 0.956 [0.941, 0.970] |
| Bant içi ortalama güven | 0.983 | 0.985 |
| Bant altı doğruluk | 39/60 = 0.65 [0.52, 0.77] | 63/93 = 0.68 [0.58, 0.77] |
| Hataların banttaki payı | 45/66 = 0.68 [0.56, 0.80] | 36/66 = 0.55 [0.42, 0.67] |

**char_lr**

| | ≥ 0.90 | ≥ 0.95 |
|---|---|---|
| Banttaki tahmin | 716/908 = 0.789 [0.760, 0.816] | 633/908 = 0.697 [0.664, 0.729] |
| Bant içi doğruluk | 697/716 = 0.973 [0.961, 0.984] | 621/633 = 0.981 [0.970, 0.991] |
| Bant içi ortalama güven | 0.984 | 0.992 |
| Bant altı doğruluk | 132/192 = 0.69 [0.61, 0.75] | 208/275 = 0.76 [0.70, 0.81] |
| Hataların banttaki payı | 19/79 = 0.24 [0.15, 0.34] | 12/79 = 0.15 [0.08, 0.23] |

- **Paydayla okuma (model içinde):**
  - BERT'in hatalarının %68'i ≥ 0.90 bandında. Ama BERT'in tahminlerinin %93'ü zaten bu
    bantta.
  - Güven hatalarla ilişkisiz olsaydı, hataların da yaklaşık %93'ü bu bantta olurdu. Gözlenen
    %68 bunun altında: hatalar bant dışında yoğunlaşıyor ve güven, doğruyla yanlışı ayırmak
    için bilgi taşıyor.
  - Aynı okuma char_lr için de geçerli: hataların %24'ü bantta, tahminlerin %79'u bantta.
  - "BERT hatalarının çoğu yüksek güvenli" cümlesi bu payda olmadan yanıltıcıdır.
- **Bant içi güven ve doğruluk:**
  - BERT: ortalama güven 0.983, doğruluk 0.947.
  - char_lr: ortalama güven 0.984, doğruluk 0.973.
  - BERT için bu, Faz 6'da raporlanan kalan aşırı güvenle benzer yönde bir gözlem. Ancak Faz
    6'da bakılan en kalabalık bin ile buradaki bant farklı satır kümeleri.
  - Bu tablo kalibrasyon sıralaması değildir. Faz 6'da kalibrasyon sonrası metrik farkları
    ayırt edilemedi.
- **Grup üyeliği:**
  - BERT'in 45 yüksek güvenli hatasından 4'ü çok satırlı bir yakın-tekrar grubunda, 2'si
    karışık etiketli bir grupta.
  - char_lr'nin 19 yüksek güvenli hatasında bu sayılar 2 ve 2.

### İki modelin birlikte yüksek güvenle yanıldığı 14 satır
- Her iki modelin de güveni ≥ 0.90 ve ikisi de yanlış. 14 satırın 14'ünde iki model aynı
  yanlış etikete gidiyor.
- Bu 14 satır, BERT'in 45 yüksek güvenli hatasının %31'i.
- **Taban yok.** İki modelin ortak hatalarında aynı yanlış etiketin genel olarak ne sıklıkta
  görüldüğü hesaplanmadı. İki model aynı train etiketlerini ve kısmen aynı kısayolları
  öğrendiği için bağımsız değildir. Ortak karışıklıkların yoğun olduğu çiftlerde aynı yanlış
  etiket, etiket gürültüsü olmadan da beklenir.
- **Olası etiket gürültüsü adayları (hipotez):** bu satırlar, etiketin konuyla uyuşmadığı
  satırlar olabilir. Alternatifler de mümkün: iki modelin paylaştığı bir kısayol, ya da birden
  fazla sınıfa yakın bir konu. Etiketler sistematik olarak incelenmedi, hüküm verilmez.
- row_id'ler: 409, 517, 790, 973, 1257, 1689, 1699, 1820, 1955, 2258, 2643, 4266, 4745, 4754.
  2643 aynı zamanda karışık etiketli bir test grubunda (5. bölüm).

## 5. Karışık etiketli gruplar (olası etiket gürültüsü adayları)
- split_v2'de 247 çok satırlı yakın-tekrar grubu var (train ve test birlikte; TF-IDF kosinüsü
  > 0.8).
- Bunların 8'inde farklı etiketli satırlar bulunuyor: 6 test ve 11 train satırı.
- Tahminler test satırlarında final modellerden, train satırlarında OOF'tan (BERT seed 42 CV,
  char_lr tekrar 0) geliyor.

| Grup (row_id: taraf, etiket → BERT / char_lr) |
|---|
| 192: test, siyaset → dünya / siyaset · 1182: test, dünya → dünya / siyaset |
| 1178: test, dünya → dünya / dünya · 2643: test, kültür → dünya / dünya |
| 2057: test, ekonomi → ekonomi / sağlık · 3478: test, sağlık → ekonomi / sağlık |
| 206: train, siyaset → siyaset / siyaset · 1207: train, dünya → siyaset / siyaset |
| 675: train, siyaset → ekonomi / ekonomi · 1609: train, ekonomi → siyaset / ekonomi |
| 1317: train, dünya → ekonomi / ekonomi · 1634: train, ekonomi → ekonomi / ekonomi |
| 1576: train, ekonomi → teknoloji / teknoloji · 1813: train, ekonomi → teknoloji / teknoloji · 4395: train, teknoloji → teknoloji / teknoloji |
| 2038: train, ekonomi → teknoloji / kültür · 4762: train, teknoloji → teknoloji / kültür |

- Kosinüs > 0.8 her zaman "aynı haber" demek değildir; aynı olay ya da aynı kalıp da olabilir.
  Satırların aynı haberin sürümleri olduğu yalnızca iki test grubunda (192/1182 ve 1178/2643)
  metne bakılarak görüldü. Diğer gruplar incelenmedi.
- Aynı haberin sürümleri farklı etiket taşıyorsa, bir modelin ikisini birden doğru bilmesi
  beklenmez. Bu gruplar **olası etiket gürültüsü adaylarıdır (hipotez)**. Etiketlere hüküm
  verilmedi, yeniden etiketleme yapılmadı.
- **Test grupları:**
  - 192/1182'de BERT iki satıra da "dünya", char_lr iki satıra da "siyaset" diyor. Her model
    satırlardan birini doğru biliyor.
  - 1178/2643 ve 2057/3478'de modellerin en az biri iki satıra aynı etiketi veriyor.
- Etkisi küçük: 908 test satırının 6'sı bu gruplarda.
- **Test örnekleri (genel konu):**
  - 192 ve 1182: iki ülke hükümeti arasındaki karşılıklı açıklamalar. Biri "siyaset", diğeri
    "dünya" etiketli.
  - 1178 ve 2643: yabancı bir ülkede bir konserle ilgili düzenleme önerisi. Biri "dünya",
    diğeri "kültür" etiketli.

## 6. C1 artefaktı ve hatalar (karıştırıcı değişken çerçevesi)
- **C1 nedir:** U+0091–U+0094 kontrol karakterleri, yani bozuk tırnaklar
  (`docs/shortcut_learning.md`).
- **Testteki dağılım:** 908 satırın 307'sinde var.
- **Sınıflara göre yayılım çok farklı:** ekonomi 95/138, kültür 67/114, teknoloji 42/129,
  spor 41/128, sağlık 34/126, siyaset 16/138, dünya 12/135.
- Bu yüzden C1 varlığı sınıfla karışıktır ve muhtemelen kaynakla ve konuyla da karışıktır.

| Model | C1 satırlarında hata | C1'siz satırlarda hata | Fark [%95 CI] |
|---|---|---|---|
| BERT (C1'i görmez; tokenizer siler) | 28/307 = 0.091 | 38/601 = 0.063 | +0.028 [−0.009, +0.067] |
| char_lr (C1'i görür) | 38/307 = 0.124 | 41/601 = 0.068 | +0.056 [+0.015, +0.100] |
| char_lr_c1clean (C1 silinmiş) | 44/307 = 0.143 | 44/601 = 0.073 | +0.070 [+0.027, +0.116] |

- **C1 silindikten sonra da C1'li satırlarda hata oranı yüksek kalıyor.**
  - char_lr_c1clean'in farkı +0.070, char_lr'ninkinden (+0.056) küçük değil. Bu iki farkın
    eşleştirilmiş farkı hesaplanmadı.
  - Bu, C1 varlığının kaynak ya da konu farkının bir işareti olabileceğiyle uyumlu.
    char_lr_c1clean yalnızca C1'i siliyor; ajans künyeleri gibi diğer kısayollar duruyor.
  - Bu yüzden C1 satırlarındaki yüksek hata oranı **artefakta tek başına atfedilmez**.
- **BERT'in CI'ı 0'ı içeriyor:** [−0.009, +0.067]. Nokta tahmini aynı yönde, ama veri hem
  0'la hem 6.7 puanlık bir farkla uyumlu.

**Sınıf içinde** (C1 hatası/n karşı C1'siz hatası/n: fark [%95 CI]; n < 20 hücrede CI yok):

| Sınıf | BERT | char_lr | char_lr_c1clean |
|---|---|---|---|
| siyaset | 2/16 vs 6/122: +0.076 (—) | 1/16 vs 5/122: +0.022 (—) | 1/16 vs 8/122: −0.003 (—) |
| dünya | 2/12 vs 11/123: +0.077 (—) | 5/12 vs 12/123: +0.319 (—) | 4/12 vs 12/123: +0.236 (—) |
| ekonomi | 14/95 vs 1/43: +0.124 [+0.037, +0.208] | 12/95 vs 4/43: +0.033 [−0.085, +0.142] | 17/95 vs 3/43: +0.109 [−0.006, +0.216] |
| kültür | 2/67 vs 2/47: −0.013 [−0.086, +0.054] | 6/67 vs 2/47: +0.047 [−0.043, +0.136] | 7/67 vs 2/47: +0.062 [−0.031, +0.155] |
| sağlık | 1/34 vs 10/92: −0.079 [−0.165, +0.010] | 1/34 vs 8/92: −0.058 [−0.137, +0.030] | 1/34 vs 8/92: −0.058 [−0.137, +0.030] |
| spor | 0/41 vs 2/87: −0.023 [−0.075, +0.000] | 1/41 vs 2/87: +0.001 [−0.049, +0.064] | 1/41 vs 2/87: +0.001 [−0.049, +0.064] |
| teknoloji | 7/42 vs 6/87: +0.098 [−0.027, +0.231] | 12/42 vs 8/87: +0.194 [+0.041, +0.353] | 13/42 vs 9/87: +0.206 [+0.054, +0.367] |

- **CI verilebilen 15 hücrenin 3'ünde CI 0'ı dışlıyor:** BERT ekonomi, char_lr teknoloji,
  char_lr_c1clean teknoloji. 12 hücrede CI 0'ı içeriyor.
- **İşaretler karışık:** sağlıkta üç modelin farkı da negatif. Kültür ve sporda BERT'in farkı
  da negatif.
- **Seyrek hücreler:** BERT ekonomideki fark, C1'siz 43 satırdaki tek bir hataya dayanıyor
  (1/43).
- **Dejenere CI'lar:** sıfıra çok yakın olaylı hücrelerde yüzdelik bootstrap bilgi vermez.
  - BERT'in spor sınıfında C1 satırlarındaki hata 0/41, oranın CI'ı [0.000, 0.000]. Bu CI
    "hata oranı kesin olarak 0" diye okunmamalı.
  - Aynı nedenle BERT spor farkının üst sınırı tam 0.000. BERT ekonomide C1'siz oranın CI'ının
    alt sınırı da 0.000.

### char_lr ile char_lr_c1clean'in ayrıştığı 13 satır (küçük, betimsel)
İki model aynı yöntem ve ayarlarla, ama farklı girdiyle eğitildi: c1clean'de C1 karakterleri
train ve test metninden silindi. Model yeniden eğitildiği için bütün katsayılar değişti. Bu
yüzden ayrışmalar C1 silmenin etkisi ile yeniden eğitimin genel etkisinin toplamıdır.

| | Satır | C1 içeren | C1 içermeyen |
|---|---|---|---|
| char_lr doğru, c1clean yanlış | 11 | 7 | 4 |
| c1clean doğru, char_lr yanlış | 2 | 1 | 1 |
| Toplam ayrışma | 13 | 8 | 5 |

- Ayrışan 13 satırın 8'i C1 içeriyor (%62). Testteki C1 payı %34.
- C1'siz 5 satırda da ayrışma var ve bunların 4'ünde char_lr doğru. Bu, yeniden eğitimin
  genel etkisini gösterir.
- Sayılar çok küçük; tek başına bir etki olarak okunmaz.
- **Örnekler (genel konu):**
  - 1849: uluslararası bir enerji/denizcilik ticareti haberi. Etiket "ekonomi"; char_lr
    "ekonomi", c1clean ve BERT "dünya". Metin C1 içeriyor.
  - 1463: yabancı bir ülkenin askerî denizcilik haberi. Etiket "ekonomi"; char_lr "ekonomi",
    c1clean ve BERT "dünya". Metin C1 içeriyor.

## Diğer örnekler (kategori başına en çok 2)
- **Seçim kuralı:** yanılan modelin kalibre güveni azalan. Bu kural, en emin hataları, yani
  etiketin tartışmalı görünebildiği satırları öne çıkarır. Örnekler hataları temsil etmez.
- BERT için seçilen iki yüksek güvenli hatanın ikisi de iki modelin ortak hatası. Oysa BERT'in
  yüksek güvenli hatalarının yalnızca %31'i ortak.
- Konu etiketleri genel tutuldu ve hüküm değildir.

| Kategori | row_id | Genel konu | Etiket → BERT / char_lr |
|---|---|---|---|
| BERT doğru, char_lr yanlış | 244 | bir siyasetçinin bir spor kulübüne kutlama mesajı | siyaset → siyaset / spor |
| | 1182 | iki ülke hükümeti arasındaki karşılıklı açıklamalar | dünya → dünya / siyaset |
| BERT yanlış, char_lr doğru | 1463 | yabancı bir ülkenin askerî denizcilik haberi | ekonomi → dünya / ekonomi |
| | 1290 | komşu bir yönetimde parti kurultayı | dünya → siyaset / dünya |
| BERT yüksek güvenli hata | 1699 | kamu bilişim güvenliği altyapısı | ekonomi → teknoloji / teknoloji |
| | 2643 | yabancı bir ülkede konserle ilgili düzenleme önerisi | kültür → dünya / dünya |
| char_lr yüksek güvenli hata | 244 | (yukarıda) | siyaset → siyaset / spor |
| | 1689 | mobil internet hız ölçümleri | ekonomi → teknoloji / teknoloji |
| İki model birlikte yüksek güvenli hata | 1689 | (yukarıda) | ekonomi → teknoloji / teknoloji |
| | 4754 | bir tıbbi cerrahi teknik | teknoloji → sağlık / sağlık |

## Sınırlılıklar
- **Tek test seti, tek eğitilmiş model.** Bulgular yalnızca BERT seed 42 ve char_lr içindir.
- **Payda gerekli.** Yüksek güvenli hata sayıları güven bantları olmadan yorumlanmamalı.
  Güven eşiği modeller arasında karşılaştırılmaz.
- **Hipotezler test edilmedi.** C1'in karıştırıcı rolü, etiket gürültüsü adayları ve sınıf
  çiftleri sonradan gözlemlenmiş hipotezlerdir.
- **Etiketler sistematik olarak incelenmedi.** "Olası etiket gürültüsü" bir aday listesidir.
- **Bu test satırları eğitime girmemeli.** Aday satırlar ileride bir temizlik ya da yeniden
  eğitim için kullanılırsa test sızıntısı olur.
- **Güven bantları adımı 7b sonuçlarından sonra eklendi.** Sonradan ve betimseldir; hiçbir
  karara girmez.

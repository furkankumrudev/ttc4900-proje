# TTC4900 Türkçe Haber Sınıflandırma

> **Taslak.** Faz 7 (hata analizi) sonrası güncellendi. Hugging Face Hub ve Gradio demosu
> henüz yapılmadı; ilgili bölümler eklendikçe güncellenecek.

7 sınıflı Türkçe haber metinlerinde klasik yöntemleri (TF-IDF + doğrusal modeller) BERTurk
fine-tuning ile karşılaştıran bir proje. Projenin odağı en yüksek skoru kovalamak değil;
karşılaştırmanın yakın-tekrar sızıntısına karşı önlem alınmış ve protokolü test açılmadan
dondurulmuş biçimde yapılması (resmî bir ön-kayıt değil; dondurma git commit'leriyle):
- Split grup bilinçli kuruldu (TF-IDF kosinüsü > 0.8 olan yakın tekrarlar aynı tarafta).
- Bütün seçimler yalnızca train içinde yapıldı.
- Test yöntemi, güç analizi ve sonuç cümleleri test açılmadan donduruldu.

## Ana sonuç
Test seti: 908 haber, 850 yakın-tekrar grubu. %95 CI'lar grup bootstrap'iyle hesaplandı.

| Model | Test macro-F1 [%95 CI] |
|---|---|
| BERTurk fine-tuning (seed 42, birincil) | 0.9291 [0.9114, 0.9451] |
| Karakter TF-IDF + LR (`char_lr`, birincil baseline) | 0.9132 [0.8931, 0.9308] |
| Kelime TF-IDF + LinearSVC | 0.9123 [0.8925, 0.9299] |
| Kelime TF-IDF + LR | 0.9092 [0.8894, 0.9271] |
| bge-m3 embedding + LR (donmuş encoder) | 0.9058 [0.8851, 0.9245] |
| Kelime TF-IDF + ComplementNB | 0.9043 [0.8841, 0.9225] |

CI'lar büyük ölçüde örtüşüyor; tablodaki sıralama tek başına bir karşılaştırma sonucu değildir.
Karşılaştırmalar aşağıdaki eşleştirilmiş testlerle yapıldı.

**Birincil karşılaştırma (BERTurk − char_lr): ayırt edilemedi.**
- Fark +0.0159 macro-F1, %95 CI [−0.0004, +0.0328].
- Exact McNemar n10/n01 = 38/25, p = 0.13.
- Testten önce yazılan güç analizi bunu olası gösteriyordu: 908 test örneğiyle %80 güçle ancak
  yaklaşık **2.5 puanlık** bir fark yakalanabilirdi. Train OOF farkında güç 0.30'du.
- Bu, iki modelin aynı olduğu anlamına gelmez. Eşdeğerlik marjı tanımlanmadı; CI hem ≈ 0'lık
  hem ≈ 3.3 puanlık bir farkla uyumlu.

**İkincil aile** (klasik modeller ve bge-m3, char_lr'ye karşı, Holm m = 4): dört
karşılaştırmanın hepsi ayırt edilemedi. Bu da "modeller aynı" demek değildir: CI'lar birkaç
puanlık farkları da içeriyor (en geniş uçlar ≈ ±2.4 puan).

**Betimsel bulgular (karar kuralına girmez):**
- Üç BERTurk seed'inin test macro-F1'i 0.9250–0.9291. Üçü de char_lr'nin üzerinde; üç farkın
  CI'ı da 0'ı içeriyor.
  - CV'de BERTurk (seed ortalaması) 5 dış fold'un 5'inde öndeydi; Faz 4'ün CV karşılaştırması
    (Nadeau-Bengio, p = 0.125) da ayırt edememişti.
  - Bu sayımlar bağımsız kanıt değildir: aynı test seti ve örtüşen train fold'ları
    kullanıldı. Birincil kararı değiştirmezler.
- Test macro-F1'i 9 modelin 8'inde CV ortalamasından 0.3–1.2 puan yüksek çıktı (bge-m3'te
  −0.15). Olası iki neden var, ikisi de ayrıştırılmadı:
  - Final modeller daha fazla veriyle eğitildi: 3630 satır, CV'de ~2904 (BERTurk'te ~2323).
  - Test tek bir 908 satırlık örneklem ve oynaması var.

Ayrıntı: [`docs/faz5_sonuclar.md`](docs/faz5_sonuclar.md).

## Olasılık kalibrasyonu (Faz 6)
Temperature scaling kullanıldı: p̂ = softmax(z / T). T yalnızca train OOF tahminlerinden
öğrenildi. Büyüklük ΔNLL = NLL(önce) − NLL(sonra); pozitif değer iyileşme demek. Ana kanıt
train içi cross-fit değerlendirmedir; test bir kez ve teyit için kullanıldı.
- Kurallar hiçbir sayı hesaplanmadan donduruldu.
- Train sonuçlarından sonra, test açılmadan üç değişiklik yapıldı: sonuç cümlesi şablonu, yorum
  kuralı ve betimsel fold dökümü. Karar mantığı değişmedi (`docs/faz6_protokol.md`).

| Model | T̂ | Train ΔNLL, cross-fit [%95 CI] | Test ΔNLL [%95 CI] | Karar |
|---|---|---|---|---|
| BERTurk (seed 42) | 1.283 | +0.020 [+0.012, +0.028] | +0.051 [+0.033, +0.071] | gösterildi; testte **teyit edildi** |
| char_lr | 0.916 | +0.001 [−0.001, +0.004] | −0.002 [−0.006, +0.002] | gösterilemedi; testte **teyit edilemedi** |

T değerleri modeller arasında karşılaştırılamaz (logit ve decision function farklı ölçeklerde);
anlamlı olan yalnızca T'nin 1'e göre yönü.

- **BERTurk:** T > 1, model aşırı emindi. Ölçekleme testte NLL'yi ve ECE'yi düşürdü.
  - ECE (15 bin) 0.058'den 0.041'e indi. ECE yanlı ve bin sayısına duyarlı bir kestiricidir:
    10/15/20 bin için sonra-değerleri 0.037/0.041/0.043. Karar yalnızca NLL'ye dayanır.
  - Kalan bir aşırı güven var: en yüksek güven bandında ortalama güven 0.98, accuracy 0.95.
  - "Teyit edildi" bu tek eğitilmiş koşu (seed 42) ve bu test seti için geçerli. Seed 43'te
    train içi kazanç gösterilemedi. Üç (koşu, fold) biriminde kalibrasyon NLL'yi kötüleştirdi.
    T farklı eğitim koşularında değişiyor (15 CV koşusunda std 0.25).
- **char_lr:** başlangıçta zaten iyi kalibreydi. ECE T = 1'de küçüktü (test 0.025) ve T = 0.92
  yalnız hafif bir çekingenlik gösteriyor (CI 1'i dışlıyor).
  - "Teyit edilemedi" etki olmadığı anlamına gelmez. Ancak CI üst sınırı +0.0025, büyük bir
    kazanç beklenmiyor.
- **Modeller arası sıralama yok.** Kalibrasyondan sonra NLL, Brier ve ECE farkları ayırt
  edilemedi.
  - Accuracy ile olasılık güvenilirliği farklı özelliklerdir: T = 1'de test NLL'si char_lr'de
    0.282, BERT'te 0.338 (fark için CI hesaplanmadı). Accuracy nokta tahminleri ise ters
    sırada (0.913 ve 0.927; Faz 5'te ayırt edilemedi). Betimseldir, sıralama değildir.
- **Sonradan gözlemlenen hipotez (test edilmedi):** CV koşularında T, best epoch ile artıyor
  görünüyor. Final model sabit 4 epoch eğitildiği için T̂_final onu eksik düzeltiyor olabilir.
  Bu bir gelecek çalışma önerisi; testte başka bir T denenmedi.

Ayrıntı: [`docs/faz6_sonuclar.md`](docs/faz6_sonuclar.md),
[`docs/faz6_protokol.md`](docs/faz6_protokol.md).

## Hata analizi (Faz 7, betimsel)
- **Kapsam:** BERTurk seed 42 final modeli ve char_lr. Tamamen betimseldir: p yok, modeller
  arasında doğruluk ya da kalibrasyon sıralaması yok.
- **Kurallar:** Faz 1–6 kuralları değişmedi. Güven bantları adımı 7b sonuçları görüldükten
  sonra, betimsel olarak eklendi.
- **Örnekler:** veri setinin lisansı belirsiz olduğu için haber metni alıntılanmadı. Örnekler
  genel konu etiketi ve row_id ile verildi.

**Karışan sınıf çiftleri.** 21 çiftin tamamı sıralandı.
- char_lr'de birinci çift ekonomi↔teknoloji (13 hata).
- BERT'te birinci çift, önceden adı konmamış dünya↔ekonomi (11 hata). Bu çift, çiftlerin
  tamamı sıralandığı için görünür oldu.
- Sıralar kararsız: farklar 1–2 satır ve beraberlikler var.

**Yüksek güvenli hatalar paydayla okunmalı.**
- Güven eşiği (kalibre güven ≥ 0.90) iki modelde farklı ölçekli olasılıklara uygulandığı için
  modeller arasında karşılaştırılmaz.
- Model içinde: BERT'in tahminlerinin %93'ü bu bantta, hatalarının ise %68'i. Hatalar bant
  dışında yoğunlaşıyor; güven bilgi taşıyor. char_lr'de aynı oranlar %79 ve %24.

**Olası etiket gürültüsü adayları (hipotez).**
- 14 satırda iki model birlikte yüksek güvenle yanılıyor ve aynı yanlış etikete gidiyor. Ortak
  hatalardaki taban oran hesaplanmadı.
- 247 çok satırlı yakın-tekrar grubunun 8'inde farklı etiketli satırlar var.
- Etiketlere hüküm verilmedi.

**Bozuk tırnaklar (C1) bir karıştırıcı değişken gibi davranıyor.**
- C1 içeren test satırlarında hata oranı farkı:
  - char_lr: +0.056 [+0.015, +0.100].
  - C1'i silen char_lr_c1clean: +0.070 [+0.027, +0.116].
  - BERT (C1'i görmüyor): +0.028 [−0.009, +0.067], CI 0'ı içeriyor.
- C1 silindikten sonra da fark sürdüğü için C1, kaynak ya da konu farkının bir işareti
  olabilir. Yüksek hata tek başına artefakta atfedilmez.
- char_lr ile char_lr_c1clean'in ayrıştığı 13 satırın 8'i C1 içeriyor (%62; testte %34).
  Ama C1'siz 5 satırda da ayrışma var, çünkü model yeniden eğitildi. Sayılar çok küçük.

**Çok hücre uyarısı.** Onlarca hücrede ayrı CI var ve çoklu karşılaştırma düzeltmesi
yapılmadı. Bazı CI'ların 0'ı dışlaması şans eseri de beklenir.

Ayrıntı: [`docs/faz7_sonuclar.md`](docs/faz7_sonuclar.md),
[`docs/faz7_protokol.md`](docs/faz7_protokol.md).

## Yöntem
### Veri ve split
- **Kaynak:** Hugging Face `savasy/ttc4900`, sabit commit SHA ile yükleniyor. 4900 haber,
  7 sınıf: siyaset, dünya, ekonomi, kültür, sağlık, spor, teknoloji.
- **Temizlik:** tam tekrarlar çıkarıldı, 4538 haber kaldı.
- **Yakın tekrarlar:** TF-IDF kosinüsü > 0.8 olan çiftler bir graf oluşturur. Grafın bağlı
  bileşenleri grup sayılır ve bir grup asla train ile test arasında bölünmez
  (`StratifiedGroupKFold`).
  - Resmî split `split_v2`: train 3630, test 908.
  - **Teşhis (betimsel):** düz stratified bir split'te (`split_v1`), train'de yakın kopyası
    bulunan 93 test satırında char_lr'nin accuracy'si diğer 815 satıra göre 5.7 puan yüksekti
    (%95 CI [+1.3, +9.4]). Bu alt küme rastgele seçilmedi ve sınıf bileşimi farklı; bu yüzden
    fark yalnızca ezberlemeye atfedilemez. split_v2'ye geçme gerekçesiyle tutarlıdır.
- **CV:** bütün CV'ler aynı grupları kullanıyor (`make_cv`). Düz `KFold` yakın tekrarları
  fold'lara bölerek CV skorunu şişirebilir (bu projede yalnız test düzeyinde betimsel olarak
  gözlendi, CV düzeyinde ölçülmedi).

### Modeller
| Faz | Model | Seçim |
|---|---|---|
| 2 | TF-IDF (karakter / kelime) + LR, LinearSVC, ComplementNB | Nested grup-bilinçli CV (5 × 3), Nadeau-Bengio düzeltmeli karşılaştırma |
| 3 | Donmuş encoder (BERTurk, multilingual-e5, bge-m3) embedding'i + LR | Aynı nested CV; teste yalnız bge-m3 taşındı |
| 4 | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuning | Önceden sabitlenmiş tek konfigürasyon; 5 fold × 3 seed CV; epoch sayısı CV'den |

### Test protokolü (test açılmadan donduruldu)
- **Testler:**
  - Her çift için exact McNemar.
  - Aile içinde Holm düzeltmesi. Birincil aile m = 1, ikincil aile m = 4.
  - macro-F1 farkı için grup düzeyinde eşleştirilmiş bootstrap CI (B = 10000).
- **Karar kuralı:** *ayırt edildi* ⇔ şu üç koşulun hepsi sağlanırsa:
  1. Holm p < 0.05.
  2. CI 0'ı dışlıyor.
  3. McNemar ile CI'ın yönü aynı.
- **Testten önce sabitlenenler:** güç analizi, birincil modelin seed'i (42) ve sonuç
  cümlelerinin şablonları. Commit sırası `docs/faz5_sonuclar.md`'de.
- Kaynak: [`docs/test_protokol.md`](docs/test_protokol.md),
  [`src/ttc4900/test_plan.py`](src/ttc4900/test_plan.py).

## Sınırlılıklar
- **Kısayol öğrenme:** veri kümesinde etiketle ilişkili iki kaynak artefaktı var.
  - Bozuk tırnaklar: U+0091–U+0094 kontrol karakterleri, ekonomide %61, dünyada %5.
  - Ajans künyeleri: "cihan", "aa", "anka".
  - Karakter modeli bozuk tırnakları kullanıyor. Bu karakterler silinince test macro-F1'i
    0.9132'den 0.9037'ye düşüyor (fark +0.0095, CI [+0.0020, +0.0182]; betimsel). BERTurk
    tokenizer'ı bu karakterleri zaten siliyor.
  - Skorlar başka kaynaklardan gelen haberler için iyimser olabilir (dış kaynakta ölçülmedi).
  - Ayrıntı: [`docs/shortcut_learning.md`](docs/shortcut_learning.md).
- **Tek test seti:** 908 satır, tek bölme. Birincil iddia tek eğitilmiş modele koşulludur.
  Seed değişkenliği yalnızca betimsel verildi.
- **İyimser p:** McNemar, test içindeki yakın-tekrar gruplarını (108 satır) yok sayar; p
  iyimser olabilir. CI'lar marjinaldir.
- **Tek derleme:** veri tek bir derlemeden geliyor (dataset card: Kemik grubu, Türk çevrimiçi
  haber siteleri). Derleme dönemi card'da belirtilmiyor. Modelin güçlü özellikleri arasında
  belirli kişi ve kurum adları var (Faz 2 özellik denetimi); güncel haberlerde bu işaretler
  değişebilir.
- **Kalibrasyon yalnız bu dağılım için geçerli.** T, CV modellerinden öğrenildi ve farklı
  eğitilen final modele uygulandı. Başka kaynaklardan gelen metinlerde olasılıklar kalibre
  olmayabilir.
- **Veri lisansı belirsiz.** Dataset card'da `license: unknown`. Veri Kemik grubundan
  (Yıldız Teknik Üniversitesi), kaynağı Türk haber siteleri. Atıf:
  <https://www.kaggle.com/savasy/ttc4900> ve Yıldırım & Yıldız (2018), Pamukkale Univ Muh
  Bilim Derg 24(5), 879–886, doi:10.5505/pajes.2018.15931.

## Yeniden üretme
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python -m pytest
```
- Faz 4 eğitimi Colab GPU'da koşuldu: [`notebooks/faz4_colab.ipynb`](notebooks/faz4_colab.ipynb),
  [`docs/faz4_protokol.md`](docs/faz4_protokol.md).
- Final değerlendirme adımları: `python -m ttc4900.final_eval {verify, predict, evaluate,
  v1-diagnosis, exploratory}` ([`docs/faz5_protokol.md`](docs/faz5_protokol.md)).
- Kalibrasyon: `python -m ttc4900.calibration {train, folds, test}`
  ([`docs/faz6_protokol.md`](docs/faz6_protokol.md)).
- Yerel demo (Gradio, ağ yok): `python space/app.py --model-dir outputs/models/42
  --calibration outputs/publish/model/calibration.json`, ardından `http://127.0.0.1:7860`
  ([`docs/faz8_protokol.md`](docs/faz8_protokol.md)).
- Hata analizi: `python -m ttc4900.error_analysis [bands]`
  ([`docs/faz7_protokol.md`](docs/faz7_protokol.md)).
- Bütün metrikler `outputs/metrics/*.json` altında. Her json, üretildiği commit'i ve kullandığı
  split'i kaydeder. Model ağırlıkları git'te değil.

## Depo düzeni
```
src/ttc4900/     yeniden kullanılabilir kod (veri, split, CV, modeller, testler, değerlendirme)
tests/           pytest
docs/            protokoller ve sonuç belgeleri
notebooks/       Colab eğitim defteri
outputs/         metrics/ (json + tahmin npz'leri), splits/, figures/
```

## Planlanan
- **Faz 8:** Hugging Face Hub'a model yükleme ve Gradio Space demosu
  ([`docs/faz8_protokol.md`](docs/faz8_protokol.md)).
  - Veri lisansı netleşene kadar model reposu private, demo Space'i public olacak.
  - Demo, seed 42 BERTurk modelini T = 1.283 ile kalibre edilmiş olasılıklarla gösterecek.
  - BERTurk bir dağıtım tercihidir: testte char_lr'den ayırt edilemedi.
  - Kalan kalibrasyon hatası (ECE 0.041), veri dağılımı sınırlılığı ve kısayol notları demo
    açıklamasında yer alacak.

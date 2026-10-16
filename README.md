# TTC4900 Türkçe Haber Sınıflandırma

> **Taslak.** Faz 5 (final test) sonrası yazıldı. Kalibrasyon (Faz 6), Hugging Face Hub ve
> Gradio demosu henüz yapılmadı; ilgili bölümler eklendikçe güncellenecek.

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
- **Tek kaynak ve dönem:** veri tek bir kaynaktan ve belirli bir dönemden geliyor. O dönemin
  kişi ve kurum adları güçlü özellikler.

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
- **Faz 6:** olasılık kalibrasyonu (temperature scaling, yalnız train içinde).
- Hugging Face Hub'a model yükleme ve Gradio Space demosu.

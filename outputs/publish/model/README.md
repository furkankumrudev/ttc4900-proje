---
language: tr
license: other
license_name: unknown-training-data
license_link: https://huggingface.co/datasets/savasy/ttc4900
base_model: dbmdz/bert-base-turkish-cased
datasets:
- savasy/ttc4900
pipeline_tag: text-classification
tags:
- turkish
- news-classification
- calibrated
---

# BERTurk, TTC4900 Türkçe haber sınıflandırma (7 sınıf)

`dbmdz/bert-base-turkish-cased` modelinin `savasy/ttc4900` üzerinde ince ayarlanmış hâli
(seed 42). Sınıflar: siyaset, dünya, ekonomi, kültür, sağlık, spor, teknoloji. Proje deposu,
protokoller ve tüm sonuç json'ları GitHub'dadır.

## Lisans
- Taban model `dbmdz/bert-base-turkish-cased`: MIT.
- Eğitim verisi `savasy/ttc4900`: dataset card'da `license: unknown`. Bu yüzden bu ağırlıklar
  için bir lisans beyan edilmedi (`license: other`). Lisans durumu netleşene kadar bu repo
  private tutulur.

## Veri ve atıf
Veri: Hugging Face `savasy/ttc4900` (revision `236e17b0c497ceb5437c175cb1ffaae201bfb287`). Dataset card'a göre veri Kemik grubundan (kemik.yildiz.edu.tr) alınmıştır; kaynağı Türk çevrimiçi haber siteleridir. Atıf: <https://www.kaggle.com/savasy/ttc4900>; Yıldırım, S. ve Yıldız, T. (2018), *A comparative analysis of text classification for Turkish language*, Pamukkale Univ Muh Bilim Derg 24(5), 879–886, doi:10.5505/pajes.2018.15931. Lisans: dataset card'da `license: unknown`.

- Tam tekrarlar temizlendi.
- Yakın tekrarlar (TF-IDF kosinüsü > 0.8) gruplandı ve hiçbir grup train ile test arasında
  bölünmedi.
- Bölme: train 3630, test 908.

## Sonuçlar (test, n = 908, %95 grup bootstrap CI)
| Model | Test macro-F1 | Test accuracy |
|---|---|---|
| Bu model (BERTurk, seed 42) | 0.9291 [0.9114, 0.9451] | 0.9273 [0.9095, 0.9440] |
| Karakter TF-IDF + LR (birincil baseline) | 0.9132 [0.8931, 0.9308] | 0.9130 [0.8933, 0.9310] |

**Birincil karşılaştırma (önceden kaydedilmiş): ayırt edilemedi.**
- macro-F1 farkı +0.0159, %95 CI [-0.0004, 0.0328].
- Exact McNemar n10/n01 = 38/25, p = 0.13.
- Testten önce yazılan güç analizi: MDE ≈ 0.025, güç ≈ 0.30.
- Bu sonuç iki modelin aynı olduğu anlamına gelmez; aynılık için önceden bir marj tanımlanmadı.
- Bu modelin demoda kullanılması bir dağıtım tercihidir, bir üstünlük iddiası değildir.

Üç seed'in test macro-F1'i: ortalama 0.9273 ± 0.0021 (std, betimsel).

## Olasılık kalibrasyonu
- Olasılıklar `softmax(logits / T)` ile hesaplanmalıdır: **T = 1.283**
  (`calibration.json`).
- T yalnızca train OOF tahminlerinden öğrenildi.
- Train içinde iyileşme gösterildi. Testte (önceden kaydedilmiş, tek
  sefer) **teyit edildi**: NLL 0.3380 → 0.2870.
- Kalan kalibrasyon hatası: test ECE (15 bin) 0.058 → 0.041.
  En yüksek güven bandında model hâlâ biraz aşırı emin.

## Sınırlılıklar
- **Kısayol öğrenme:** veri setinde etiketle ilişkili kaynak işaretleri var (bozuk tırnak
  karakterleri, ajans künyeleri). Skorlar başka kaynaklardan gelen metinler için iyimser
  olabilir; bu ölçülmedi.
- **Veri dağılımı:** model yalnızca bu veri setindeki haber metinleriyle eğitildi. Kalibrasyon
  da yalnızca bu dağılım için geçerlidir.
- **Tek eğitilmiş model ve tek test seti:** sonuçlar bu tek koşuya ve bu test setine
  koşulludur.
- **Girdi uzunluğu:** girdinin ilk 512 token'ından sonrası kullanılmaz.

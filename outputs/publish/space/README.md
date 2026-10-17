---
title: BERTurk TTC4900 Haber Sınıflandırma
emoji: 📰
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: other
models:
- <user>/berturk-ttc4900-haber
---

# BERTurk TTC4900 haber sınıflandırma demosu

Model yalnızca `savasy/ttc4900` veri setindeki haber metinleriyle ince ayarlandı. Başka türden ya da başka kaynaklardan gelen metinlerde tahminler ve olasılıklar güvenilir olmayabilir. Ajans adları ve tırnak biçimi gibi konu dışı işaretler tahmini etkileyebilir. Metnin ilk 512 token'ından sonrası kullanılmaz.

- Olasılıklar T = 1.283 ile kalibre edildi (yalnız bu veri dağılımı için). Testte
  kalan ECE 0.041.
- BERTurk'ün kullanılması bir dağıtım tercihidir. Testte karakter TF-IDF + LR baseline'ından
  ayırt edilemedi (macro-F1 farkı +0.0159, %95 CI
  [-0.0004, 0.0328]).

Veri: Hugging Face `savasy/ttc4900` (revision `236e17b0c497ceb5437c175cb1ffaae201bfb287`). Dataset card'a göre veri Kemik grubundan (kemik.yildiz.edu.tr) alınmıştır; kaynağı Türk çevrimiçi haber siteleridir. Atıf: <https://www.kaggle.com/savasy/ttc4900>; Yıldırım, S. ve Yıldız, T. (2018), *A comparative analysis of text classification for Turkish language*, Pamukkale Univ Muh Bilim Derg 24(5), 879–886, doi:10.5505/pajes.2018.15931. Lisans: dataset card'da `license: unknown`.

---
description: Bir kavramı sezgi, formül, projedeki kod karşılığı ve numpy doğrulamasıyla anlatır
argument-hint: <konu, ör. macro-F1, TF-IDF, ECE, McNemar, bootstrap>
---

Şu konuyu açıkla: **$ARGUMENTS**

Okuyucu matematik mühendisliği 4. sınıf öğrencisi. Doğrusal cebir, olasılık ve istatistik biliyor,
data scientist olmak istiyor. Açıklama Türkçe olsun, kısa ve yoğun tut, gereksiz giriş yapma.

Şu dört bölümü sırayla yaz:

### 1. Sezgi
Kavram hangi problemi çözüyor? 2–4 cümle ve mümkünse bu projeden (TTC4900, 7 sınıf, Türkçe haber)
bir örnek ver.

### 2. Formül
Tanımı LaTeX ile yaz ve her sembolü tanımla. Gerekirse kısa bir türetme ya da önemli bir özellik ekle
(ör. sınırlar, hangi durumda neye indirgendiği). Yaygın tuzakları belirt.

### 3. Projedeki kod karşılığı
`src/` ve `notebooks/` içinde kavramın geçtiği yeri bul ve `dosya:satır` formatında göster.
Formüldeki her terimin koddaki karşılığını eşleştir. Kod henüz yazılmadıysa bunu açıkça söyle ve
hangi fazda, hangi dosyada yer alacağını öner. Var olmayan kodu varmış gibi gösterme.

### 4. Numpy doğrulaması
Elle seçilmiş küçük bir örnekle kavramı saf numpy ile hesaplayan, kendi başına çalışan kısa bir
Python kodu yaz. Uygun bir kütüphane fonksiyonu varsa (sklearn, scipy) sonucu onunla karşılaştır
(`np.isclose`). Kodu proje ortamında (`.venv`) çalıştır ve çıktısını göster. Sonuçlar uyuşmazsa
nedenini açıkla.

Son olarak bu kavramla ilgili mülakatta sorulabilecek bir soru ve kısa cevabını ekle.

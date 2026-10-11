---
description: Bir deneyi çalıştırır, metrikleri outputs/metrics/ altına json olarak kaydeder ve kısa yorum yazar
argument-hint: <deney adı ve parametreler, ör. tfidf_lr C=1.0>
---

Şu deneyi çalıştır: **$ARGUMENTS**

## Adımlar

1. **Kodu bul**: Deney `src/ttc4900/` içinde tanımlı mı, kontrol et. Tanımlı değilse geçici bir
   betik yazıp çalıştırma. Önce deney kodunu `src/ttc4900/` altına nasıl ekleyeceğini öner ve
   onay bekle.
2. **Kuralları doğrula** (`CLAUDE.md`):
   - Veri `load_train_test()` ile resmî `outputs/splits/split_v2.json` dosyasından okunuyor.
   - CV `make_cv(train)` ile kuruluyor ve `groups=g` veriliyor.
   - Ön-işleme sadece train (ya da train fold'u) üzerinde fit ediliyor.
   - Test seti seçim veya ayar için kullanılmıyor; final değerlendirme fazından önce hiç
     kullanılmıyor (o fazlarda `test` alanı `null`).
3. **Çalıştır**: Proje ortamıyla çalıştır (Windows'ta `.venv\Scripts\python`). Seed'i sabitle.
   Süreyi ölç.
4. **Kaydet**: Sonucu `outputs/metrics/<deney_adı>_<YYYYMMDD-HHMMSS>.json` dosyasına şu şemayla yaz:

```json
{
  "experiment": "tfidf_lr",
  "timestamp": "2026-10-11T14:30:00",
  "git_commit": "<git rev-parse --short HEAD, kirli çalışma ağacıysa '-dirty' eki>",
  "split": "split_v2",
  "seed": 42,
  "params": {},
  "cv": {"n_folds": 5, "macro_f1_folds": [], "macro_f1_mean": 0.0, "macro_f1_std": 0.0},
  "test": {
    "macro_f1": 0.0,
    "accuracy": 0.0,
    "per_class_f1": {"<sınıf>": 0.0}
  },
  "runtime_seconds": 0.0,
  "notes": ""
}
```

   Bir alan bu deney için anlamlı değilse (ör. BERT'te CV yoksa) `null` yaz, alanı silme.
   Sonraki istatistiksel testler için test seti tahminlerini `outputs/metrics/preds/<aynı_ad>.npz`
   dosyasına kaydet: `y_true`, `y_pred` ve varsa `proba`.

5. **Yorumla** (3–5 cümle, Türkçe):
   - Ana sonuç: CV macro-F1 ortalama ± std ve test macro-F1.
   - `outputs/metrics/` içindeki önceki deneylerle karşılaştırma. Fark CV std'sinden küçükse bunu
     söyle; anlamlılık iddiası için bootstrap veya McNemar gerektiğini belirt.
   - En zayıf sınıf veya sınıflar ve olası nedeni.
   - Bir sonraki mantıklı deney önerisi.

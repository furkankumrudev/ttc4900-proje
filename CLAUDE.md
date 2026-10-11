# TTC4900 Türkçe Haber Sınıflandırma

Portfolyo projesi. Klasik ML yöntemlerini (TF-IDF + Logistic Regression, karakter n-gram, lineer SVM)
BERTurk fine-tuning ile karşılaştırır. Sonuçlar istatistiksel olarak sınanır, modeller kalibre edilir,
en iyi model Hugging Face Hub'a yüklenir ve bir Gradio Space ile sunulur.

Proje sahibi matematik mühendisliği 4. sınıf öğrencisi ve data scientist olmayı hedefliyor.
Amaç hem öğrenmek hem portfolyo oluşturmak.

## Veri

- Kaynak: Hugging Face `savasy/ttc4900`
- 4900 örnek, 7 sınıf, sütunlar: `category` (etiket), `text`
- Sadece `train` split'i var; test ayrımını biz yapıyoruz.
- `load_dataset` her zaman `revision=<commit SHA>` ile çağrılır. SHA, `ttc4900.data` içinde
  tek bir sabit olarak tutulur.
- Ham veride sınıflar tam dengeli (700/sınıf). Tam tekrar temizliğinden sonra 4538 örnek kalır
  (sınıf başına 566–690). Resmî split `split_v2`: train 3630, test 908 (test oranı 0.2001,
  sınıf başına 114–138). Ayrıntı: `outputs/data_report.json`.

## Kurallar

### Split ve sızıntı
- **Resmî split: `outputs/splits/split_v2.json`** (grup-bilinçli). Yakın tekrarlar (TF-IDF
  kosinüs > 0.8) bir graf oluşturur. Bu grafın bağlı bileşenleri grup sayılır ve hiçbir grup
  bölünmez. Split, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` ile
  üretilir ve fold 0 test olur. Test oranı ≈ 0.2'dir, tam değil.
  - Neden v2: düz stratified `split_v1`'de aynı haberin yeniden yazılmış sürümleri train ile
    test arasına bölünmüştü. 0.9 eşiğinde bu 88 çift ve 78 test satırıydı (testin %8.6'sı).
    Model bu satırları genellemeden, ezberleyerek doğru bilebiliyordu. v2'de bölünmüş çift
    sayısı 0. Eşik başta 0.9'du. (0.8, 0.9] aralığında da aynı haberin sürümleri bulununca
    0.8'e indirildi. En büyük grup yine 5 satır, stratifikasyon bozulmuyor.
  - `split_v1.json` silinmez ve değiştirilmez. Yalnızca Faz 2'deki tek "sızıntı etkisi"
    baseline'ı için kullanılır. Ana sonuçlar v1'de raporlanmaz.
- Split **bir kez** üretilir ve dosyaya yazılır. Bundan sonra tüm modeller split'i yalnızca bu
  dosyadan okur. Böylece hepsi aynı test örnekleri üzerinde değerlendirilir; McNemar ve paired
  bootstrap gibi eşleştirilmiş testler bunu gerektirir. Her `outputs/metrics/*.json` dosyası
  hangi split'i kullandığını `"split": "split_v2"` alanıyla belirtir.
- Tekrar eden (ve neredeyse aynı) metinler split'ten **önce** kontrol edilir.
- **CV her zaman `ttc4900.data.make_cv(train)` ile yapılır.** Bu fonksiyon
  `StratifiedGroupKFold` ile split_v2 gruplarını döndürür. `groups=g` parametresi `cross_validate`,
  `GridSearchCV.fit` ve `cross_val_predict` çağrılarına verilir. Düz `StratifiedKFold`, `KFold`
  ya da `cv=5` kullanılmaz; bunlar yakın tekrarları fold'lara böler ve CV skorunu sessizce
  şişirir. Train'den ayrılacak bir validation parçası (BERT early stopping, temperature scaling)
  da `train_test_split` ile değil, `make_cv` fold'larından biriyle ayrılır.
- **Test seti yalnızca final değerlendirmede, bir kez kullanılır.** Hiperparametre araması, model
  seçimi, early stopping (`Trainer`'ın `eval_dataset`'i ve `load_best_model_at_end` dahil),
  cross-validation ve temperature scaling **yalnızca train içinde** yapılır (CV fold'ları ya da
  train'den ayrılan validation parçası).
- Train/test verisi yalnızca `ttc4900.data.load_train_test()` ile okunur (varsayılan v2).
  Başka hiçbir yerde `train_test_split` ile ayrım yeniden yapılmaz.
- TF-IDF gibi veriden öğrenen dönüşümler `sklearn.pipeline.Pipeline` içinde, sadece train
  fold'unda fit edilir.

### Metrik ve istatistik
- Ana metrik: **macro-F1** (`f1_score(..., average="macro")`). Accuracy ve sınıf bazlı F1 yanında
  raporlanır.
- Karşılaştırmalar nokta tahminle bırakılmaz: CV için ortalama ± std, test için bootstrap güven
  aralığı, model çiftleri için McNemar testi kullanılır.
- BERTurk sonuçları birden fazla seed ile raporlanır.

### Kod düzeni
- Tüm yeniden kullanılabilir kod `src/ttc4900/` altına yazılır. Notebook'lar sadece bu kodu çağırır.
- Deney çıktıları `outputs/metrics/*.json`, grafikler `outputs/figures/` altına gider.
  Model dosyaları `outputs/models/` altına gider ve git'e girmez.
- Bağımlılıkların tek kaynağı `requirements.txt`. `pyproject.toml` sadece ruff ve pytest ayarlarını içerir.
- Kod hem yerelde (Windows, Python 3.12, `.venv`) hem Colab'da çalışmalıdır. Yol (path) bilgisi
  kod içine sabit yazılmaz; proje köküne göre çözülür.
- Her faz bitince commit atılır.

### Açıklama tarzı
- Her adım kısaca açıklanır: sezgi → formül → koddaki karşılığı.
- Matematiksel terimler doğru kullanılır; okuyucunun doğrusal cebir, olasılık ve istatistik
  bildiği varsayılır.

## Ortam

`pyproject.toml` paket tanımı içermediği için `ttc4900` paketi `src/` dizini yola eklenerek import edilir.

Yerel (Windows):
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
# .venv içinde src/'yi yola ekleyen bir .pth dosyası var (ttc4900_src.pth)
.venv\Scripts\python -m pytest
```

Colab:
```python
!git clone <repo-url> ttc4900-proje
%cd ttc4900-proje
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, "src")
```

pytest, `pyproject.toml` içindeki `pythonpath = ["src"]` ayarı sayesinde paketi kendisi bulur.

## Claude Code yardımcıları

- `/acikla <konu>`: kavramı sezgi, formül, projedeki kod ve numpy doğrulamasıyla anlatır.
- `/deney <isim>`: deneyi çalıştırır, metrikleri `outputs/metrics/` altına json olarak yazar ve yorumlar.
- `leakage-reviewer` agent'ı: veri sızıntısı denetimi yapar (salt okunur).
- `stats-reviewer` agent'ı: istatistiksel karşılaştırma denetimi yapar (salt okunur).
- PostToolUse hook'u: Edit veya Write ile değişen `.py` dosyasını `ruff format` ile formatlar.

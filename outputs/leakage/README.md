# Hedef sızıntısı denetimi (Faz 2 final modelleri, yalnızca train)

Üretim: `python -m ttc4900.feature_audit`. Faz 2 final konfigürasyonları (char_lr: C=100,
char_wb (2,4); word_lr: C=100, (1,1)) tüm train'de (3630 satır) fit edildi ve her sınıf için
`coef_`'in en büyük 20 bileşeni çıkarıldı. Test seti belleğe alınmadı (`load_train`).
**Yalnızca rapor: model, veri ve split değiştirilmedi.**

| Dosya | İçerik |
|---|---|
| `top_features_char_lr.csv`, `top_features_word_lr.csv` | 7 × 20 özellik: katsayı, sınıf içi/dışı belge frekansı, sınır payı, işaretler, char için örnek kelimeler |
| `artifact_by_class.csv` | Sınıf başına kaynak kelimesi ve C1 artefaktı içeren belge oranı |
| `boundary_ngrams.csv` | Sınıf başına en sık ilk/son 3 kelime (şablon arama) |
| `feature_audit.json` | Kurallar, işaret sayıları, işaretli özellikler, `artifact_by_class` |

Kural listesi ilk koşudan sonra bir kez genişletildi: `cihan` kaynak listesine eklendi ve
`encoding_artifact` kuralı yazıldı. İkisi de ilk koşunun top-20 listesine bakılarak fark edildi.
Kurallar yalnızca işaret koyar, model seçimini etkilemez.

## Bulgular

### 1. Kodlama artefaktı: C1 kontrol karakterleri (şüpheli, en güçlü bulgu)
Metinlerde cp1252 "akıllı tırnak"ları (‘ ’ “ ”) yanlış çözülmüş ve U+0091–U+0094 kontrol
karakterleri olarak kalmış. Train'deki 15 753 C1 karakterinin 15 619'u bu dört karakter
(1157 belge). U+0095–U+009F (• – ™ …) yalnızca 66 belgede, U+0080–U+0090 1 belgede geçiyor.
Aşağıdaki oranlar ve `c1_chars` U+0080–U+009F aralığının tamamını sayıyor; duyarlılık analizi
(`char_lr_c1clean`) yalnızca U+0091–U+0094'ü siliyor. Bu karakterlerin geçtiği belge oranı
sınıfa göre çok değişiyor:

| dunya | ekonomi | kultur | saglik | siyaset | spor | teknoloji |
|---|---|---|---|---|---|---|
| 0.046 | **0.611** | **0.529** | 0.302 | 0.170 | 0.323 | 0.309 |

- **char_lr:** ekonominin 1., 2. ve 3. özellikleri (`" \x92 "` vb., katsayı ≈ 7.6) ve 13.–15.
  özellikleri bu karakterler. Bunlar modeldeki en büyük katsayılar arasında.
- **word_lr etkilenmez:** `TfidfVectorizer`'ın varsayılan `token_pattern`'i (`\b\w\w+\b`) bu
  karakterleri token saymaz.
- Yorum: karakter konuyla ilgili değil. Hangi kaynaktan ve hangi kazıma hattından geldiğini
  gösteriyor. Ekonomi ile kültür ayrımında model kısmen bu artefakta dayanıyor olabilir.
  Bu bir hipotez, ölçülmedi. char_lr ile word_lr arasındaki CV farkının (+0.0076, N-B+Holm
  p=0.32) bir kısmı bu artefakttan gelebilir.
- Faz 3/4 için (Faz 3'te ölçüldü, `phase3_lengths_20261012-075659.json` `c1_probe`):
  BERTurk tokenizer'ı kontrol karakterlerini siler, yani bu sinyali **görmez**. XLM-R tabanlı
  tokenizer'lar (multilingual-e5, bge-m3) ise bu karakterlere gerçek token id'leri verir ve
  sinyali **görür**. Bu yüzden encoder'lar arası ve char_lr'ye karşı karşılaştırmalar artefakta
  erişim farkını içerebilir. Aynı asimetri Faz 4'ün birincil karşılaştırmasında da var:
  BERTurk fine-tuning C1'i görmez, ham char_lr görür, bu da char_lr lehinedir.

### 2. Ajans / kaynak imzaları (şüpheli)
| Özellik | Model, sınıf, sıra | Sınıf içi / dışı df | Not |
|---|---|---|---|
| `cihan` (Cihan Haber Ajansı) | word_lr dunya **#1** (8.23); char_lr dunya #2, #6, #9, #10 (`ihan`, `han `, `cih`, `ciha`) | 108 / 89 | dunya belgelerinin %20'sinde geçiyor, diğer sınıflarda en çok %8 |
| `aa` (Anadolu Ajansı) | word_lr dunya #2 (7.83); char_lr dunya #3, #4, #5, #15 | 137 / 238 | dunya %25, diğerleri %2–16 |
| `anka` (ANKA ajansı) | word_lr siyaset #13 | 64 / 15 | siyaset %11.6, diğerleri ≤ %2.4 |
| `bbc` | word_lr dunya #11 | 14 / 16 | |
| `com` | word_lr teknoloji #20 | 55 / 99 | `hurriyet com tr` künyesi (ekonomi son 3 kelime, %2.5) |

Ajans adları çoğunlukla metnin sonunda künye olarak geçiyor (`... dile getirdi cihan`).
Hangi ajansın kullanıldığı kategoriyle ilişkili; bu bir veri kümesi artefaktı. Başka bir
kaynaktan gelen haberlerde bu sinyal olmaz.

Yanlış pozitif: char_lr teknoloji #11 `" ap"`. Örnek kelimesi `apple` / `app_store`; AP ajansı
değil.

### 3. Kategori adı (büyük ölçüde meşru)
`spor`, `sağlık`, `sağlık_bakanlığı`, `sağlıklı`, `kültür` ve bunların char n-gram'ları
(`spor`, ` spo`, `sağl`, `ğlık`, …) işaretlendi. Bu kelimelerin sınırda geçme payı düşük
(≤ %15). Yani başlık ya da etiket gibi sabit bir yere eklenmiş değiller; metnin içinde doğal
geçiyorlar (ör. `trabzonspor`, `sağlık bakanlığı`). Etiketin kendisinin metne sızdığına dair
bir işaret görülmedi.

### 4. Şablon kalıbı (görülmedi)
`template` kuralı (sınıf belgelerinin ≥ %30'unda ilk/son 5 kelimede geçme) hiçbir özellikte
tetiklenmedi. En sık sınır 3-gram'ı sınıfının en çok %3.4'ünde geçiyor (ekonomi
`diye konuştu`). Sabit bir başlık ya da bitiş şablonu yok.

### 5. Tek sınıfa yığılan özellikler (meşru)
`maçında`, `maç`, `gol`, `kılıçdaroğlu`, `mhp`, `bdp`, `yönetmen`, `microsoft` gibi
özellikler tek bir sınıfta yoğunlaşıyor. Bunlar konu kelimeleri, artefakt değil.

## Sonuç
Etiketin kendisi metne sızmamış. Buna karşılık iki **kaynak artefaktı** sınıfla güçlü ilişkili
ve modeller bunları kullanıyor:
1. C1 kodlama karakterleri (yalnızca char_lr kullanıyor),
2. ajans künyeleri (`cihan`, `aa`, `anka`; iki model de kullanıyor).

Bunlar train ve test'te aynı dağılımla bulunduğundan test skorunu geçersiz kılmaz. Ancak
"genelleme" iddiasını zayıflatır ve modeller arası karşılaştırmayı (char ile kelime, klasik ile
transformer) çarpıtabilir.

Düzeltme yapılmadı. Teşhis deneyi (kural CV'den önce dondurulduktan sonra koşuldu,
`phase2_c1_sensitivity_20261012-080120.json`): yalnızca U+0091–U+0094 silinmiş char_lr,
aynı grid ve protokolle 0.9009 ± 0.0115 (ham 0.9054). Ham − temiz = +0.0045, N-B %95 CI
[−0.0002, +0.0091], p = 0.057. Birincil baseline ham char_lr olarak kalır. Ayrıntı ve demo
sınırlılıkları: `docs/shortcut_learning.md`.

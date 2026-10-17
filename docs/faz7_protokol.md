# Faz 7 protokolü: betimsel hata analizi

**Kaynak:** `src/ttc4900/error_analysis.py` (`ERROR_RULES`). Bu belge ile kaynak çelişirse kaynak
geçerlidir. Kurallar 7a commit'inde, analiz koşulmadan donduruldu. Komut:
`python -m ttc4900.error_analysis` (7b, temiz commit'te).

## Amaç ve sınırlar
- Amaç, modellerin testte **nerede ve nasıl** yanıldığını betimlemektir.
- **Tamamen betimseldir.** p hesaplanmaz, karar kuralı yoktur.
- Hiçbir model, ayar ya da Faz 1–6 kuralı değişmez. Bu analizden yola çıkarak model seçilmez,
  yeniden eğitilmez, eşik ayarlanmaz.
- Faz 5'in karşılaştırma sonuçları ve Faz 6'nın kalibrasyon kararları olduğu gibi kalır.

## Test erişimi (`ERROR_RULES['access']`)
- `load_train_test` için kod düzeyinde bir AST kilidi var (`tests/test_final_eval.py`). Bu kilit
  yalnızca bu modül için, kullanıcı onayıyla açıldı.
- Modül **model fit etmez ve tahmin üretmez**. Bir AST testi `fit`, `predict`,
  `predict_proba`, `decision_function`, `make_frozen` gibi çağrıların modülde olmadığını
  denetler.
- **Okunanlar:**
  - Faz 5 test npz'leri: `berturk_ft`, `char_lr`, `char_lr_c1clean`. sha256'ları Faz 5
    manifest'iyle (`phase5_test_preds_20261016-171044.json`) doğrulanır; tutmazsa analiz durur.
    Bu kontrol test metni okunmadan önce yapılır.
  - 6b'nin T̂_final değerleri.
  - split_v2 bileşenleri.
  - Train OOF tahminleri: yalnızca karışık etiketli grupların train satırları için.
  - Test metni.
- **Test metninin kullanımı:** yalnızca C1 varlığını ve karakter uzunluğunu hesaplamak için.
- **Tutarlılık denetimleri:** npz satırları ve etiketleri metinle aynı olmalı, skorların
  argmax'ı kayıtlı `y_pred` ile aynı olmalı.

## Modeller
| Model | Kullanıldığı maddeler |
|---|---|
| `berturk_ft` (seed 42) | hepsi |
| `char_lr` | hepsi |
| `char_lr_c1clean` | yalnız C1 maddesi |

## Tanımlar
- **Kalibre güven:** max_k softmax(z / T)_k.
  - T, 6b json'undaki T̂_final: berturk_ft 1.283, char_lr 0.916 (6c'de kullanılanla aynı).
  - char_lr'de z = log(predict_proba).
- **Yüksek güvenli hata:** yanlış tahmin ve kalibre güven ≥ **0.90**. Duyarlılık için
  ≥ 0.95 de betimsel olarak verilir.
- **C1:** metinde U+0091–U+0094 karakterlerinden en az biri var. Tanım `baselines.C1_QUOTES`
  ile, yani char_lr_c1clean'in sildiği karakterlerle aynıdır.
- **Çok satırlı grup:** split_v2'deki bir yakın-tekrar bileşeni (TF-IDF kosinüsü > 0.8).
- **Karışık etiketli grup:** içinde etiketi farklı satırlar bulunan bileşen. Train ve test
  birlikte sayılır. Bu gruplar olası etiket gürültüsüdür; etiketlere hüküm verilmez ve yeniden
  etiketleme yapılmaz.

## Maddeler
1. **Test karışıklık matrisleri** (berturk_ft, char_lr): sayılar ve satır normalize oranlar.
   Figür: `outputs/figures/phase7_confusion.png`.
2. **Sınıf çiftleri:**
   - 21 sırasız çiftin **tümü**, simetrik karışma sayısına (a→b + b→a) göre sıralanır.
   - ekonomi↔teknoloji ve siyaset↔dünya, sıralamadaki yerleri neyse onunla raporlanır. Yalnız
     önceden adı konmuş çiftlere bakılmaz.
3. **Ayrışan satırlar:** BERT doğru/char_lr yanlış (n10) ve tersi (n01). Her yön için:
   - Gerçek sınıf ve yanılan modelin tahmini.
   - İki modelin kalibre güveni.
   - Karakter uzunluğu.
   - C1 varlığı.
   - Karşılaştırma için tüm test satırlarının değerleri de verilir.
4. **Yüksek güvenli hatalar** (model başına, 0.90 ve 0.95 eşikleri):
   - Sayı ve hatalar içindeki pay.
   - Gerçek→tahmin çiftleri.
   - Çok satırlı ve karışık etiketli grup üyeliği.
   - C1 varlığı.
   - Ayrıca iki modelin birlikte yüksek güvenle yanıldığı satırlar.
5. **Karışık etiketli gruplar:** her satır için taraf (train/test), etiket ve iki modelin
   tahmini.
   - Test satırlarında final modellerin tahmini kullanılır.
   - Train satırlarında OOF tahmini kullanılır: BERT seed 42 CV ve char_lr tekrar 0.
6. **C1 ve hatalar:**
   - C1 içeren ve içermeyen test satırlarında, model başına hata oranı ve farkı.
   - Aynı karşılaştırma sınıf içinde de yapılır, çünkü C1 sınıfla karışıktır (ekonomide %61,
     dünyada %5).
   - char_lr ile char_lr_c1clean'in ayrıştığı satırlarda C1 varlığı ve hangi modelin doğru
     olduğu.

## Belirsizlik (`ERROR_RULES['uncertainty']`)
- **Oranlar** için Faz 5 ile aynı grup bootstrap'i kullanılır: split_v2 test bileşenleri,
  B = 10000, seed 42, %95 yüzdelik.
  - Kod, accuracy CI'larının Faz 5 json'uyla birebir aynı çıktığını denetler.
  - Oran = Σ_g w_g k_g / Σ_g w_g n_g. Paydası 0 olan replikatlar düşer ve sayısı yazılır.
- **n < 20 olan hücrelerde CI verilmez**, yalnızca sayı yazılır. İki oranın farkında CI için
  iki hücrenin de n ≥ 20 olması gerekir.
- **Sayımlar** (karışıklık matrisi, sınıf çiftleri, ayrışan satırların sınıf dağılımı) için CI
  verilmez.
- p yok. CI'lar marjinaldir. Bu kadar çok hücrede bazı CI'ların 0'ı dışlaması şans eseri de
  beklenir; hiçbiri karar olarak okunmaz.

## Örnekler ve metin (`ERROR_RULES['examples']`, `['text']`)
- **Seçim:** her kategoriden en çok 2 row_id, deterministik bir kuralla.
  - Sıra: yanılan modelin kalibre güveni azalan; eşitlikte row_id artan.
  - Karışık etiketli test satırlarında row_id artan.
  - Elle seçim yok.
- **Sonuç json'una metin yazılmaz**, yalnızca row_id yazılır.
- **Sonuç belgesinde** (`docs/faz7_sonuclar.md`, 7c) kısaltılmış alıntılar verilir:
  - En çok 160 karakter (`excerpt`; C1 karakterleri görünür bir işaretle, `¤` ile gösterilir).
  - Kategori başına en çok 2 alıntı.
- **Veri kaynağı ve lisans:**
  - Kaynak: Hugging Face `savasy/ttc4900` (sabit revision `236e17b0…`).
  - Dataset card'a göre veri Kemik grubundan (Yıldız Teknik Üniversitesi) alınmış, kaynağı Türk
    haber siteleri.
  - Lisans: card'ın metadata'sında `license: unknown`, "Licensing Information" bölümü boş
    ("More Information Needed").
  - Bu yüzden belgede yalnız kısa alıntılar kullanılır ve kaynak belirtilir. Card'daki atıf
    isteği de belgeye eklenir: Kaggle sayfası ve Yıldırım & Yıldız (2018).

## Commit düzeni
1. **7a:** kurallar, kod, testler ve bu belge. Hiçbir analiz sayısı hesaplanmamış olur.
2. **7b:** temiz commit'te koşu; json ve figür, yorumsuz.
3. **7c:** yorum, README, leakage-reviewer ve stats-reviewer. Kullanıcı onayıyla başlar.

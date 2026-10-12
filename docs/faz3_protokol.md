# Faz 3 protokolü: dondurulmuş embedding + LogisticRegression

Bu kurallar **CV koşmadan önce** ayrı bir commit'le donduruldu. Kaynağı `src/ttc4900/frozen.py`
(`FAMILY`, `SPEC`, `RULES`) ve `src/ttc4900/embeddings.py` (`ENCODERS`) dosyalarıdır.
Dondurmadan önce yalnızca etiket kullanmayan ölçümler yapıldı:

- `outputs/metrics/phase3_lengths_20261012-075659.json`: token uzunlukları
- `outputs/metrics/phase3_benchmark_20261012-081009.json`: 64 metinlik CPU hız ölçümü

## Aile (m = 3)
| Ad | Model @ revision | Pooling | Önek | Max token |
|---|---|---|---|---|
| `berturk_mean` | `dbmdz/bert-base-turkish-cased` @ `b6e1de16` | mask'li ortalama + L2 | — | 512 |
| `e5_base` | `intfloat/multilingual-e5-base` @ `d1287505` | mask'li ortalama + L2 | `query: ` | 512 |
| `bge_m3` | `BAAI/bge-m3` @ `5617a9f6` | CLS + L2 | — | 512 |

**max_length = 512, baştan kesme.** Kesilen metin oranı %22–32 ve önceden belirlenen %40
parçalama eşiğinin altında.

**bge-m3 notu:** bge-m3'ün kendi sınırı **8192 token**dır. Üç model de BERT ailesinin sınırı olan
512 ile koşulur. Bu seçim iki nedene dayanır: modeller arası karşılaştırılabilirlik ve CPU
maliyeti. Bu yüzden bge-m3 bu fazda uzun metin avantajını kullanmaz. Metinlerin yaklaşık
%31'i 512 token'ı aşıyor.

## Girdi ve C1 asimetrisi
Encoder'lar **ham metinle** koşulur, yani Faz 2'nin birincil modeli char_lr ile aynı girdi
kullanılır. Bu girdide bir asimetri var:

- `e5_base` ve `bge_m3` XLM-R tokenizer kullanır. Bu tokenizer U+0091–U+0094 kodlama
  artefaktını gerçek token id'leriyle görür.
- `berturk_mean`'in tokenizer'ı bu karakterleri siler, yani BERTurk artefaktı görmez.

Ayrıntı için `docs/shortcut_learning.md` dosyasına bak.

## Protokol
- **Veri:** Embedding'ler yalnızca train için üretilir, test embedding'i yoktur. Embedding'ler
  `outputs/cache/` altına yazılır; bu klasör git'e girmez. Üretim devam ettirilebilir: 256
  satırlık parçalar halinde yazılır.
- **Model:** `Pipeline(StandardScaler, LogisticRegression)`. Scaler her fit'te yalnızca o anki
  train parçasında fit edilir.
- **Grid:** C ∈ {0.001, 0.01, 0.1, 1, 10}.
- **CV:** Faz 2 ile birebir aynı. `make_cv` ile 5 fold × 3 tekrar, iç CV 3 fold, skor
  macro-F1. Dış fold'lar char_lr'ninkilerle aynıdır ve bu npz dosyalarından doğrulanır.
- **Karşılaştırma:** Her encoder için fark, sabit referansa göre hesaplanır:
  d = encoder − char_lr. Tabloda üç sütun var: düz p (karar için kullanılmaz), Nadeau-Bengio
  p, Nadeau-Bengio + Holm. Holm ailesi 3 encoder'dan oluşur.
- **Grid kenarı:** Final seçim ya da dış fold seçimlerinin en az 8/15'i grid'in bir ucuna
  düşerse, grid o yönde bir dekad genişletilir ve model bir kez daha koşulur. İki koşunun
  sonucu da raporlanır.

## Teste taşıma
- Ham metinle CV ortalaması en yüksek encoder teste taşınır. char_lr'yi geçip geçmemesine
  bakılmaz.
- Testte bu encoder char_lr'ye karşı McNemar ve paired bootstrap ile karşılaştırılır. Bu
  karşılaştırma Faz 2'nin ikincil Holm ailesine eklenir ve aile 3'ten 4 karşılaştırmaya çıkar.
- Birincil aile (BERTurk fine-tuning vs char_lr) değişmez.
- Test sonucuna göre encoder, pooling ya da C değiştirilmez.

## Duyarlılık analizi (ikincil)
- En iyi encoder, U+0091–U+0094 silinmiş metinle bir kez daha embed edilir ve aynı protokolle
  koşulur.
- Tek karşılaştırma yapılır: ham − temiz, Nadeau-Bengio ile. Holm uygulanmaz.
- Sonuç model seçiminde **kullanılmaz** ve teste taşınmaz.

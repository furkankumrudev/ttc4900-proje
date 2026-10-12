# Kısayol öğrenme (shortcut learning) bulguları ve demo sınırlılıkları

*README'nin "Sınırlılıklar" bölümü için not. Kaynaklar: `outputs/leakage/README.md`,
`outputs/metrics/phase2_c1_sensitivity_20261012-080120.json`. Tüm sayılar yalnızca train
(3630 haber) üzerinden; test seti kullanılmadı.*

## Ne bulundu
Modeller haberin konusunun yanı sıra, etiketle birlikte gelen ama konuyla ilgisi olmayan iki
**kaynak artefaktını** da kullanıyor. Kısayol öğrenme budur: veri kümesinde etiketle ilişkili ama
gerçek dünyada genellemeyen bir ipucu.

### 1. Kodlama artefaktı (bozuk tırnaklar)
Haberlerdeki ‘ ’ “ ” tırnakları kazıma sırasında yanlış çözülmüş ve görünmez U+0091–U+0094
kontrol karakterlerine dönüşmüş. Bu karakterleri içeren belge oranı sınıfa göre çok değişiyor:
ekonomi %61, kültür %53, dünya %5.

- Karakter n-gram modeli (char_lr) bunları ekonominin en güçlü üç özelliği olarak kullanıyor.
- **Duyarlılık analizi:** aynı protokolle (nested grup-bilinçli CV, 5×3) yalnızca bu dört
  karakter silinince macro-F1 0.9054'ten 0.9009'a düşüyor.
  - Fark: +0.0045, Nadeau-Bengio %95 CI [−0.0002, +0.0091], p = 0.057.
  - Fark 0.05 düzeyinde ayırt edilemiyor, ama CI neredeyse tamamen pozitif tarafta.
  - Büyüklüğü, char_lr ile kelime TF-IDF + LR arasındaki CV farkının (+0.0076) yaklaşık %60'ı.
  - En çok ekonomi sınıfı etkileniyor: F1 farkı +0.017. Bu sayı betimseldir; belirsizliği
    verilmemiştir.
- Kelime tabanlı TF-IDF modelleri etkilenmiyor: tokenizer bu karakterleri token saymıyor.
- Transformer'lar farklı davranıyor (Faz 3'te tokenizer çıktısıyla ölçüldü):
  - BERTurk tokenizer'ı bu karakterleri siliyor.
  - XLM-R tabanlı tokenizer'larda (multilingual-e5, bge-m3) bu karakterlerin gerçek sözlük
    id'leri var. Bu modeller artefaktı görebilir.
  - Faz 3 duyarlılık analizi (bge-m3): temizliğin etkisi ayırt edilemedi.
    Ham 0.9072, temiz 0.9058; fark +0.0015, N-B CI [−0.0036, +0.0065], p = 0.54. Bu, etkinin
    olmadığını göstermez. Nokta tahmini char_lr'ninkinden (+0.0045) küçük, ama iki CI büyük
    ölçüde örtüşüyor ve aradaki fark test edilmedi. Bu yüzden "bge-m3 artefakta daha az
    dayanıyor" denemez.
  - Faz 4 notu: BERTurk fine-tuning C1'i görmez, ham char_lr görür. Birincil karşılaştırmada
    bu asimetri char_lr lehinedir.

### 2. Ajans künyeleri
Haberlerin sonundaki ajans adı sınıfla ilişkili:

| Künye | Nerede yoğun |
|---|---|
| `cihan` | dünya haberlerinin %20'si, diğer sınıflarda ≤ %8 |
| `aa` | dünya %25 |
| `anka` | siyaset %12, diğer sınıflarda ≤ %2 |

Kelime modelinde `cihan` dünya sınıfının **1 numaralı** özelliği. Hangi ajansın kullanıldığı
konudan değil, bu veri kümesini toplayan kaynağın alışkanlıklarından geliyor.

## Raporlanan skorlar için anlamı
- Artefaktlar train ve test'te aynı dağılımla bulunuyor. Bu yüzden test skoru bu veri kümesi
  için geçerli bir tahmin.
- Ama bu skor, başka kaynaklardan gelen haberlerdeki performansın **iyimser** bir tahminidir.
- Modeller arası karşılaştırmalar artefakta farklı erişimden etkilenebilir: karakter ve kelime
  modelleri farklı, BERTurk ve XLM-R tabanlı modeller farklı.
- Önceden dondurulmuş kural: birincil klasik baseline ham char_lr olarak kalır. Temizlenmiş
  sürüm teste yalnızca betimsel taşınır; ikisi arasında seçim yapılmaz.

## Demo (Gradio Space) için sınırlılık
- Kullanıcının yazdığı ya da yapıştırdığı metinde bozuk tırnaklar ve ajans künyesi genellikle
  bulunmaz. Demodaki doğruluk, raporlanan test skorundan düşük olabilir; özellikle ekonomi ve
  dünya sınıflarında.
- Tersi de geçerli: metne "cihan" ya da "aa" eklemek tahmini dünya sınıfına itebilir. Bu bir
  modelleme hatası değil, verideki kısayolun görünür hale gelmesidir. Demo açıklamasında
  belirtilmeli.
- Veri kümesi tek bir kaynaktan ve belirli bir dönemden geliyor. Model, o dönemin gündemine
  ait kişi ve kurum adlarını da (ör. `kılıçdaroğlu`, `bdp`) güçlü özellik olarak kullanıyor.
  Güncel haberlerde bu sinyaller değişir.

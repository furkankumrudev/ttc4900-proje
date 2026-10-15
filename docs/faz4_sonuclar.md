# Faz 4 sonuçları: BERTurk fine-tuning (CV + final modeller)

**Kaynaklar:**
- Özet: `outputs/metrics/phase4_summary_20261015-214140.json`
- Seed 42 OOF: `phase4_berturk_ft_20261015-214140.json` ve `preds/cv_phase4_berturk_ft_20261015-214140.npz`
- 15 CV koşusu ve 3 final json'u, Colab ortam kaydı: `phase4_colab_env_20261015-160848.json`
- Protokol, eğitimden önce donduruldu: `docs/faz4_protokol.md`

Test seti açılmadı. Bu belgedeki sayılar train içinden gelir. Tek istisna, test boyutudur
(n_test = 908, split dosyasındaki test id'lerinin sayısı); test metni ya da etiketi okunmadı.

## Koşu koşulları
- Tüm koşular `faaba3e` commit'inden, Colab'da Tesla T4 üzerinde yapıldı.
  - Ortam: Python 3.13.15, torch 2.11.0+cu130, transformers 5.18.0, datasets 4.8.5,
    sklearn 1.6.1.
  - Kurulumda hiçbir paket yükseltilmedi; eksik paket olmadığı için hiçbir şey kurulmadı.
  - numpy için iki kayıt farklı: ortam kaydında (paket metadata'sı) 1.26.4, koşu json'larında
    (eğitim sürecindeki `np.__version__`) 2.1.3. Kaynağı netleştirilemedi. Kodun yüklediği
    sürüm koşu json'larındakidir; geçerli değer odur.
- Fold'lar `make_cv` ile yeniden üretilmedi.
  - Dış fold'lar char_lr `fold_id[0]`'dan, iç validation `outputs/splits/phase4_folds.json`'dan
    okundu.
  - Fold dosyasının içerik sha'sı `9c0c31e8…` ve 15 koşunun hepsinde aynı (`summarize` denetler).
  - Train parmak izi (row_id, etiket, metin) eğitimden önce doğrulandı.
- Durum özeti:
  - 15/15 CV koşusu tamamlandı; çöken koşu yok, en düşük best-val macro-F1 0.913.
  - 3/3 final model eğitildi; çöken yok.
  - Başarısız deneme kaydı ve geçersiz kılınan dosya yok.

## CV: fold × seed (OOF macro-F1)
Fold skoru, 3 seed'in ortalamasıdır. char_lr satırı, char_lr'nin aynı dış fold'lardaki
skorudur (repeat 0). d = BERT − char_lr.

| Fold | s42 | s43 | s44 | Fold skoru | Seed std | char_lr | d |
|---|---|---|---|---|---|---|---|
| 0 | 0.9064 | 0.8981 | 0.8946 | 0.8997 | 0.0061 | 0.8955 | +0.0042 |
| 1 | 0.9183 | 0.9262 | 0.9286 | 0.9243 | 0.0054 | 0.9179 | +0.0065 |
| 2 | 0.9308 | 0.9284 | 0.9269 | 0.9287 | 0.0020 | 0.9036 | +0.0251 |
| 3 | 0.9258 | 0.9140 | 0.9028 | 0.9142 | 0.0115 | 0.8986 | +0.0156 |
| 4 | 0.9127 | 0.9164 | 0.9232 | 0.9174 | 0.0053 | 0.9115 | +0.0060 |
| **Ort.** | 0.9188 | 0.9166 | 0.9152 | **0.9169** | | 0.9054 | **+0.0115** |

- **BERT CV macro-F1:** 0.9169. Accuracy 0.9159.
  - Fold std 0.0112 (ddof=1; bu bir CI değildir).
  - Nadeau-Bengio %95 CI [0.896, 0.938], oran 0.25. BERT'in gerçek eğitim boyutuyla oran 0.31
    alınırsa CI [0.895, 0.939] olur.
- **Seed std (eğitim değişkenliği):** her fold'da 3 seed, yani 2 serbestlik derecesi var. Bu
  yüzden tek fold'un seed std'si çok belirsizdir.
  - Havuzlanmış seed std: s_p = 0.0068 (df = 10). σ için %95 aralık [0.0047, 0.0119] (χ²,
    normal varsayımı).
  - Fold başına s'lerin ortalaması 0.0061. Bu değer σ'yı aşağı yanlı tahmin eder
    (E[s] = c4·σ, c4(3) ≈ 0.886).
  - Seed başına 5-fold ortalamaları 0.9188, 0.9166 ve 0.9152, std'leri 0.0018.
    - Bu değer, seed'in fold'lar arasında ortak bir ana etkisi olmasaydı beklenecek
      σ_p/√5 ≈ 0.0030'dan küçük. Ortak bir seed etkisine dair kanıt yok (df = 2, çok belirsiz).
    - Yine de `set_seed` her fold'da aynı başlattığı için seed'ler fold'lar arasında tam
      bağımsız değildir. df = 10 varsayımı bu yüzden iyimser olabilir ve σ aralığı biraz dar
      çıkabilir.
- **Sınıf F1 (seed ve fold ortalaması):**
  - En düşük: ekonomi 0.874, teknoloji 0.888, dünya 0.891.
  - En yüksek: spor 0.988.

## char_lr ile eşleşmiş karşılaştırma (keşifsel)
d_j = BERT_j (seed ortalaması) − char_lr_j (repeat 0), j = 1..5. Nadeau-Bengio düzeltmeli
t testi kullanıldı (J = 5, df = 4). Tek karşılaştırma olduğu için Holm uygulanmadı.

| Oran | d ort. | SE | %95 CI | t | p |
|---|---|---|---|---|---|
| 0.25 (char_lr tanımı, birincil) | +0.0115 | 0.0059 | [−0.0050, +0.0279] | 1.93 | 0.125 |
| 0.31 (BERT n_oof/n_fit, duyarlılık) | +0.0115 | 0.0063 | [−0.0061, +0.0290] | 1.81 | 0.144 |

**Okuma:**
- BERT 5 fold'un 5'inde de önde. Ortalama fark +1.15 puan.
- Fark fold'lar arasında heterojen: std_diff 0.0088, ortalamanın %77'si.
  - Fold 2 tek başına +2.5 puan.
  - Fold 2 olmadan ortalama +0.81 puan olur.
- Fold'lar arası train örtüşmesi düzeltilince CI 0'ı içeriyor. CV, iki prosedürü bu
  belirsizlikte **ayırt etmiyor**. Bu bir eşdeğerlik iddiası da değildir.
- Fold'ları bağımsız sayan düzeltmesiz t testi (json'da `p_naive`) yalnızca düzeltmenin
  etkisini göstermek için saklanır. Çıkarımda kullanılmaz.
- Fold skorlarının korelasyonu 0.64: zor fold iki model için de zor.
- Bu sonuç teste taşıma kararını etkilemez. Seed 42 final modeli, CV sonucundan bağımsız olarak
  teste gider (`RULES['test_transfer']`). Test iddiası Faz 5'tedir.

**Farkı etkileyen asimetriler (yönleri; büyüklükleri ölçülmedi):**
1. **Eğitim boyutu (CV'ye özgü; olası yön BERT aleyhine):**
   - CV'de BERT ortalama 2323 satırla eğitildi, çünkü dış train'den 581 satır early stopping
     için validation olarak ayrıldı. char_lr ise 2904 satırla eğitildi.
   - Final modellerde ikisi de 3630 satırla eğitilir.
   - Bu asimetri BERT'i ancak öğrenme eğrisi char_lr'ninkinden dikse dezavantajlı kılar. Bu
     varsayım ölçülmedi. "CV BERT için muhafazakâr" ifadesi bu yüzden bir hipotezdir.
2. **Model seçimi (kazananın laneti; yön BERT aleyhine):**
   - char_lr, 4 klasik adayın CV maksimumu, bu yüzden CV skoru yukarı yanlı.
   - BERT önceden belirlenmiş tek konfigürasyon, ayar taraması yok. Bu, d'yi aşağı çeker.
3. **Girdi (CV'ye özgü değil):** BERTurk tokenizer'ı U+0091–U+0094 kontrol karakterlerini
   siler, char_lr ise bunları görür (`docs/shortcut_learning.md`).
   - Bu karakterler testte de var. Bu yüzden CV'ye özgü bir haksızlık değil, iki modelin
     gördüğü girdinin farkıdır ve final testte de geçerli kalır.

Karşılaştırılan şey prosedürlerdir: char_lr'de iç ayarlama (nested CV grid) var, BERT'te ayar
taraması yok.

## Epoch'lar ve E_final
- **Best epoch dağılımı (15 koşu):** 1 epoch ×2, 2 ×2, 3 ×3, 4 ×5, 5 ×3.
  - **E_final = medyan = 4.**
  - 3 koşunun best epoch'u üst sınır olan 5'te. Bu koşularda daha uzun eğitim daha iyi
    olabilirdi (sağdan sansür).
  - Medyan sansürden etkilenmez: 15 koşunun 8. sıra istatistiğidir ve en fazla 3 koşu 5'te.
- **Best epoch'un gürültüsü:** komşu epoch'lar arasındaki val F1 farkları (~0.01), 581
  satırlık val'in örneklem hatası (~0.01) mertebesinde. Best epoch dağılımının büyük kısmı bu
  yüzden gürültüdür. E_final = 4 ile 3 ya da 5 arasındaki bilgi farkı küçüktür.
- **Erken durma:** 4/15 koşu 5 epoch'tan önce durdu (`frac_epochs_run_lt_max` = 0.27).
  - Bu oran erken durmanın tamamını saymaz. Patience 2 ile best epoch'u 3 olan koşular
    5. epoch'ta durur ve `epochs_run` = 5 görünür.
- **E_final'ın anlamı:** epoch biriminde aktarılır. Final'de epoch başına ~227 güncelleme
  yapılır (3630/16), CV'de ~146 (2323/16). lr eğrisinin biçimi aynıdır: 5 epoch üzerinden
  warmup + linear decay, eğitim 4. epoch sonunda kesilir (908/1135 adım).

## Final modeller (tüm train, validation yok)
| Seed | Epoch | Train in-sample macro-F1 | Çöktü mü | Birincil | Ağırlık sha256 (ilk 12) |
|---|---|---|---|---|---|
| 42 | 4 | 0.9943 | hayır | **evet** | `72cbeb438a89` |
| 43 | 4 | 0.9956 | hayır | hayır | `21a0cd72d6c8` |
| 44 | 4 | 0.9931 | hayır | hayır | `12b126d871e3` |

- **Çöküş kuralı:** `FINAL_RULES['final_collapse']` hiçbir modelde tetiklenmedi (eşik 0.5).
  - In-sample F1 iyimserdir. Yalnızca bariz çöküşü yakalamak içindir, performans tahmini değildir.
- **Birincil test modeli:** seed 42, `phase4_summary.finals.primary_model_dir`.
  - Ağırlıklar Drive'da (`faz4/models/42`); zip'te ve git'te yok.
  - Faz 5'te sha256 (`72cbeb43…`) doğrulanır.

## fp16 notu: fold 1, seed 43'te grad_norm = inf
- **Gözlem:** Colab çıktısında bu koşunun eğitim log'unda bir adımda `grad_norm = inf`
  göründü (kullanıcı gözlemi). Adım başına log json'a kaydedilmediği için adım numarası
  dosyalardan doğrulanamaz.
- **Kayıt sınırı:** log epoch başınadır (`logging_strategy="epoch"`). Loglanan `grad_norm`,
  yalnızca epoch'un son optimizer adımının normudur; yani inf bir epoch sonu adımında görüldü.
  Diğer adımlarda ve diğer 14 koşuda atlanan adım olup olmadığı kayıtlı değil. fp16 eğitiminin
  başında (ölçek 65536) birkaç adımın atlanması olağandır.
- **Muhtemel neden:** fp16'da dinamik kayıp ölçeklemesi (GradScaler) kullanılır. Ölçeklenmiş
  gradyan fp16 aralığını aşınca norm inf/nan olur. Scaler bu adımda optimizer güncellemesini
  atlar ve ölçeği küçültür.
  - Loglanan eğitim kaybı (epoch ortalaması) sonlu. Bu, ileri geçişin hiçbir adımda inf
    olmadığını gösterir; taşma geri geçişteki ölçeklenmiş gradyandadır.
- **Etkisi:** atlanan adımda parametreler güncellenmez. Trainer bu adımda lr scheduler'ı da
  ilerletmez (`optimizer_step_was_skipped` kontrolü). Bu kontrol Colab'daki transformers
  5.18.0'da (`trainer.py:1903`) ve yerel 5.19.0'da doğrulandı. Gözlenen en az bir atlanan adım
  var; toplam sayı bilinmiyor.
- **Koşunun kendisi normal:**
  - Kayıplar sonlu: train 1.24 → 0.25 → 0.12, val 0.30–0.33.
  - Val macro-F1: 1. epoch 0.921, 2. epoch 0.913, 3. epoch 0.918. Best epoch 1, patience 2 ile
    3. epoch'ta durdu.
  - OOF macro-F1 0.926. Fold 1'in diğer seed'leri 0.918 ve 0.929, yani aynı aralıkta.
  - Çöküş kuralının dışında (0.92 > 0.5) ve yeniden koşu gerektirmez. Kural önceden
    tanımlıydı ve tek istisna çöküştür.
- **Not:** best epoch'u 1 olan iki koşudan biri bu (diğeri fold 3, seed 44). Val F1'deki
  0.921 → 0.913 düşüşü gürültü düzeyindedir. Bu yüzden inf adımının bir etkisi, olağan bir
  koşudan ayırt edilemez.

## Birincil çift için güç (test öncesi, yalnız train OOF)
`paired_test --power`, BERT seed 42 CV OOF'u ile char_lr repeat 0 OOF'unu karşılaştırır. İkisi
aynı dış fold'lardadır (doğrulandı). n_test = 908, α = 0.05 (birincil aile, m = 1).

| Çift | psi | n10 / n01 (OOF) | OOF Δacc | OOF ΔmacroF1 | MDE acc | MDE macro-F1 (kaba) | OOF farkında güç |
|---|---|---|---|---|---|---|---|
| berturk_ft − char_lr | 0.068 | 148 / 99 | +0.0135 | +0.0134 | 0.025 | 0.025 | 0.30 |

- **psi:** iki modelden yalnızca birinin doğru olduğu satırların oranı. Vekil tahmin ≈ 0.07
  idi, gerçek değer 0.068 çıktı.
- **MDE:** 908 test örneğiyle %80 güçle ancak yaklaşık **2.5 puanlık** bir fark yakalanabilir.
- **Güç tek bir sayı değil, bir aralıktır.** Exact McNemar, psi = 0.068, α = 0.05:

  | Varsayılan gerçek Δacc | Kaynağı | Güç |
  |---|---|---|
  | +0.0135 | seed 42 OOF | 0.30 |
  | +0.0115 | 3 seed ortalaması | 0.23 |
  | +0.0046 / +0.0224 | seed 42 OOF farkının kaba %95 aralığının uçları (SE 0.0045) | 0.07 / 0.70 |

  - Seed 43 ve 44'ün OOF'ları aynı hesapla:
    - psi 0.067 ve 0.069, n10/n01 142/101 ve 143/108.
    - Δacc +0.0113 ve +0.0096; bu Δ'larda güç 0.22 ve 0.17.
    - psi seed'ler arasında kararlı, fark ise seed'e duyarlı.
  - Seed 42 sonuca bakılarak seçilmedi, eğitimden önce belirlenmişti. Seed seçim yanlılığı
    yok, yalnızca tek örneklemin oynaklığı var.
  - Karar kuralı iki koşulu birlikte istediği için (McNemar ve CI) ortak güç daha düşüktür.
    İki koşul güçlü korelasyonlu olduğu için kabaca 0.25–0.30.
  - "Ayırt edilemedi" olası bir sonuçtur ve eşdeğerlik kanıtı sayılmaz.
- **Bootstrap SE ile N-B'nin farkı:** seed 42 OOF farkı bootstrap SE'sine göre ~3 SE uzakta,
  N-B düzeltmeli CV CI'ı ise 0'ı içeriyor. Bu bir çelişki değil:
  - OOF bootstrap'i yalnızca satır örneklem belirsizliğini ölçer.
  - N-B ise fold'lar arası train örtüşmesini ve prosedürün (eğitimin) değişkenliğini de
    hesaba katar.
  - Bu yüzden bootstrap CI'ı CV'deki prosedür karşılaştırması için kanıt sayılmaz; yalnızca
    test gücünün ölçeği için kullanıldı.
- **Yanlılığın yönü (birincil çift):** kazananın laneti burada char_lr'nin CV skorunu
  şişirir; BERT önceden belirlenmiş tek konfigürasyondur. Bu yüzden OOF farkı aşağı yanlıdır.
  Öte yandan final model 3630 satırla eğitilir (OOF modeli 2323), bu da test farkını ve psi'yi
  değiştirebilir. Net yön belirsiz.
- **Yaklaşıklık:**
  - Birincil OOF tek tekrardır (R = 1).
  - Grup bootstrap SE'sinin satır bootstrap SE'sine oranı 1.04.

## Tekrarlanabilirlik
fp16 ve CUDA çekirdekleri nedeniyle aynı seed'le yeniden koşu bit düzeyinde aynı sonucu
vermeyebilir. Kayıtlı OOF tahminleri ve final ağırlıklar (sha256) tek doğruluk kaynağıdır.

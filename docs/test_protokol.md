# Final test protokolü

**Kaynak:** `src/ttc4900/test_plan.py` (`TEST_PLAN`, `FAMILIES`, `DESCRIPTIVE`). Kod:
`src/ttc4900/paired_test.py`. Bu belge ile kaynak çelişirse kaynak geçerlidir. Yöntem **test seti
açılmadan** donduruldu; test sonucuna göre değiştirilmez.

## Aileler
| Aile | m | Karşılaştırmalar (fark = a − b) |
|---|---|---|
| Birincil | 1 | `berturk_ft − char_lr` |
| İkincil | 4 | `char_lr − word_lr`, `char_lr − word_svc`, `char_lr − word_cnb`, `bge_m3 − char_lr` |
| Betimsel (aile dışı, p yok) | — | `char_lr − char_lr_c1clean`, `berturk_ft − bge_m3`, `berturk_ft_s43 − char_lr`, `berturk_ft_s44 − char_lr` |

- **Aileler kesin (Faz 5a, test açılmadan):** birincil m=1, ikincil m=4. Bundan sonra
  `FAMILIES` ve `DESCRIPTIVE` değişmez.
- **`berturk_ft − bge_m3`:** hiçbir Holm ailesine girmez. Yalnızca fark ve grup bootstrap CI'ı
  raporlanır; p hesaplanmaz.
  - İkincil aile büyümediği için α/m değişmez ve güç json'u yeniden üretilmez.
- **Seed 43 ve 44:** final modellerinin `char_lr`'ye karşı farkı da betimsel çifttir (CI, p yok).
  - Bu çiftlerin birleşik OOF'u olmadığı için güç tablosunda yer almazlar.
- **Birincil test modeli:** seed 42 ile eğitilen BERTurk final modeli. Bu seçim eğitimden önce
  yapıldı; test ya da validation skoruna göre değiştirilmez.
  - Tek önceden tanımlı istisna (`finetune.FINAL_RULES`, tüm eğitimden önce donduruldu):
    seed 42 final modeli train üzerinde çökmüş çıkarsa (in-sample macro-F1 < 0.5), seed 142
    ile bir kez yeniden eğitilir ve birincil model o olur.
  - Diğer seed'lerin test macro-F1'i betimsel raporlanır: ortalama ± std ve her biri için CI.
    p ve Holm yok.
  - Birincil iddia bu tek modele koşulludur. McNemar ve bootstrap test örneklemesinin
    belirsizliğini ölçer, seed'den (eğitimden) gelen belirsizliği ölçmez.

## Yöntem
- **p:** Her n için exact binom McNemar.
  - n10: A doğru, B yanlış. n01: A yanlış, B doğru.
  - H0 altında n10 ~ Binom(n01 + n10, ½); p iki yönlüdür.
  - Süreklilik düzeltmesi ve χ² yaklaşımı yok.
- **Çoklu karşılaştırma:**
  - Holm her aile içinde ayrı uygulanır ve yalnızca bu exact p'leri kullanır (α = 0.05).
  - Aileler ayrı ayrı α'da tutulur; aileler arası toplam hata kontrolü yok.
  - %95 CI'lar marjinaldir, eşzamanlı değildir.
- **Etki:** macro-F1 farkı (7 sınıf, zero_division=0), grup düzeyinde eşleştirilmiş bootstrap
  %95 CI ile.
  - Gruplar: split_v2 `components`, yani yakın-tekrar bileşenleri. Test kümesinde 850 grup var:
    800 tekil satır, 44 ikili, 5 üçlü ve 1 beşli grup.
  - B = 10000, `np.random.default_rng(42)`, yüzdelik yöntem.
  - Uygulama: her grubun karışıklık matrisi C_g bir kez hesaplanır. Bir replikat Σ w_g C_g'dir.
    Ağırlıklar w = `rng.multinomial(G, [1/G]*G)` ile 1000'lik parçalar halinde çekilir; gruplar
    `np.unique` sıralı id düzenindedir. Tüm modeller aynı w'leri kullanır.
- **Karar kuralı:** *Ayırt edildi* ⇔ şu üç koşulun hepsi sağlanırsa:
  1. Holm p < 0.05.
  2. macro-F1 farkının CI'ı 0'ı dışlıyor.
  3. Yönler tutarlı: sign(n10 − n01) = CI'ın işareti.

  Üçüncü koşul gerekli, çünkü McNemar accuracy'yi, CI ise macro-F1'i ölçer ve iki yön
  ayrışabilir. Ayrışırsa sonuç *ayırt edilemedi (yön çelişkisi)* olur. Herhangi bir koşul
  sağlanmazsa sonuç "ayırt edilemedi"dir ve hangi koşulların sağlandığı ayrıca yazılır.
  "Ayırt edilemedi" eşdeğerlik demek değildir; eşdeğerlik marjı tanımlanmadı.
- **Betimsel çift:** p hesaplanmaz; yalnızca skorlar ve fark CI'ı verilir
  (`paired_test.describe_pair`).

## Sınırlılık
- McNemar test satırlarını bağımsız sayar ve test içindeki yakın-tekrar grup bağımlılığını
  yok sayar. Bu yüzden p iyimser olabilir.
  - Testte bu bağımlılık 50 grupta, 908 satırın 108'inde var.
  - Etkinin küçük olması beklenir. Train OOF'ta grup bootstrap SE'sinin satır bootstrap SE'sine
    oranı 1.02–1.06 çıktı (macro-F1 farkı). Bu oran test için doğrudan ölçülmedi.
  - Bootstrap CI grupları hesaba katar.
- İki koşul farklı büyüklükleri ölçer: McNemar accuracy uyuşmazlıklarını, CI macro-F1 farkını.
  Kural koşulların kesişimi olduğu için gücü, her koşulun tek başına gücünden yüksek olamaz.

## Güç analizi
Kaynak: yalnızca train OOF tahminleri,
`outputs/metrics/test_power_20261015-215451.json` (commit `2f1ce27`, Faz 4c). Önceki sürüm
(`test_power_20261012-133032.json`, commit `489279d`) birincil çifti yalnızca vekillerle
veriyordu; `outputs/metrics/superseded/` altına taşındı. Ortak satırların sayıları aynı.

- **psi:** iki modelden yalnızca birinin doğru olduğu OOF satırlarının oranı.
  - Klasik ve encoder çiftlerinde 3 tekrarın ortalamasıdır.
  - Birincil çiftte tek tekrardır (R = 1): BERT seed 42 CV OOF'u ve char_lr repeat 0.
- **MDE:** n = 908 ile %80 güce ulaşan en küçük fark.
  - Accuracy MDE'si exact McNemar'dan hesaplanır. psi, alternatif hipotez altında sabit tutulur.
  - macro-F1 MDE'si kaba bir normal yaklaşımdır: (z₁₋α/₂ + z₀.₈) · SE. SE, OOF grup bootstrap
    SE'sinin √(3630/908) ile ölçeklenmiş hâlidir. Çarpan α = 0.05'te 2.80, α/4'te 3.34.
- **İkincil aile:** Holm eşiği α/4 ile α arasındadır. İki α'daki güç, gerçek gücün alt ve üst
  sınırıdır.

| Çift | psi | OOF Δacc | OOF ΔmacroF1 | MDE acc (α/4 ; α) | MDE macro-F1, kaba (α/4 ; α) | OOF farkında güç (α/4 ; α) |
|---|---|---|---|---|---|---|
| **berturk_ft − char_lr (birincil, yalnız α)** | 0.068 | +0.0135 | +0.0134 | — ; 0.025 | — ; 0.025 | — ; 0.30 |
| char_lr − word_lr | 0.052 | +0.0079 | +0.0076 | 0.026 ; 0.022 | 0.025 ; 0.021 | 0.05 ; 0.15 |
| char_lr − word_svc | 0.051 | +0.0037 | +0.0036 | 0.025 ; 0.022 | 0.025 ; 0.021 | 0.02 ; 0.06 |
| char_lr − word_cnb | 0.068 | +0.0105 | +0.0106 | 0.029 ; 0.025 | 0.030 ; 0.025 | 0.08 ; 0.19 |
| bge_m3 − char_lr | 0.074 | +0.0017 | +0.0019 | 0.031 ; 0.026 | 0.031 ; 0.026 | 0.01 ; 0.04 |
| char_lr − char_lr_c1clean (betimsel, yalnız CI) | 0.016 | +0.0047 | +0.0044 | — | — ; 0.011 | — |
| berturk_mean − char_lr (eski vekil, yalnız karşılaştırma) | 0.071 | −0.0051 | −0.0048 | 0.030 ; 0.026 | 0.030 ; 0.025 | 0.02 ; 0.07 |
| e5_base − char_lr (eski vekil, yalnız karşılaştırma) | 0.074 | −0.0023 | −0.0023 | 0.031 ; 0.026 | 0.031 ; 0.026 | 0.01 ; 0.04 |

**Okuma:**
- psi ≈ 0.05–0.075 olan çiftlerde 908 test örneğiyle ancak yaklaşık **2–3 puanlık** farklar %80
  güçle yakalanabilir.
- **İkincil aile:**
  - CV'de gözlenen farklar 0.2–1.1 puan. Bu farklar gerçekse ikincil karşılaştırmaların gücü
    α'da bile %20'nin altında.
  - Bu çiftlerde a tarafı (char_lr, bge_m3) adaylarının CV maksimumu. OOF farkları kazananın
    lanetiyle yukarı yanlı olabilir, bu yüzden gerçek güç daha da düşük olabilir.
  - "Ayırt edilemedi" beklenen sonuçtur ve eşdeğerlik kanıtı sayılmaz.
- **Birincil çift:** CV farkı +1.15 puan (3 seed), seed 42 OOF farkı +1.34 puan; MDE ≈ 2.5 puan.

**Birincil çift (BERTurk fine-tuning − char_lr), ayrıntı `docs/faz4_sonuclar.md`:**
- Güç, Faz 4'ün seed 42 CV OOF'u ile hesaplandı. Bu OOF, char_lr'nin 1. tekrar dış fold'larıyla
  aynı fold'larda üretildi (`phase4_berturk_ft_20261015-214140.json` + npz).
- **Vekil ile karşılaştırma:** vekillerin psi değeri ≈ 0.07 idi, gerçek değer 0.068 çıktı.
  n10/n01 = 148/99.
- **Güç tek sayı değil:** varsayılan gerçek farka göre değişir (exact McNemar, α = 0.05).
  - Seed 42 OOF farkında (+0.0135) güç 0.30.
  - 3 seed ortalamasında (+0.0115) 0.23.
  - OOF farkının kaba %95 aralığının uçlarında 0.07 ile 0.70.
  - Karar kuralının ortak gücü bundan düşüktür.
- **Yanlılığın yönü:**
  - Kazanan char_lr b tarafında, BERT önceden belirlenmiş tek konfigürasyon. Bu yüzden OOF farkı
    kazananın laneti açısından aşağı yanlıdır.
  - Seed 42 sonuca bakılarak seçilmedi; yalnızca tek seed'in oynaklığı var.

**Yaklaşıklık:**
- OOF modelleri ~2904 satırla (BERT fine-tuning'de validation ayrıldığı için ~2323), final
  modeller 3630 satırla eğitilir. Bu yüzden psi ve SE yaklaşıktır.
- Birincil OOF tek tekrar ve tek seed'dir.
- McNemar gücü test içi grup bağımlılığını yok sayar.

## Değişiklik notu
- `TEST_PLAN`, `cv_compare.py`'den `test_plan.py`'ye taşındı. `cv_compare` onu içe aktarır. Faz 2
  özetlerine gömülü eski metin tarihî kayıttır.
- Önceki metinde yalnızca "McNemar + paired bootstrap" yazıyordu. Şimdi sabitlenenler:
  - Her n'de exact McNemar.
  - Grup düzeyinde bootstrap (B=10000, seed=42).
  - Üç koşullu karar kuralı, birincil seed kuralı ve güç analizi.
- Seçim kuralları değişmedi: birincil baseline char_lr, Faz 3'ün teste taşınan encoder'ı
  bge_m3 (`frozen.RULES['test_transfer']`, `cv_compare.SELECTION_RULE`).

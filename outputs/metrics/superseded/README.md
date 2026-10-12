# Geçersiz kılınmış Faz 2 koşuları

Kayıt amaçlı saklanır; özet ve rapor bunları KULLANMAZ (`cv_compare.latest_phase2` bu klasöre inmez).
Hiçbirinde test seti kullanılmadı.

| Dosya | Commit | Neden geçersiz |
|---|---|---|
| `phase2_word_lr_20261011-183328.json`, `phase2_char_lr_20261011-184232.json` | 8ab6911 | İlk LR grid'i: iç CV seçimleri kenarda yığıldı (C=100; char (2,4)). Grid genişletildi; dış skorlar değişmedi (word_lr birebir, char_lr −0.0001). Npz'leri üzerine yazıldığı için tahminleri yok. |
| `phase2_word_svc_20261011-183327.json`, `phase2_word_cnb_20261011-183332.json` | 8ab6911 | Eski `selection_bias` tanımı (iç 3-fold − dış; öğrenme eğrisiyle karışık). |
| `phase2_word_lr_20261011-190400.json`, `phase2_char_lr_20261011-190355.json` | ec0c2e6 | `selection_bias` 15 fold ortalamasına göre (doğrusu 0. tekrar, aynı bölümleme); npz adı zaman damgasız. |
| `phase2_summary_20261011-191817.json` | ec0c2e6-dirty | Commit edilmemiş kodla üretildi. |
| `phase2_summary_20261011-194634.json` | b80dcdc | `p_table` (ham / N-B / N-B+Holm) ve `grid_edges` alanlarından önce. Sayılar yenisiyle (`phase2_summary_20261011-221201.json`, dfbdc13) birebir aynı; yalnızca alan eklendi. |
| `phase2_summary_20261011-221201.json` | dfbdc13 | `test_plan.sensitivity_c1` kuralından ve stats-reviewer düzeltmelerinden (erratum metni) önce. Sayılar yenisiyle (`phase2_summary_20261012-080148.json`) birebir aynı. |
| `preds/cv_*.npz` | 8ab6911 / ec0c2e6 | Zaman damgasız tahmin dosyaları (json ile eşleşmesi doğrulanamaz). |

Yeniden koşularda CV skorları aynı kalmalıdır (deterministik; aynı grid, aynı fold'lar).

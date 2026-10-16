"""Final test yönteminin TEK kaynağı (test seti açılmadan donduruldu).

Kodu: `ttc4900.paired_test` (McNemar, grup bootstrap, karar kuralı, güç analizi).
İnsan okuması için: docs/test_protokol.md. Bu dosya ile belge çelişirse bu dosya geçerlidir.

Önceki sürüm `cv_compare.TEST_PLAN` içindeydi (Faz 2 özetlerine gömülü olan metin). Bu sürümde
değişen: p her n için exact binom McNemar; bootstrap test içi yakın-tekrar grupları düzeyinde;
"ayırt edildi" karar kuralı ve güç analizi eklendi. Seçim kuralları (birincil baseline, Faz 3
teste taşıma) değişmedi.
"""

from __future__ import annotations

ALPHA = 0.05
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 1000
CI_LEVEL = 0.95
POWER_TARGET = 0.8
PRIMARY_SEED = 42  # birincil test modeli: BERTurk fine-tuning final modeli, bu seed

# (a, b): fark her zaman a - b olarak raporlanır
FAMILIES: dict[str, dict] = {
    "primary": {
        "comparisons": [("berturk_ft", "char_lr")],
        "note": (
            f"BERTurk fine-tuning final modeli (seed {PRIMARY_SEED}, önceden seçildi) vs "
            "birincil klasik baseline"
        ),
    },
    "secondary": {
        "comparisons": [
            ("char_lr", "word_lr"),
            ("char_lr", "word_svc"),
            ("char_lr", "word_cnb"),
            ("bge_m3", "char_lr"),
        ],
        "note": (
            "Faz 2'nin 3 klasik karşılaştırması + Faz 3'ün teste taşınan encoder'ı "
            "(frozen.RULES['test_transfer']). m=4 kesin: berturk_ft - bge_m3 bu aileye girmez "
            "(DESCRIPTIVE)"
        ),
    },
}
# p yok, Holm yok; yalnız grup bootstrap CI'ı (paired_test.describe_pair)
DESCRIPTIVE = [
    ("char_lr", "char_lr_c1clean"),  # C1 duyarlılığı (sensitivity_c1)
    ("berturk_ft", "bge_m3"),  # Faz 4 vs Faz 3 en iyisi (membership_note)
    (f"berturk_ft_s{PRIMARY_SEED + 1}", "char_lr"),  # diğer seed'ler (primary_seed_rule)
    (f"berturk_ft_s{PRIMARY_SEED + 2}", "char_lr"),
]

_M1 = len(FAMILIES["primary"]["comparisons"])
_M2 = len(FAMILIES["secondary"]["comparisons"])

TEST_PLAN = {
    "status": "test açılmadan donduruldu; test sonucuna göre hiçbir alan değiştirilmez",
    "families": {
        name: {
            "m": len(f["comparisons"]),
            "comparisons": [f"{a} - {b}" for a, b in f["comparisons"]],
            "note": f["note"],
        }
        for name, f in FAMILIES.items()
    },
    "descriptive": [f"{a} - {b}" for a, b in DESCRIPTIVE],
    "membership_note": (
        f"Birincil aile m={_M1}, ikincil aile m={_M2}; ikisi de kesin (Faz 5a, test açılmadan). "
        "Karar: berturk_ft - bge_m3 hiçbir Holm ailesine girmez, yalnız betimsel raporlanır (fark "
        "ve grup bootstrap CI'ı; p yok). İkincil aile büyümediği için α/m ve güç json'u "
        "değişmez. Bu karardan sonra FAMILIES ve DESCRIPTIVE değişmez."
    ),
    "descriptive_note": (
        "Betimsel çiftler (DESCRIPTIVE) hiçbir Holm ailesine girmez: yalnız iki modelin skorları "
        "ve farkın grup bootstrap %95 CI'ı (effect ile aynı ayarlar, aynı w'ler; "
        "paired_test.describe_pair). char_lr - char_lr_c1clean: sensitivity_c1. "
        "berturk_ft - bge_m3: membership_note. "
        f"berturk_ft_s{PRIMARY_SEED + 1} - char_lr ve berturk_ft_s{PRIMARY_SEED + 2} - char_lr: "
        "primary_seed_rule'daki diğer seed'lerin betimsel test skorlarının referansa göre farkı; "
        "birincil karara girmez."
    ),
    "primary": (
        f"BERTurk fine-tuning (seed {PRIMARY_SEED} final modeli; istisna: primary_seed_rule) vs "
        f"char_lr; ayrı aile, m={_M1}"
    ),
    "primary_seed_rule": (
        f"McNemar ve bootstrap'e giren BERTurk modeli seed {PRIMARY_SEED} final modelidir; bu "
        "seçim eğitimden önce yapıldı, test (ya da validation) skoruna göre değiştirilmez. Tek "
        "önceden tanımlı istisna finetune.FINAL_RULES['final_collapse']: seed "
        f"{PRIMARY_SEED} final modeli train üzerinde çökmüş çıkarsa (in-sample macro-F1 < 0.5) "
        f"seed {PRIMARY_SEED}+100 ile bir kez yeniden eğitilir ve birincil model odur. Diğer "
        "seed'lerin final modellerinin test macro-F1'i betimsel raporlanır (ortalama ± std, her "
        "biri için bootstrap CI; p yok, Holm yok). Birincil iddia bu tek eğitilmiş modele "
        "koşulludur: McNemar ve bootstrap test örneklemesi belirsizliğini ölçer, seed (eğitim) "
        "belirsizliğini ölçmez; seed değişkenliği CV'deki 3 seed ve betimsel test skorlarıyla "
        "verilir."
    ),
    "secondary": (
        "Dört klasik modelin donmuş final.best_params konfigürasyonları betimsel raporlanır; "
        f"char_lr'ye karşı 3 klasik karşılaştırma + bge_m3 - char_lr ayrı bir Holm ailesi (m={_M2})"
    ),
    "p_value": (
        "Exact binom McNemar, her n için: n10 = A doğru & B yanlış, n01 = A yanlış & B doğru; "
        "H0 altında n10 ~ Binom(n01 + n10, 1/2); iki yönlü p = scipy.stats.binomtest(n10, "
        "n01 + n10, 0.5).pvalue. Süreklilik düzeltmesi yok, χ² yaklaşımı kullanılmaz. "
        "n01 + n10 = 0 ise p = 1."
    ),
    "multiplicity": (
        f"Holm, her aile içinde ayrı, yalnızca exact McNemar p'leri üzerinde; α = {ALPHA}. "
        f"Birincil aile m={_M1} (düzeltme yok). Aileler ayrı ayrı α'da tutulur; aileler arası "
        "toplam hata kontrolü yapılmaz (bilinçli seçim: birincil soru ikincillerden bağımsızdır). "
        "Raporlanan %95 CI'lar marjinaldir (eşzamanlı değil)."
    ),
    "effect": (
        "Etki büyüklüğü: macro-F1 farkı (A - B; 7 sınıf, zero_division=0). Belirsizlik: "
        "eşleştirilmiş grup (cluster) bootstrap. Gruplar split_v2 'components' yakın-tekrar "
        "bileşenleridir (tekil satır = kendi grubu; data.test_groups). Gruplar iadeli örneklenir, "
        "iki model aynı yeniden örneklemde ölçülür. "
        f"B = {BOOTSTRAP_B}, np.random.default_rng({BOOTSTRAP_SEED}), yüzdelik yöntemle %95 CI "
        "(2.5. ve 97.5. yüzdelikler). Tek model skorları da aynı bootstrap'ten CI ile verilir. "
        "Tekrarlanabilirlik ayrıntısı: grup ağırlıkları w = rng.multinomial(G, [1/G]*G), "
        f"{BOOTSTRAP_CHUNK}'lik parçalar halinde; gruplar np.unique sıralı id düzeninde; replikat "
        "karışıklık matrisi Σ_g w_g C_g (paired_test.cluster_bootstrap)."
    ),
    "decision_rule": (
        "'Ayırt edildi' ⇔ Holm-düzeltilmiş p < 0.05 VE macro-F1 farkının %95 CI'ı 0'ı dışlıyor "
        "VE iki yön tutarlı: sign(n10 - n01) = CI'ın işareti. McNemar accuracy'yi, CI macro-F1'i "
        "ölçtüğü için yönler ayrışabilir; ayrışırsa sonuç 'ayırt edilemedi (yön çelişkisi)' olur. "
        "Koşullardan biri sağlanmazsa sonuç 'ayırt edilemedi'dir ve hangi koşulların sağlandığı "
        "ayrıca yazılır. 'Ayırt edilemedi' eşdeğerlik demek değildir (eşdeğerlik marjı "
        "tanımlanmadı)."
    ),
    "limitations": (
        "McNemar test satırlarını bağımsız sayar; test içindeki yakın-tekrar grup bağımlılığını "
        "yok sayar, bu yüzden p iyimser olabilir (testte satırların küçük bir kısmı çok satırlı "
        "gruplarda: 908 satırın 108'i, 50 grup; etkinin büyüklüğü ölçülmedi). Bootstrap CI "
        "grupları hesaba katar. İki koşul farklı büyüklükleri ölçer: McNemar doğru/yanlış "
        "(accuracy) uyuşmazlıklarını, CI macro-F1 farkını. Karar kuralı koşulların kesişimi "
        "olduğu için gücü her birinin tek başına gücünden yüksek olamaz."
    ),
    "power": (
        "Güç analizi yalnızca train OOF tahminlerinden (paired_test --power -> "
        "outputs/metrics/test_power_<ts>.json): model çifti başına uyuşmazlık oranı psi ve test "
        "boyutuyla %80 güçteki MDE (exact McNemar, psi alternatif altında sabit tutulur; birincil "
        f"için α = {ALPHA}, ikincil için Holm'un en kötü adımı α/{_M2} ile en iyi adımı α "
        "arasında). Birincil çiftin gücü, Faz 4'ün seed "
        f"{PRIMARY_SEED} OOF tahminlerinden (char_lr'nin 1. tekrar dış fold'larıyla aynı "
        "fold'lar, phase4_berturk_ft_<ts>.json + preds npz; tek tekrar, R=1) aynı CLI ile "
        "hesaplandı (Faz 4c); donmuş encoder vekilleri yalnız karşılaştırma için kalır. Küçük "
        "farkların yakalanması beklenmez; anlamsız sonuç eşdeğerlik kanıtı sayılmaz."
    ),
    "sensitivity_c1": (
        "C1 duyarlılık analizi (kural CV'den ÖNCE donduruldu): birincil klasik baseline ham "
        "char_lr olarak KALIR. U+0091-U+0094 silinmiş char_lr_c1clean yalnızca duyarlılık "
        "analizidir; final.best_params ile teste ikincil/betimsel olarak taşınır ve ham "
        "char_lr ile birlikte raporlanır. İkisi arasında test (ya da CV) sonucuna göre seçim "
        "YAPILMAZ; BERTurk karşılaştırmasındaki referans değişmez. Testte bu çift için p "
        "hesaplanmaz: yalnızca iki modelin skorları ve farkın grup düzeyinde paired bootstrap %95 "
        "CI'ı (effect ile aynı ayarlar; paired_test.describe_pair) betimsel raporlanır (hiçbir "
        "Holm ailesine girmez)."
    ),
    "expectation_note": (
        "CV skoru 2904 satırla eğitimi ölçer (BERT fine-tuning CV'sinde validation ayrıldığı "
        "için ~2323), final model 3630 satırla eğitilir; ayrıca seçilen klasik modelin CV skoru "
        "4 adayın maksimumudur (kazananın laneti). Test skorunun CV'den farklı çıkması tek "
        "başına anomali değildir."
    ),
}

# Sonuç cümleleri: test görülmeden donduruldu ("test öncesi son commit"). final_eval yalnız
# decide() çıktısına göre şablon seçer ve sayıları doldurur; sonuç dili sonuca göre yazılmaz.
# Fark her zaman a - b; CI = %95 grup bootstrap (yüzdelik), p = exact McNemar.
SENTENCES = {
    "ayırt edildi": (
        "{a}, test macro-F1'de {b}'den {yon} olarak ayırt edildi: fark {diff:+.4f} (%95 grup "
        "bootstrap CI [{lo:+.4f}, {hi:+.4f}]); exact McNemar n10/n01 = {n10}/{n01}, ham p = "
        "{p:.3g}, Holm p = {p_holm:.3g} ({aile} aile, m={m}). Üç koşul da sağlandı."
    ),
    "ayırt edilemedi (yön çelişkisi)": (
        "{a} ile {b} ayırt edilemedi (yön çelişkisi): McNemar (accuracy) {mc_yon} (n10/n01 = "
        "{n10}/{n01}, Holm p = {p_holm:.3g}), macro-F1 farkının CI'ı ise {ci_yon} (fark "
        "{diff:+.4f}, [{lo:+.4f}, {hi:+.4f}]) ({aile} aile, m={m}). İki ölçüt farklı büyüklükleri "
        "ölçtüğü için yön hakkında karar verilmez."
    ),
    "ayırt edilemedi": (
        "{a} ile {b} test setinde ayırt edilemedi: fark {diff:+.4f} (%95 grup bootstrap CI "
        "[{lo:+.4f}, {hi:+.4f}]); exact McNemar n10/n01 = {n10}/{n01}, Holm p = {p_holm:.3g} "
        "({aile} aile, m={m}). Sağlanan koşullar: {saglananlar}; sağlanmayanlar: "
        "{saglanmayanlar}. Bu bir eşdeğerlik iddiası değildir (eşdeğerlik marjı tanımlanmadı). "
        "Test öncesi güç analizi: {guc}."
    ),
    "betimsel": (
        "{a} − {b} (betimsel, p yok): macro-F1 {fa:.4f} vs {fb:.4f}, fark {diff:+.4f} (%95 grup "
        "bootstrap CI [{lo:+.4f}, {hi:+.4f}]). Hiçbir Holm ailesine girmez; çıkarım için "
        "kullanılmaz."
    ),
    "seed_ozeti": (
        "BERTurk fine-tuning test macro-F1'i ({k} seed, betimsel): {values}; ortalama "
        "{mean:.4f} ± {std:.4f} (std, ddof=1; aralık {lo:.4f}–{hi:.4f}). Bu {k} eğitimin "
        "yayılımıdır, ortalamanın güven aralığı değildir; std {k} değerden kestirildiği için çok "
        "kabadır ve ortalama birincil modeli de içerir. Birincil karar yalnız {primary} (seed "
        "{seed}) modeline dayanır; seed std'si eğitim değişkenliğidir, test örneklemesi "
        "belirsizliği değildir."
    ),
    "birincil_kosul": (
        "Birincil iddia bu tek eğitilmiş modele koşuludur: McNemar ve bootstrap test "
        "örneklemesinin belirsizliğini ölçer, seed'den (eğitimden) gelen belirsizliği ölçmez."
    ),
    "sinirliliklar": (
        "Sınırlılıklar: exact McNemar test satırlarını bağımsız sayar ve test içi yakın-tekrar "
        "bağımlılığını yok sayar, bu yüzden p iyimser olabilir (accuracy farkının grup bootstrap "
        "CI'ı bunu denetlemek için sonuç json'unda acc_diff_ci95 olarak betimsel verildi; "
        "karar kuralına girmez). Raporlanan %95 CI'lar marjinaldir, "
        "eşzamanlı değildir. Sınıf bazlı F1'ler CI'sızdır ve gürültülüdür (sınıf başına 114–138 "
        "test satırı). Test tahminleri yerel CPU'da fp32 ile, CV/OOF tahminleri Colab'da fp16 ile "
        "üretildi; sınırdaki satırlarda bu küçük bir oynama yaratabilir."
    ),
    "seed_tutarlilik": (
        "Diğer seed'lerle tutarlılık (betimsel, karara girmez): birincil farkın %95 CI'ı "
        "{birincil_yon}; {ayrintilar}. Birincil karar bu satırdan bağımsızdır; satır yalnız "
        "sonucun tek eğitime ne kadar bağlı olduğunu gösterir."
    ),
    "cv_test_notu": (
        "Test skorlarının CV skorlarından farklı çıkması tek başına anomali değildir: final "
        "modeller 3630 satırla (CV'de ~2904, BERT'te ~2323) eğitildi ve seçilen klasik modelin "
        "CV skoru adayların maksimumudur (kazananın laneti; TEST_PLAN['expectation_note'])."
    ),
}

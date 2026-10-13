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
            "(frozen.RULES['test_transfer'])"
        ),
    },
}
DESCRIPTIVE = [("char_lr", "char_lr_c1clean")]  # p yok, Holm yok; yalnız grup bootstrap CI'ı

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
        f"Birincil aile m={_M1} ve değişmez. İkincil aile şimdilik m={_M2}. Faz 4'ten ek bir "
        "test karşılaştırması gelirse (örn. berturk_ft - bge_m3) hangi aileye gireceği ya da "
        "betimsel kalacağı, test açılmadan önceki son commit'te FAMILIES'e yazılarak kesinleşir; "
        "ikincil aile büyürse güç json'u o commit'te yeniden üretilir (α/m değişir). O commit'ten "
        "sonra hiçbir aile değişmez. Mevcut öneri: berturk_ft - bge_m3 yalnız betimsel CI."
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
        f"{PRIMARY_SEED} OOF tahminleri (char_lr'nin 1. tekrar dış fold'larıyla aynı fold'lar, "
        "phase4_berturk_ft_<ts>.json + preds npz) üretilince aynı CLI ile hesaplanır; o zamana "
        "dek donmuş encoder'lar yalnız kaba vekildir. Küçük farkların yakalanması beklenmez; "
        "anlamsız sonuç eşdeğerlik kanıtı sayılmaz."
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
        "CV skoru 2904 satırla eğitimi ölçer, final model 3630 satırla eğitilir; ayrıca seçilen "
        "modelin CV skoru 4 adayın maksimumudur (kazananın laneti). Test skorunun CV'den "
        "farklı çıkması tek başına anomali değildir."
    ),
}

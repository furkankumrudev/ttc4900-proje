# Faz 8 protokolü: yayın (Hugging Face Hub + Gradio Space)

**Kaynak:** `src/ttc4900/publish.py` (`PUBLISH_RULES`). Bu belge ile kaynak çelişirse kaynak
geçerlidir. Kurallar 8a commit'inde, ağa hiçbir şey yazılmadan donduruldu.

## Kararlar (kullanıcı)
- **Görünürlük: D + A.**
  - Veri setinin lisansı (`license: unknown`) için veri yazarına kullanıcı yazar.
  - Beklerken **model reposu private**, **Space public** olur.
  - Varsayılan private'tır. Kod modeli private oluşturur ve görünürlüğü hiçbir zaman
    değiştirmez. Modelin public yapılması ayrı bir kullanıcı kararıdır.
- **Repo adları:**
  - Model: `<kullanıcı>/berturk-ttc4900-haber`.
  - Space: `<kullanıcı>/berturk-ttc4900-haber-demo`.
  - Kullanıcı adı kodda sabit değil; 8b ve 8c'de `--user` ile verilir. Boş ya da yer tutucu
    (ör. `[...]`) ise kod durur.
- **Agent:** yayın işleri `publisher` agent'ıyla yapılabilir (`.claude/agents/publisher.md`).

## Token politikası (`PUBLISH_RULES['token']`)
- **Giriş:** kullanıcı Hugging Face'e kendi terminalinde `hf auth login` ile girer. Token
  HF'nin kendi önbelleğinde kalır.
- **Kod:**
  - Token'ı okumaz, yazdırmaz, dosyaya ya da commit'e yazmaz.
  - Kimlik doğrulama `HfApi()`'nin varsayılanıdır. Yalnızca `whoami` ile giriş yapan
    kullanıcının adı `--user` ile karşılaştırılır.
  - Bir AST testi modülde `get_token`, `environ`, `add_space_secret` ve görünürlük değiştiren
    çağrıların olmadığını denetler.
- **Space:**
  - Space'te **yazma yetkili token olmaz**.
  - Private modeli okumak için **yalnız bu model reposunu okuyabilen fine-grained bir token**
    kullanılır. Kullanıcı bunu Space secret'ı (`HF_TOKEN`) olarak Hub arayüzünden kendisi
    girer; kod secret oluşturmaz.
  - `space/app.py` token'ı yalnız `from_pretrained`'e iletir; yazdırmaz.
- Token sohbete yazılmaz.

## Bütünlük (`PUBLISH_RULES['integrity']`)
- **Yüklenecek model:** yalnız `outputs/models/42`, yani Faz 5'in birincil modeli (seed 42).
- **Yüklemeden önce** (`check-local`): `config.json`, `model.safetensors`, `tokenizer.json` ve
  `tokenizer_config.json`'un sha256'ları, Faz 5 `phase5_model_check` json'undaki
  `model_files_sha256` ile birebir aynı olmalı. Tutmazsa hiçbir şey yüklenmez.
- **Yüklemeden sonra** (`verify-remote`): bütün dosyalar Hub'dan commit sha'sıyla geri indirilir
  ve sha256'ları yeniden karşılaştırılır. Card ve `calibration.json` da bu kontrole dahil.
- **Space:** model bu commit sha'sına sabitlenir (`space_config.json`, tam 40 karakter sha).

## İçerik (`PUBLISH_RULES['content']`, `['space']`)
- **Model card ve Space README'si** json'lardan `bundle` komutuyla üretilir; sayılar elle
  yazılmaz. Üretilen dosyalar: `outputs/publish/model/README.md`, `calibration.json` ve
  `outputs/publish/space/README.md`.
- **Dil:**
  - Birincil sonuç "ayırt edilemedi".
  - "char_lr'den iyi", "neredeyse anlamlı", "eğilim" ve "eşdeğer" kullanılmaz.
  - BERTurk'ün kullanılması bir dağıtım tercihidir.
  - T = 1.283 ve kalan ECE belirtilir.
  - Bir kontrol (`check_text`) yasak ifadeleri reddeder.
- **Lisans:** card `license: other` der. Taban model MIT, eğitim verisinin lisansı belirsiz.
  Veri atfı (Kaggle sayfası ve Yıldırım & Yıldız 2018) card'da yer alır.
- **Doğrulanabilirlik:**
  - Dataset card'da olmayan kaynak ya da dönem iddiası yazılmaz. Space uyarısı yalnızca
    şunu söyler: "Model yalnızca `savasy/ttc4900` veri setindeki haber metinleriyle ince
    ayarlandı".
  - Haber metni alıntılama yapılmaz. Space'te veri setinden örnek metin yoktur.
- **Space uygulaması** (`space/app.py`): olasılıklar softmax(logits / T); ilk 3 sınıf; ilk 512
  token kullanılır (eğitimle aynı, baştan kesme).

## Adımlar
1. **8a (ağ yok):**
   - Kurallar, `publish.py`, `space/app.py`, testler, bu belge ve publisher agent'ı.
   - `bundle` ile üretilmiş card ve Space README'si.
   - Çevrimdışı uygulama testi: yerel modelle uygulamanın olasılıkları, bağımsız bir ileri
     geçişin softmax(z / T) değeriyle aynı olmalı. Test seti kullanılmaz.
   - Commit.
2. **8b (kullanıcının o turdaki onayıyla):**
   - Kullanıcı kendi terminalinde giriş yapar.
   - `check-local`.
   - `upload-model --user U --confirm` (private).
   - `verify-remote --user U --revision <commit sha>`.
   - Commit sha'sı kaydedilir.
3. **8c (kullanıcının o turdaki onayıyla):**
   - `create-space --user U --model-revision <sha> --confirm`.
   - Kullanıcı, fine-grained salt okunur token'ı Space secret'ı olarak girer.
   - Uçtan uca bir istekle denenir.
4. **8d:** README'ye bağlantılar eklenir; leakage-reviewer ve stats-reviewer denetler; commit.

## Kayıt (`outputs/publish/hub_record.json`)
- **8b (2026-10-17), tamam.**
  - `FurkanHug/berturk-ttc4900-haber` private olarak oluşturuldu.
  - Hub commit sha'sı `8c649303eea44ba049b171f08b25b500bddb6c71`.
  - Hub'dan geri indirilen 6 dosyanın sha256'ları yerel kayıtla aynı.
- **8c (2026-10-17), başarısız.**
  - Gradio Space oluşturma isteği `402 Payment Required` ile reddedildi.
  - Hub'ın mesajı: ücretsiz cpu-basic üzerinde Gradio ve Docker Space'leri PRO gerektiriyor;
    static Space'ler ücretsiz.
  - Space oluşmadı, model reposu private kaldı.
  - Kullanıcı kararı: 8c şimdilik bırakıldı.

## Not: belgelerde düzeltilen ifade
8a sırasında, README'de ve Faz 6 ile kısayol belgelerinde "tek kaynak / tek dönem" ifadesinin
dataset card'la doğrulanamadığı görüldü. Card, verinin Kemik grubundan alındığını ve kaynağının
birden fazla "Türk çevrimiçi haber sitesi" olduğunu söylüyor; dönem belirtmiyor. İfade "tek
derleme; dönemi card'da belirtilmiyor" olarak düzeltildi. Aynı eski ifade donmuş
`CALIB_RULES['limitations']` metninde ve 6b/6c json'larında da geçiyor; bunlar donmuş kayıt
olduğu için değiştirilmedi.

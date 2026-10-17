"""Faz 8: yayın (Hugging Face Hub modeli + Gradio Space). Kurallar `PUBLISH_RULES`'tadır; 8a
commit'inde, ağa hiçbir şey yazılmadan donduruldu. İnsan okuması: docs/faz8_protokol.md.

Akış (ağa yazan her adım kullanıcının o turdaki açık onayıyla; CLI ayrıca --confirm ister):
    python -m ttc4900.publish bundle                       # 8a, ağ yok: card + calibration.json
    python -m ttc4900.publish check-local                  # 8a/8b, ağ yok: sha256 = Faz 5 kaydı
    python -m ttc4900.publish upload-model --user U --confirm             # 8b: private model
    python -m ttc4900.publish verify-remote --user U --revision SHA       # 8b: geri indir, sha256
    python -m ttc4900.publish create-space --user U --model-revision SHA --confirm   # 8c

Token hiçbir yerde okunmaz, yazdırılmaz, dosyaya yazılmaz: kimlik doğrulama HfApi()'nin
varsayılanıdır (kullanıcı kendi terminalinde `hf auth login` yapar).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from ttc4900.cv_compare import latest_metrics
from ttc4900.final_eval import model_files_sha256, sha256_file
from ttc4900.paths import METRICS_DIR, MODELS_DIR, OUTPUTS, PROJECT_ROOT

REPO_NAME = "berturk-ttc4900-haber"
SPACE_NAME = "berturk-ttc4900-haber-demo"
MODEL_SLOT = "berturk_ft"
MODEL_DIR = MODELS_DIR / "42"
BUNDLE_DIR = OUTPUTS / "publish"
SPACE_SRC = PROJECT_ROOT / "space"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
BUNDLE_FILES = ("README.md", "calibration.json")  # modele eşlik eden, üretilen dosyalar
SPACE_FILES = ("app.py", "requirements.txt")  # space/ içinde, elle yazılmış
BANNED = ("neredeyse", "eğilim", "eşdeğer", "char_lr'den iyi", "daha iyi kalibre", "2010",
          "tek bir kaynak", "tek kaynak")  # fmt: skip
DATA_ATTRIBUTION = (
    "Veri: Hugging Face `savasy/ttc4900` (revision `236e17b0c497ceb5437c175cb1ffaae201bfb287`). "
    "Dataset card'a göre veri Kemik grubundan (kemik.yildiz.edu.tr) alınmıştır; kaynağı "
    "Türk çevrimiçi haber siteleridir. Atıf: <https://www.kaggle.com/savasy/ttc4900>; Yıldırım, S. "
    "ve Yıldız, T. (2018), *A comparative analysis of text classification for Turkish language*, "
    "Pamukkale Univ Muh Bilim Derg 24(5), 879–886, doi:10.5505/pajes.2018.15931. Lisans: dataset "
    "card'da `license: unknown`."
)
SPACE_NOTICE = (
    "Model yalnızca `savasy/ttc4900` veri setindeki haber metinleriyle ince ayarlandı. Başka "
    "türden ya da başka kaynaklardan gelen metinlerde tahminler ve olasılıklar güvenilir "
    "olmayabilir. Ajans adları ve tırnak biçimi gibi konu dışı işaretler tahmini etkileyebilir. "
    "Metnin ilk 512 token'ından sonrası kullanılmaz."
)

PUBLISH_RULES = {
    "status": (
        "8a commit'inde, ağa hiçbir şey yazılmadan donduruldu. Modeller, T, sonuç json'ları ve "
        "Faz 1-7 kuralları değişmez; test seti ve test metni okunmaz."
    ),
    "visibility": (
        "Kullanıcı kararı (D + A): veri yazarına lisans sorulur (kullanıcı yazar); beklerken "
        "model reposu PRIVATE, Space PUBLIC. Varsayılan private'tır; kod modeli private "
        "oluşturur ve görünürlüğü değiştirmez. Modelin public yapılması ayrı bir kullanıcı "
        "kararıdır."
    ),
    "token": (
        "Kullanıcı Hugging Face'e kendi terminalinde `hf auth login` ile girer. Kod token'ı "
        "okumaz, yazdırmaz, dosyaya ya da commit'e yazmaz; kimlik doğrulama HfApi()'nin "
        "varsayılanıdır; yalnız whoami ile kullanıcı adı denetlenir. Space'te YAZMA yetkili "
        "token olmaz. Space, yalnız bu model reposunu okuyabilen fine-grained bir token "
        "kullanır; kullanıcı bunu Space secret'ı (HF_TOKEN) olarak Hub arayüzünden kendisi "
        "girer. Kod Space secret'ı oluşturmaz."
    ),
    "approval": (
        "Ağa yazan her işlem (repo oluşturma, yükleme, Space oluşturma) kullanıcının o turdaki "
        "açık onayıyla ve CLI'da --confirm ile yapılır. Kullanıcı adı CLI'dan verilir "
        "(--user); boş ya da yer tutucu ([..]) ise kod durur; whoami ile aynı olmalıdır."
    ),
    "integrity": (
        "Yüklenecek model yalnız outputs/models/42 (Faz 5 birincil, seed 42). Yüklemeden önce "
        "dosyaların sha256'sı Faz 5 model_check json'undaki model_files_sha256 ile birebir "
        "aynı olmalı. Yüklemeden sonra dosyalar Hub'dan commit sha'sıyla geri indirilip "
        "yeniden doğrulanır. Space modeli bu commit sha'sına sabitlenmiş olarak yükler."
    ),
    "content": (
        "Model card ve Space metnindeki sayılar json'lardan bu modülle üretilir, elle yazılmaz. "
        "Dil: birincil sonuç 'ayırt edilemedi'; 'char_lr'den iyi', 'neredeyse anlamlı', "
        "'eğilim', 'eşdeğer' yok; BERTurk bir dağıtım tercihidir; T = 1.283 ve kalan ECE; "
        "card lisansı 'other' (taban model MIT, veri lisansı unknown); veri atfı; kısayol ve "
        "veri dağılımı sınırlılıkları. Haber metni alıntılama yapılmaz. Dataset card'da "
        "olmayan kaynak ya da dönem iddiası yazılmaz."
    ),
    "space": (
        "Space: Gradio; model Hub'dan sabit commit'le yüklenir; olasılık softmax(z / T), "
        "T = calibration.json; ilk 3 sınıf gösterilir; girdi üstünde doğrulanabilir kısa "
        "uyarı (SPACE_NOTICE). Örnek metin olarak veri setinden haber konulmaz."
    ),
    "commits": (
        "8a: kurallar, kod, testler, Space şablonu, protokol, publisher agent, üretilmiş card "
        "(ağ yok). 8b: model yükleme + uzak doğrulama (kullanıcı onayıyla). 8c: Space "
        "(kullanıcı onayıyla). 8d: README bağlantıları ve denetçiler."
    ),
}


# --------------------------------------------------------------------------- kimlik ve dosyalar
def repo_id(user: str | None, name: str = REPO_NAME) -> str:
    """Kullanıcı adı zorunlu; boş ya da yer tutucu ise durur (tahmin edilmez)."""
    if not user or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", user):
        raise ValueError(f"geçerli bir Hub kullanıcı adı verilmeli (--user), alınan: {user!r}")
    return f"{user}/{name}"


def check_local(model_dir: Path = MODEL_DIR, metrics_dir: Path = METRICS_DIR) -> dict[str, str]:
    """PUBLISH_RULES['integrity']: yerel dosyalar = Faz 5 model_check kaydı (birebir)."""
    chk = latest_metrics("model_check", metrics_dir, phase=5)
    if chk is None or not chk.get("passed"):
        raise RuntimeError("geçmiş bir phase5_model_check json'u yok")
    want = chk["models"][MODEL_SLOT]["model_files_sha256"]
    have = model_files_sha256(model_dir)
    if set(want) != set(MODEL_FILES) or have != want:
        raise ValueError(f"{model_dir}: dosyalar/sha256'lar Faz 5 kaydıyla aynı değil")
    return have


# --------------------------------------------------------------------------- sayılar
def numbers(metrics_dir: Path = METRICS_DIR) -> dict:
    """Card ve Space metnindeki tüm sayılar (tek kaynak: commit'li json'lar)."""
    p5 = latest_metrics("test", metrics_dir, phase=5)
    power = json.loads(Path(metrics_dir, p5["power_source"]).read_text(encoding="utf-8"))
    six_b = latest_metrics("train_calibration", metrics_dir, phase=6)
    six_c = latest_metrics("test_calibration", metrics_dir, phase=6)
    prim = p5["families"]["primary"]["comparisons"][0]
    pw = next(r for r in power["pairs"] if r["role"] == "primary")["by_alpha"]["0.05"]
    m6 = six_c["models"][MODEL_SLOT]["metrics"]
    return {
        "sources": {
            k: v["_file"] for k, v in {"phase5": p5, "phase6b": six_b, "phase6c": six_c}.items()
        }
        | {"power": p5["power_source"]},  # fmt: skip
        "n_test": p5["n_test"],
        "bert": p5["models"][MODEL_SLOT],
        "char_lr": p5["models"]["char_lr"],
        "primary": prim,
        "power": {"power": pw["power_at_oof_acc_diff"], "mde": pw["mde_macro_f1_rough"]},
        "seeds": p5["seed_summary"],
        "T": six_b["models"][MODEL_SLOT]["T_final"],
        "T_ci": six_b["models"][MODEL_SLOT]["T_final_ci95"],
        "calib_test": {
            "verdict": six_c["models"][MODEL_SLOT]["verdict"],
            "nll": m6["nll"],
            "ece15": m6["ece15"],
        },  # fmt: skip
        "calib_train_verdict": six_b["models"][MODEL_SLOT]["verdict"],
    }


def calibration_payload(nums: dict, metrics_dir: Path = METRICS_DIR) -> dict:
    src = Path(metrics_dir, nums["sources"]["phase6b"])
    return {
        "method": "temperature scaling: p = softmax(logits / temperature)",
        "temperature": nums["T"],
        "temperature_ci95_data_sampling": nums["T_ci"],
        "source": {"file": src.name, "sha256": sha256_file(src)},
        "test_ece15_after": nums["calib_test"]["ece15"]["after"],
        "note": "Yalnız bu veri dağılımı için kalibre edildi; ayrıntı docs/faz6_sonuclar.md.",
    }


def _ci(x, d: int = 4) -> str:
    return f"[{x[0]:.{d}f}, {x[1]:.{d}f}]"


def build_model_card(nums: dict) -> str:
    b, c, pr, pw, s = nums["bert"], nums["char_lr"], nums["primary"], nums["power"], nums["seeds"]
    ct = nums["calib_test"]
    ece = f"{ct['ece15']['before']:.3f} → {ct['ece15']['after']:.3f}"

    def row(m: dict) -> str:
        acc = f"{m['accuracy']:.4f} {_ci(m['accuracy_ci95'])}"
        return f"{m['macro_f1']:.4f} {_ci(m['macro_f1_ci95'])} | {acc}"

    return f"""---
language: tr
license: other
license_name: unknown-training-data
license_link: https://huggingface.co/datasets/savasy/ttc4900
base_model: dbmdz/bert-base-turkish-cased
datasets:
- savasy/ttc4900
pipeline_tag: text-classification
tags:
- turkish
- news-classification
- calibrated
---

# BERTurk, TTC4900 Türkçe haber sınıflandırma (7 sınıf)

`dbmdz/bert-base-turkish-cased` modelinin `savasy/ttc4900` üzerinde ince ayarlanmış hâli
(seed 42). Sınıflar: siyaset, dünya, ekonomi, kültür, sağlık, spor, teknoloji. Proje deposu,
protokoller ve tüm sonuç json'ları GitHub'dadır.

## Lisans
- Taban model `dbmdz/bert-base-turkish-cased`: MIT.
- Eğitim verisi `savasy/ttc4900`: dataset card'da `license: unknown`. Bu yüzden bu ağırlıklar
  için bir lisans beyan edilmedi (`license: other`). Lisans durumu netleşene kadar bu repo
  private tutulur.

## Veri ve atıf
{DATA_ATTRIBUTION}

- Tam tekrarlar temizlendi.
- Yakın tekrarlar (TF-IDF kosinüsü > 0.8) gruplandı ve hiçbir grup train ile test arasında
  bölünmedi.
- Bölme: train 3630, test {nums["n_test"]}.

## Sonuçlar (test, n = {nums["n_test"]}, %95 grup bootstrap CI)
| Model | Test macro-F1 | Test accuracy |
|---|---|---|
| Bu model (BERTurk, seed 42) | {row(b)} |
| Karakter TF-IDF + LR (birincil baseline) | {row(c)} |

**Birincil karşılaştırma (önceden kaydedilmiş): ayırt edilemedi.**
- macro-F1 farkı {pr["diff"]:+.4f}, %95 CI {_ci(pr["diff_ci95"])}.
- Exact McNemar n10/n01 = {pr["n10"]}/{pr["n01"]}, p = {pr["p"]:.2f}.
- Testten önce yazılan güç analizi: MDE ≈ {pw["mde"]:.3f}, güç ≈ {pw["power"]:.2f}.
- Bu sonuç iki modelin aynı olduğu anlamına gelmez; aynılık için önceden bir marj tanımlanmadı.
- Bu modelin demoda kullanılması bir dağıtım tercihidir, bir üstünlük iddiası değildir.

Üç seed'in test macro-F1'i: ortalama {s["mean"]:.4f} ± {s["std_ddof1"]:.4f} (std, betimsel).

## Olasılık kalibrasyonu
- Olasılıklar `softmax(logits / T)` ile hesaplanmalıdır: **T = {nums["T"]:.3f}**
  (`calibration.json`).
- T yalnızca train OOF tahminlerinden öğrenildi.
- Train içinde iyileşme {nums["calib_train_verdict"]}. Testte (önceden kaydedilmiş, tek
  sefer) **{ct["verdict"]}**: NLL {ct["nll"]["before"]:.4f} → {ct["nll"]["after"]:.4f}.
- Kalan kalibrasyon hatası: test ECE (15 bin) {ece}.
  En yüksek güven bandında model hâlâ biraz aşırı emin.

## Sınırlılıklar
- **Kısayol öğrenme:** veri setinde etiketle ilişkili kaynak işaretleri var (bozuk tırnak
  karakterleri, ajans künyeleri). Skorlar başka kaynaklardan gelen metinler için iyimser
  olabilir; bu ölçülmedi.
- **Veri dağılımı:** model yalnızca bu veri setindeki haber metinleriyle eğitildi. Kalibrasyon
  da yalnızca bu dağılım için geçerlidir.
- **Tek eğitilmiş model ve tek test seti:** sonuçlar bu tek koşuya ve bu test setine
  koşulludur.
- **Girdi uzunluğu:** girdinin ilk 512 token'ından sonrası kullanılmaz.
"""


def build_space_readme(nums: dict, model_repo: str) -> str:
    return f"""---
title: BERTurk TTC4900 Haber Sınıflandırma
emoji: 📰
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: other
models:
- {model_repo}
---

# BERTurk TTC4900 haber sınıflandırma demosu

{SPACE_NOTICE}

- Olasılıklar T = {nums["T"]:.3f} ile kalibre edildi (yalnız bu veri dağılımı için). Testte
  kalan ECE {nums["calib_test"]["ece15"]["after"]:.3f}.
- BERTurk'ün kullanılması bir dağıtım tercihidir. Testte karakter TF-IDF + LR baseline'ından
  ayırt edilemedi (macro-F1 farkı {nums["primary"]["diff"]:+.4f}, %95 CI
  {_ci(nums["primary"]["diff_ci95"])}).

{DATA_ATTRIBUTION}
"""


def check_text(text: str) -> None:
    """PUBLISH_RULES['content']: yasak ifade yok."""
    low = text.lower()
    found = [b for b in BANNED if b in low]
    if found:
        raise ValueError(f"yasak ifade(ler): {found}")


# --------------------------------------------------------------------------- 8a: paket (ağ yok)
def bundle(out_dir: Path = BUNDLE_DIR, metrics_dir: Path = METRICS_DIR,
           model_repo_placeholder: str = f"<user>/{REPO_NAME}") -> dict[str, str]:  # fmt: skip
    nums = numbers(metrics_dir)
    card = build_model_card(nums)
    space = build_space_readme(nums, model_repo_placeholder)
    for t in (card, space, SPACE_NOTICE):
        check_text(t)
    out_dir = Path(out_dir)
    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    (out_dir / "space").mkdir(parents=True, exist_ok=True)
    files = {
        out_dir / "model" / "README.md": card,
        out_dir / "model" / "calibration.json": json.dumps(
            calibration_payload(nums, metrics_dir), ensure_ascii=False, indent=2
        ),
        out_dir / "space" / "README.md": space,
    }
    for p, t in files.items():
        p.write_text(t + ("" if t.endswith("\n") else "\n"), encoding="utf-8", newline="\n")
    return {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in files}


# --------------------------------------------------------------------------- 8b/8c: ağ (onaylı)
def _api(api=None):
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    return api


def _check_user(api, user: str) -> None:
    who = api.whoami()["name"]  # yalnız kullanıcı adı; token okunmaz
    if who != user:
        raise RuntimeError(f"giriş yapılan kullanıcı {who!r}, --user {user!r} ile aynı değil")


def upload_model(
    user: str,
    confirm: bool,
    api=None,
    model_dir: Path = MODEL_DIR,
    bundle_dir: Path = BUNDLE_DIR,
    metrics_dir: Path = METRICS_DIR,
) -> str:
    """8b: private repo + model dosyaları + card + calibration.json. Döner: commit sha."""
    if not confirm:
        raise RuntimeError("ağa yazmak için --confirm (kullanıcının o turdaki onayı) gerekir")
    rid = repo_id(user)
    check_local(model_dir, metrics_dir)
    api = _api(api)
    _check_user(api, user)
    api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(rid).private:
        raise RuntimeError(f"{rid} private değil; görünürlük kullanıcı kararıdır, durdu")
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(f, str(Path(model_dir, f))) for f in MODEL_FILES]
    ops += [CommitOperationAdd(f, str(Path(bundle_dir, "model", f))) for f in BUNDLE_FILES]
    info = api.create_commit(rid, ops, commit_message="Faz 8b: seed 42 modeli ve model card")
    return info.oid


def verify_remote(
    user: str,
    revision: str,
    api=None,
    model_dir: Path = MODEL_DIR,
    bundle_dir: Path = BUNDLE_DIR,
    metrics_dir: Path = METRICS_DIR,
) -> dict:
    """8b: dosyalar Hub'dan commit sha'sıyla geri indirilir; sha256 = yerel/Faz 5 kaydı."""
    want = check_local(model_dir, metrics_dir)
    want |= {f: sha256_file(Path(bundle_dir, "model", f)) for f in BUNDLE_FILES}
    api = _api(api)
    rid = repo_id(user)
    got = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in want:
            p = api.hf_hub_download(rid, f, revision=revision, local_dir=tmp)
            got[f] = sha256_file(Path(p))
    bad = sorted(k for k in want if got.get(k) != want[k])
    if bad:
        raise ValueError(f"Hub dosyaları yerel kayıtla aynı değil: {bad}")
    return got


def create_space(user: str, model_revision: str, confirm: bool, api=None,
                 space_src: Path = SPACE_SRC, bundle_dir: Path = BUNDLE_DIR) -> str:  # fmt: skip
    """8c: public Space; model sabit commit'le. Secret (HF_TOKEN) kullanıcı tarafından girilir."""
    if not confirm:
        raise RuntimeError("ağa yazmak için --confirm (kullanıcının o turdaki onayı) gerekir")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("model_revision tam bir commit sha'sı olmalı (40 hex)")
    api = _api(api)
    _check_user(api, user)
    rid, sid = repo_id(user), repo_id(user, SPACE_NAME)
    readme = Path(bundle_dir, "space", "README.md").read_text(encoding="utf-8")
    readme = readme.replace(f"<user>/{REPO_NAME}", rid)
    check_text(readme)
    cfg = json.dumps({"model_repo": rid, "model_revision": model_revision}, indent=2)
    from huggingface_hub import CommitOperationAdd

    api.create_repo(sid, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    ops = [CommitOperationAdd(f, str(Path(space_src, f))) for f in SPACE_FILES]
    ops += [CommitOperationAdd("README.md", readme.encode("utf-8")),
            CommitOperationAdd("space_config.json", cfg.encode("utf-8"))]  # fmt: skip
    info = api.create_commit(sid, ops, repo_type="space", commit_message="Faz 8c: demo")
    return info.oid


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["bundle", "check-local", "upload-model", "verify-remote",
                                    "create-space"])  # fmt: skip
    ap.add_argument("--user")
    ap.add_argument("--revision")
    ap.add_argument("--model-revision")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.cmd == "bundle":
        print(json.dumps(bundle(), indent=1))
    elif a.cmd == "check-local":
        print(json.dumps(check_local(), indent=1))
    elif a.cmd == "upload-model":
        print("commit:", upload_model(a.user, a.confirm))
    elif a.cmd == "verify-remote":
        print(json.dumps(verify_remote(a.user, a.revision), indent=1))
    else:
        print("space commit:", create_space(a.user, a.model_revision, a.confirm))


if __name__ == "__main__":
    main()

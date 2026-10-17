"""Faz 8 (publish) testleri: ağ yok. Hub çağrıları sahte bir API ile; Space uygulaması yerel
modelle (varsa) çevrimdışı denenir."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import shutil
from pathlib import Path

import pytest

from ttc4900 import publish as pub
from ttc4900.final_eval import sha256_file
from ttc4900.paths import PROJECT_ROOT

APP_PATH = PROJECT_ROOT / "space" / "app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("space_app", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- sahte dünya
@pytest.fixture
def fake(tmp_path):
    """Küçük sahte model klasörü + bu dosyaların sha256'larını içeren Faz 5 model_check."""
    model, md, bundle = tmp_path / "model", tmp_path / "metrics", tmp_path / "bundle"
    model.mkdir()
    md.mkdir()
    (bundle / "model").mkdir(parents=True)
    (bundle / "space").mkdir(parents=True)
    for f in pub.MODEL_FILES:
        (model / f).write_text(f"icerik {f}", encoding="utf-8")
    for f in pub.BUNDLE_FILES:
        (bundle / "model" / f).write_text(f"paket {f}", encoding="utf-8")
    (bundle / "space" / "README.md").write_text(
        f"models:\n- <user>/{pub.REPO_NAME}\n", encoding="utf-8"
    )
    shas = {f: sha256_file(model / f) for f in pub.MODEL_FILES}
    chk = {"split": "split_v2", "passed": True,
           "models": {pub.MODEL_SLOT: {"model_files_sha256": shas}}}  # fmt: skip
    (md / "phase5_model_check_20260101-000000.json").write_text(json.dumps(chk), encoding="utf-8")
    return {"model": model, "md": md, "bundle": bundle, "tmp": tmp_path}


class FakeApi:
    def __init__(self, user="ali", private=True, files=None):
        self.user, self.private, self.calls, self.files = user, private, [], files or {}

    def whoami(self):
        self.calls.append(("whoami",))
        return {"name": self.user}

    def create_repo(self, rid, **kw):
        self.calls.append(("create_repo", rid, kw))

    def model_info(self, rid):
        return type("Info", (), {"private": self.private})()

    def create_commit(self, rid, ops, **kw):
        self.calls.append(("create_commit", rid, [o.path_in_repo for o in ops], kw))
        return type("Commit", (), {"oid": "a" * 40})()

    def hf_hub_download(self, rid, f, revision, local_dir):
        dst = Path(local_dir) / f
        shutil.copy(self.files[f], dst)
        return str(dst)


# --------------------------------------------------------------------------- kimlik, kural, metin
@pytest.mark.parametrize("bad", [None, "", "[KULLANICI_ADIN]", "<user>", "a b"])
def test_repo_id_rejects_missing_or_placeholder(bad):
    with pytest.raises(ValueError, match="kullanıcı adı"):
        pub.repo_id(bad)


def test_repo_id_ok():
    assert pub.repo_id("ali-1") == f"ali-1/{pub.REPO_NAME}"


def test_rules_text():
    R = pub.PUBLISH_RULES
    assert "PRIVATE" in R["visibility"] and "Space PUBLIC" in R["visibility"]
    assert "YAZMA yetkili token olmaz" in R["token"] and "fine-grained" in R["token"]
    assert "Haber metni alıntılama yapılmaz" in R["content"]
    assert "--confirm" in R["approval"]


def test_module_never_touches_token():
    tree = ast.parse(Path(inspect.getsourcefile(pub)).read_text(encoding="utf-8"))
    names = {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(tree)}
    assert not names & {"get_token", "HfFolder", "environ", "getenv", "add_space_secret",
                        "update_repo_visibility", "update_repo_settings"}  # fmt: skip


def test_notice_same_in_app_and_publish():
    assert _load_app().NOTICE == pub.SPACE_NOTICE


def test_check_text_rejects_banned():
    for bad in ("neredeyse anlamlı", "2010'ların başı", "tek bir kaynaktan"):
        with pytest.raises(ValueError, match="yasak"):
            pub.check_text(bad)


def test_bundle_from_real_metrics(tmp_path):
    a = pub.bundle(tmp_path / "a")
    b = pub.bundle(tmp_path / "b")
    assert a == b  # deterministik
    card = (tmp_path / "a" / "model" / "README.md").read_text(encoding="utf-8")
    musts = ("ayırt edilemedi", "T = 1.283", "license: other", "`license: unknown`",
             "kaggle.com/savasy/ttc4900", "dağıtım tercihi", "kemik.yildiz.edu.tr")  # fmt: skip
    for must in musts:
        assert must in card, must
    pub.check_text(card)
    cal = json.loads((tmp_path / "a" / "model" / "calibration.json").read_text(encoding="utf-8"))
    assert cal["temperature"] == pytest.approx(1.28259, abs=1e-4)
    space = (tmp_path / "a" / "space" / "README.md").read_text(encoding="utf-8")
    assert pub.SPACE_NOTICE in space and "sdk: gradio" in space


# --------------------------------------------------------------------------- yerel bütünlük
def test_check_local_ok_and_mismatch(fake):
    assert set(pub.check_local(fake["model"], fake["md"])) == set(pub.MODEL_FILES)
    (fake["model"] / "config.json").write_text("değişti", encoding="utf-8")
    with pytest.raises(ValueError, match="Faz 5"):
        pub.check_local(fake["model"], fake["md"])


@pytest.mark.skipif(not pub.MODEL_DIR.exists(), reason="yerel final model yok")
def test_check_local_real_model():
    shas = pub.check_local()
    assert shas["model.safetensors"].startswith("72cbeb438a89")


# --------------------------------------------------------------------------- 8b/8c kapıları
def test_upload_requires_confirm(fake):
    with pytest.raises(RuntimeError, match="--confirm"):
        pub.upload_model("ali", False, FakeApi(), fake["model"], fake["bundle"], fake["md"])


def test_upload_requires_matching_user(fake):
    with pytest.raises(RuntimeError, match="aynı değil"):
        pub.upload_model("veli", True, FakeApi("ali"), fake["model"], fake["bundle"], fake["md"])


def test_upload_private_and_exact_files(fake):
    api = FakeApi("ali")
    assert pub.upload_model("ali", True, api, fake["model"], fake["bundle"], fake["md"]) == "a" * 40
    create = next(c for c in api.calls if c[0] == "create_repo")
    assert create[2]["private"] is True and create[2]["repo_type"] == "model"
    commit = next(c for c in api.calls if c[0] == "create_commit")
    assert sorted(commit[2]) == sorted((*pub.MODEL_FILES, *pub.BUNDLE_FILES))


def test_upload_stops_if_repo_not_private(fake):
    with pytest.raises(RuntimeError, match="private değil"):
        pub.upload_model("ali", True, FakeApi("ali", private=False), fake["model"],
                         fake["bundle"], fake["md"])  # fmt: skip


def test_upload_stops_if_local_changed(fake):
    (fake["model"] / "model.safetensors").write_text("bozuk", encoding="utf-8")
    api = FakeApi("ali")
    with pytest.raises(ValueError, match="Faz 5"):
        pub.upload_model("ali", True, api, fake["model"], fake["bundle"], fake["md"])
    assert not [c for c in api.calls if c[0] in ("create_repo", "create_commit")]


def test_verify_remote_ok_and_mismatch(fake):
    files = {f: fake["model"] / f for f in pub.MODEL_FILES}
    files |= {f: fake["bundle"] / "model" / f for f in pub.BUNDLE_FILES}
    api = FakeApi(files=files)
    got = pub.verify_remote("ali", "b" * 40, api, fake["model"], fake["bundle"], fake["md"])
    assert set(got) == set(files)
    bad = fake["tmp"] / "bad.json"
    bad.write_text("başka", encoding="utf-8")
    api.files["calibration.json"] = bad
    with pytest.raises(ValueError, match="aynı değil"):
        pub.verify_remote("ali", "b" * 40, api, fake["model"], fake["bundle"], fake["md"])


def test_create_space_public_pinned_no_secret(fake):
    with pytest.raises(RuntimeError, match="--confirm"):
        pub.create_space("ali", "c" * 40, False, FakeApi(), bundle_dir=fake["bundle"])
    with pytest.raises(ValueError, match="commit sha"):
        pub.create_space("ali", "main", True, FakeApi(), bundle_dir=fake["bundle"])
    api = FakeApi("ali")
    assert pub.create_space("ali", "c" * 40, True, api, bundle_dir=fake["bundle"]) == "a" * 40
    create = next(c for c in api.calls if c[0] == "create_repo")
    assert create[1] == "ali/" + pub.SPACE_NAME
    assert create[2]["private"] is False and create[2]["space_sdk"] == "gradio"
    commit = next(c for c in api.calls if c[0] == "create_commit")
    assert sorted(commit[2]) == ["README.md", "app.py", "requirements.txt", "space_config.json"]
    assert not [c for c in api.calls if "secret" in c[0]]


# --------------------------------------------------------------------------- Space uygulaması
@pytest.mark.skipif(not pub.MODEL_DIR.exists(), reason="yerel final model yok")
def test_app_offline_matches_direct_forward(monkeypatch, tmp_path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pub.bundle(tmp_path)
    app = _load_app()
    cal = tmp_path / "model" / "calibration.json"
    tok, model, T, labels = app.load(str(pub.MODEL_DIR), calibration_path=str(cal))
    assert T == pytest.approx(1.28259, abs=1e-4)
    assert labels == ["siyaset", "dunya", "ekonomi", "kultur", "saglik", "spor", "teknoloji"]
    texts = ["merkez bankası faiz kararını açıkladı",
             "milli takım deplasmanda iki golle kazandı",
             "yeni akıllı telefon modeli tanıtıldı"]  # fmt: skip
    p = app.predict_proba(texts, tok, model, T)
    t2 = AutoTokenizer.from_pretrained(pub.MODEL_DIR)
    m2 = AutoModelForSequenceClassification.from_pretrained(pub.MODEL_DIR).eval()
    for i, t in enumerate(texts):  # bağımsız yol: tek tek, padding yok
        with torch.inference_mode():
            z = m2(**t2(t, truncation=True, max_length=512, return_tensors="pt")).logits[0]
        assert torch.allclose(p[i], torch.softmax(z / T, -1), atol=1e-5)
    out = app.classify(texts[0], (tok, model, T, labels))
    assert set(out) == set(labels) and sum(out.values()) == pytest.approx(1.0, abs=1e-5)
    assert app.classify("   ", (tok, model, T, labels)) == {}


def test_app_main_requires_calibration_with_model_dir():
    app = _load_app()
    with pytest.raises(SystemExit, match="--calibration"):
        app.main(["--model-dir", "herhangi"])


@pytest.mark.skipif(not pub.MODEL_DIR.exists(), reason="yerel final model yok")
def test_app_main_local_demo_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pub.bundle(tmp_path)
    app = _load_app()
    seen = {}

    class FakeUI:
        def launch(self, **kw):
            seen["launch"] = kw

    def fake_build(state):
        seen["state"] = state
        return FakeUI()

    monkeypatch.setattr(app, "build_ui", fake_build)
    app.main(["--model-dir", str(pub.MODEL_DIR),
              "--calibration", str(tmp_path / "model" / "calibration.json")])  # fmt: skip
    tok, model, T, labels = seen["state"]
    assert T == pytest.approx(1.28259, abs=1e-4) and len(labels) == 7
    assert seen["launch"] == {"share": False}

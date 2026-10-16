"""Faz 5 (final_eval) testleri: yalnız sahte veriyle; gerçek test seti hiç açılmaz."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ttc4900 import embeddings, final_eval, report, test_plan
from ttc4900.baselines import SPECS
from ttc4900.finetune import FINAL_RULES, FT_CONFIG
from ttc4900.paired_test import decide
from ttc4900.paths import PROJECT_ROOT

NAMES = ["siyaset", "dunya", "ekonomi", "kultur", "saglik", "spor", "teknoloji"]
K = len(NAMES)
N_TR, N_TE = 10, 4  # sınıf başına
FAKE_PARAMS = {
    "char_lr": {"clf__C": 100.0, "tfidf__ngram_range": [2, 4]},
    "char_lr_c1clean": {"clf__C": 100.0, "tfidf__ngram_range": [2, 4]},
    "word_lr": {"clf__C": 100.0, "tfidf__ngram_range": [1, 1]},
    "word_svc": {"clf__C": 1.0, "tfidf__ngram_range": [1, 1]},
    "word_cnb": {"clf__alpha": 0.1, "tfidf__ngram_range": [1, 1]},
}


# --------------------------------------------------------------------------- sahte dünya
def _frame(ids, labels, rng) -> pd.DataFrame:
    words = ["haber", "bugün", "dedi", "yeni", "büyük", "açıkladı", "\u0091alıntı\u0092"]
    texts = [
        f"konu{y} anahtar{y} " + " ".join(rng.choice(words, 6)) + f" ek{y % 3}" for y in labels
    ]
    return pd.DataFrame(
        {"row_id": np.asarray(ids, dtype=np.int64), "label": labels,
         "category": [NAMES[y] for y in labels], "text": texts}
    )  # fmt: skip


def _label_of(text: str) -> int:
    return int(re.search(r"konu(\d)", text)[1])


def _emb(texts) -> np.ndarray:
    E = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        E[i, _label_of(t)] = 1.0
        E[i, 8 + len(t) % 8] = 0.3
    return E


def _fake_logits(seed: int):
    """Sınıfı metinden okur; seed'e bağlı bir satır kümesini bir sonraki sınıfa kaydırır."""

    def fn(texts):
        L = np.zeros((len(texts), K), dtype=np.float32)
        for i, t in enumerate(texts):
            y = _label_of(t)
            wrong = (len(t) + seed) % 9 == 0
            L[i, (y + 1) % K if wrong else y] = 5.0
        return L

    return fn


def _fake_loader(local_dir, names):
    seed = int(Path(local_dir).name)
    info = {
        "id2label_ok": True,
        "tokenizer_check": {"vocab_sha256": "v"},
        "same_ids": lambda t: True,
    }
    return _fake_logits(seed), info


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    ytr = np.repeat(np.arange(K), N_TR)
    yte = np.repeat(np.arange(K), N_TE)
    ntr = len(ytr)
    train = _frame(np.arange(ntr), ytr, rng)
    test = _frame(np.arange(ntr, ntr + len(yte)), yte, rng)
    split = {
        "revision": "x",
        "label_names": NAMES,
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "components": [[ntr, ntr + 1], [ntr + 5, ntr + 6, ntr + 7], [0, 1]],
    }
    sp = tmp_path / "split_v2.json"
    sp.write_text(json.dumps(split), encoding="utf-8")
    md = tmp_path / "metrics"
    md.mkdir()
    stamp = "20261016-000000"
    for n, bp in FAKE_PARAMS.items():
        (md / f"phase2_{n}_{stamp}.json").write_text(json.dumps(
            {"split": "split_v2", "git_commit": "f2", "final": {"best_params": bp}}
        ), encoding="utf-8")  # fmt: skip
    (md / f"phase3_bge_m3_{stamp}.json").write_text(json.dumps(
        {"split": "split_v2", "git_commit": "f3", "final": {"best_params": {"clf__C": 1.0}}}
    ), encoding="utf-8")  # fmt: skip
    models = tmp_path / "models"
    slots = {}
    for seed in FT_CONFIG["seeds"]:
        d = models / str(seed)
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(f"agirlik-{seed}".encode())
        sha = {"model.safetensors": final_eval.sha256_file(d / "model.safetensors")}
        pred = _fake_logits(seed)(train["text"].tolist()).argmax(-1)
        from ttc4900.finetune import oof_metrics

        ins = oof_metrics(ytr, pred, NAMES)["macro_f1"]
        colab = f"/content/drive/MyDrive/ttc4900/faz4/models/{seed}"
        f = f"phase4_berturk_ft_final_s{seed}_{stamp}.json"
        (md / f).write_text(json.dumps(report.to_jsonable({
            "split": "split_v2", "ft_config": FT_CONFIG, "revision": FT_CONFIG["revision"],
            "final_collapse_rule": FINAL_RULES["final_collapse"], "model_dir": colab,
            "weights_sha256": sha, "train_in_sample_macro_f1": ins,
            "tokenizer_check": {"vocab_sha256": "v"}, "primary_for_test": seed == 42,
        })), encoding="utf-8")  # fmt: skip
        slots[str(seed)] = {"used_seed": seed, "used_file": f, "used_model_dir": colab}
    (md / f"phase4_summary_{stamp}.json").write_text(json.dumps({"split": "split_v2", "finals": {
        "slots": slots, "missing_slots": [],
        "primary_model_dir": slots["42"]["used_model_dir"],
        "primary_model_sha256": {"model.safetensors": final_eval.sha256_file(
            models / "42" / "model.safetensors")},
    }}), encoding="utf-8")  # fmt: skip
    opened = {"n": 0}

    def load_data(path):
        opened["n"] += 1
        return train.copy(), test.copy()

    return {
        "tmp": tmp_path, "split": sp, "md": md, "models": models, "preds": md / "preds",
        "cache": tmp_path / "cache", "train": train, "test": test, "opened": opened,
        "load_data": load_data,
    }  # fmt: skip


def _ready(md):
    return {"git_commit": "abc1234", "model_check": "x", "model_check_commit": "abc1234"}


def _predict(w, **kw):
    args = dict(
        metrics_dir=w["md"], models_dir=w["models"], preds_dir=w["preds"], out_dir=w["md"],
        path=w["split"], load_data=w["load_data"], train_loader=lambda p: w["train"].copy(),
        loader=_fake_loader, features=lambda n, df: _emb(df["text"].tolist()),
        encode=lambda t: (_emb(t), 0), cache_dir=w["cache"], env=dict, ready=_ready,
    )  # fmt: skip
    return final_eval.predict(**(args | kw))


def _verify(w, **kw):
    args = dict(metrics_dir=w["md"], models_dir=w["models"], out_dir=w["md"],
                train=w["train"], loader=_fake_loader, path=w["split"], env=dict)  # fmt: skip
    return final_eval.verify_models(**(args | kw))


# --------------------------------------------------------------------------- RULES['refit']
FORBIDDEN = ("GridSearchCV", "RandomizedSearchCV", "make_cv", "cross_val", "train_test_split",
             "StratifiedKFold", "nested_cv", "final_params")  # fmt: skip


def test_refit_has_no_search_or_cv():
    """Klasik ve bge_m3 modelleri yeniden seçilmez/ayarlanmaz: modülde arama/CV aracı yok."""
    src = inspect.getsource(final_eval)
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    code = re.sub(r'"[^"\n]*"', '""', code)  # kural metinlerindeki adlar sayılmaz
    for name in FORBIDDEN:
        assert name not in code, name


@pytest.mark.parametrize("name", [*FAKE_PARAMS, "bge_m3"])
def test_refit_uses_frozen_params_on_all_train(world, name, monkeypatch):
    pipe, info = final_eval.make_frozen(name, world["md"])
    want = final_eval._tuples(FAKE_PARAMS[name] if name in FAKE_PARAMS else {"clf__C": 1.0})
    assert info["best_params"] == want
    got = pipe.get_params()
    assert all(got[k] == v for k, v in want.items())
    seen = []
    orig = type(pipe).fit
    monkeypatch.setattr(
        type(pipe), "fit", lambda self, X, y: seen.append(len(X)) or orig(self, X, y)
    )
    tr = world["train"]
    X = _emb(tr["text"].tolist()) if name == "bge_m3" else tr["text"]
    final_eval.fit_frozen(pipe, X, tr["label"])
    assert seen == [len(tr)]  # tek fit, tüm train


def test_make_frozen_rejects_value_outside_grid(world):
    f = next(world["md"].glob("phase2_word_lr_*.json"))
    obj = json.loads(f.read_text(encoding="utf-8"))
    obj["final"]["best_params"]["clf__C"] = 3.0
    f.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="grid'de yok"):
        final_eval.make_frozen("word_lr", world["md"])


def test_real_frozen_params_match_protocol():
    """Gerçek Faz 2/3 kayıtları: v1 teşhisindeki char_lr ayarları ve grid üyeliği."""
    try:
        params = {n: final_eval.make_frozen(n)[1]["best_params"]
                  for n in (*final_eval.CLASSICAL, final_eval.ENCODER)}  # fmt: skip
    except FileNotFoundError:
        pytest.skip("Faz 2/3 metrik json'ları yok")
    assert params["char_lr"] == final_eval.V1_FROZEN
    assert set(params["word_svc"]) == set(SPECS["word_svc"].grid)


def test_candidates_cover_test_plan():
    plan = {m for f in test_plan.FAMILIES.values() for p in f["comparisons"] for m in p}
    plan |= {m for p in test_plan.DESCRIPTIVE for m in p}
    assert plan == set(final_eval.CANDIDATES)


def test_load_train_test_only_in_allowed_modules():
    """Kod düzeyinde (AST: ad/öznitelik/import; metinlerdeki geçişler sayılmaz) kullanım."""
    import ast

    allowed = {"data.py", "eda.py", "final_eval.py"}  # eda: Faz 1 veri raporu (tarihî)
    users = set()
    for f in (PROJECT_ROOT / "src" / "ttc4900").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = (
                [getattr(node, "id", None), getattr(node, "attr", None)]
                + [a.name for a in getattr(node, "names", [])]
                + [getattr(node, "name", None)]
            )
            if "load_train_test" in names:
                users.add(f.name)
    assert "final_eval.py" in users and users <= allowed


# --------------------------------------------------------------------------- verify
def test_verify_passes(world):
    _, out = _verify(world)
    assert out["passed"] and out["test"] is None
    assert set(out["models"]) == {"berturk_ft", "berturk_ft_s43", "berturk_ft_s44"}
    assert all(r["abs_diff"] == 0 for r in out["models"].values())
    assert world["opened"]["n"] == 0


def test_verify_fails_on_sha_mismatch(world):
    (world["models"] / "43" / "model.safetensors").write_bytes(b"bozuk")
    _, out = _verify(world)
    r = out["models"]["berturk_ft_s43"]
    assert not r["sha256_ok"] and not r["passed"] and "train_in_sample_macro_f1" not in r
    assert not out["passed"]


def test_verify_fails_beyond_tolerance(world):
    f = next(world["md"].glob("phase4_berturk_ft_final_s44_*.json"))
    obj = json.loads(f.read_text(encoding="utf-8"))
    obj["train_in_sample_macro_f1"] += 2 * final_eval.IN_SAMPLE_TOL
    f.write_text(json.dumps(obj), encoding="utf-8")
    _, out = _verify(world)
    assert not out["models"]["berturk_ft_s44"]["in_sample_ok"] and not out["passed"]


# --------------------------------------------------------------------------- predict ön koşulları
def test_predict_blocked_without_model_check(world, monkeypatch):
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234")
    with pytest.raises(RuntimeError, match="model_check"):
        _predict(world, ready=final_eval.check_predict_ready)
    assert world["opened"]["n"] == 0


def test_predict_blocked_when_check_failed_or_dirty(world, monkeypatch):
    (world["models"] / "42" / "model.safetensors").write_bytes(b"bozuk")
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234")
    _verify(world)  # geçmeyen kayıt
    with pytest.raises(RuntimeError, match="geçmedi"):
        _predict(world, ready=final_eval.check_predict_ready)
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234-dirty")
    with pytest.raises(RuntimeError, match="temiz değil"):
        _predict(world, ready=final_eval.check_predict_ready)
    assert world["opened"]["n"] == 0


def test_predict_blocked_when_code_changed_since_check(world, monkeypatch):
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234")
    _verify(world)
    monkeypatch.setattr(final_eval, "_src_changed_since", lambda c: ["src/ttc4900/x.py"])
    with pytest.raises(RuntimeError, match="değişmiş"):
        _predict(world, ready=final_eval.check_predict_ready)
    monkeypatch.setattr(final_eval, "_src_changed_since", lambda c: [])
    monkeypatch.setattr(final_eval, "_untracked_inputs", lambda: ["outputs/metrics/phase2_x.json"])
    with pytest.raises(RuntimeError, match="izlenmeyen"):
        _predict(world, ready=final_eval.check_predict_ready)
    assert world["opened"]["n"] == 0
    monkeypatch.setattr(final_eval, "_untracked_inputs", lambda: [])
    _, out = _predict(world, ready=final_eval.check_predict_ready)
    assert world["opened"]["n"] == 1 and out["model_check_commit"] == "abc1234"


def test_gate_ignores_only_phase5_outputs():
    ok = ["outputs/metrics/phase5_test_preds_20261016-120000.json",
          "outputs/metrics/phase5_model_check_20261016-120000.json",
          "outputs/metrics/preds/test_berturk_ft.npz"]  # fmt: skip
    bad = ["outputs/metrics/phase2_char_lr_20261016-120000.json", "src/ttc4900/x.py",
           "outputs/metrics/phase4_summary_20261016-120000.json", "outputs/splits/split_v2.json",
           "outputs/metrics/preds/cv_char_lr_20261011-192701.npz"]  # fmt: skip
    assert all(final_eval._PHASE5_OUTPUT.fullmatch(f) for f in ok)
    assert not any(final_eval._PHASE5_OUTPUT.fullmatch(f) for f in bad)


def test_predict_blocked_when_weights_changed_after_check(world):
    (world["models"] / "44" / "model.safetensors").write_bytes(b"sonradan")
    with pytest.raises(RuntimeError, match="sha256"):
        _predict(world)
    assert world["opened"]["n"] == 0


# --------------------------------------------------------------------------- write-once
def test_write_once_refuses_existing(tmp_path):
    p = tmp_path / "a.npz"
    final_eval.write_once(p, x=np.arange(3))
    with pytest.raises(FileExistsError):
        final_eval.write_once(p, x=np.arange(4))
    assert np.array_equal(np.load(p)["x"], np.arange(3))
    assert [f.name for f in tmp_path.iterdir()] == ["a.npz"]  # geçici dosya kalmadı


def test_predict_writes_all_once_and_rerun_keeps_files(world):
    _, out = _predict(world)
    assert set(out["files"]) == set(final_eval.CANDIDATES)
    assert all(e["status"] == "üretildi" for e in out["files"].values())
    shas = {n: e["sha256"] for n, e in out["files"].items()}
    mtimes = {n: final_eval.preds_path(n, world["preds"]).stat().st_mtime_ns for n in shas}
    # yeniden çalıştırma: hiçbir dosya yeniden üretilmez
    import time

    time.sleep(1.1)  # manifest adı saniye çözünürlüklü
    _, out2 = _predict(world)
    assert all(e["status"].startswith("vardı") for e in out2["files"].values())
    assert {n: e["sha256"] for n, e in out2["files"].items()} == shas
    assert {n: final_eval.preds_path(n, world["preds"]).stat().st_mtime_ns for n in shas} == mtimes
    # eksik bir model yalnız o üretilir
    with np.load(final_eval.preds_path("word_cnb", world["preds"])) as z:
        first_cnb = z["y_pred"]
    final_eval.preds_path("word_cnb", world["preds"]).unlink()
    time.sleep(1.1)
    _, out3 = _predict(world)
    assert out3["files"]["word_cnb"]["status"] == "üretildi"
    with np.load(final_eval.preds_path("word_cnb", world["preds"])) as z:
        assert np.array_equal(z["y_pred"], first_cnb)  # aynı donmuş fit (meta'da zaman farklı)
    assert sum(e["status"] == "üretildi" for e in out3["files"].values()) == 1


def test_predict_detects_tampered_file(world):
    _predict(world)
    p = final_eval.preds_path("char_lr", world["preds"])
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    arr["y_pred"] = (arr["y_pred"] + 1) % K
    p.unlink()
    np.savez(p, **arr)
    with pytest.raises(ValueError, match="sha256"):
        _predict(world)


def test_npz_contents(world):
    _predict(world)
    with np.load(final_eval.preds_path("berturk_ft", world["preds"])) as z:
        meta = json.loads(str(z["meta"]))
        assert z["scores"].shape == (len(world["test"]), K)
        assert np.array_equal(z["y_pred"], z["scores"].argmax(-1))
        assert np.array_equal(z["row_id"], world["test"]["row_id"])
    assert meta["kind"] == "phase4_final" and meta["split"] == "split_v2"
    with np.load(final_eval.preds_path("word_svc", world["preds"])) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["kind"] == "frozen_refit" and meta["n_fit"] == len(world["train"])
    assert meta["score_kind"] == "decision_function"


# --------------------------------------------------------------------------- test önbelleği
def test_test_cache_is_separate_and_rejected_as_train(world, monkeypatch):
    test, train = world["test"], world["train"]
    X, meta = final_eval.test_embeddings("bge_m3", test, lambda t: (_emb(t), 0), world["cache"])
    p = final_eval.test_cache_path("bge_m3", world["cache"])
    assert "_TEST_" in p.name and meta["side"] == "test"
    # aynı çağrı önbellekten okur; başka satırlarla reddeder
    X2, _ = final_eval.test_embeddings("bge_m3", test, lambda t: 1 / 0, world["cache"])
    assert np.array_equal(X, X2)
    with pytest.raises(ValueError, match="uyuşmuyor"):
        final_eval.test_embeddings("bge_m3", test.iloc[:-1], lambda t: 1 / 0, world["cache"])
    # test önbelleği train yükleyicisine verilirse reddedilir
    monkeypatch.setattr(embeddings, "CACHE_DIR", world["cache"])
    p.rename(embeddings.cache_path("bge_m3"))
    with pytest.raises(ValueError, match="test önbelleği"):
        embeddings.load_embeddings("bge_m3", train)


# --------------------------------------------------------------------------- evaluate
def test_evaluate_end_to_end(world):
    _predict(world)
    _, out = final_eval.evaluate(world["md"], world["preds"], world["md"], world["split"], B=200)
    assert out["split"] == "split_v2" and out["n_test"] == len(world["test"])
    assert out["families"]["primary"]["m"] == 1 and out["families"]["secondary"]["m"] == 4
    for r in out["families"]["secondary"]["comparisons"]:
        assert {"p", "p_holm", "verdict", "diff_ci95"} <= set(r)
    assert len(out["descriptive"]) == len(test_plan.DESCRIPTIVE)
    for d in out["descriptive"]:
        assert "p" not in d and "p_holm" not in d
    assert len(out["sentences"]["aileler"]["secondary"]) == 4
    sents = [*out["sentences"]["aileler"]["primary"], *out["sentences"]["aileler"]["secondary"],
             *out["sentences"]["betimsel"], out["sentences"]["seed"]]  # fmt: skip
    assert all("{" not in t and "}" not in t for t in sents)
    s = out["seed_summary"]
    assert s["models"] == ["berturk_ft", "berturk_ft_s43", "berturk_ft_s44"]
    assert all(f"{n} {v:.4f}" in out["sentences"]["seed"] for n, v in s["macro_f1"].items())
    assert "güven aralığı değildir" in out["sentences"]["seed"]
    assert out["sentences"]["sinirliliklar"] == test_plan.SENTENCES["sinirliliklar"]
    assert "seed_tutarlilik" in out["sentences"] and "{" not in out["sentences"]["seed_tutarlilik"]
    assert all("accuracy_ci95" in m for m in out["models"].values())
    assert all("acc_diff_ci95" in r for r in out["families"]["primary"]["comparisons"])
    assert all("acc_diff_ci95" in d for d in out["descriptive"])
    # aynı w'ler: betimsel char_lr CI'ı ile ikincil ailedeki char_lr CI'ı aynı replikatlardan
    sec = out["families"]["secondary"]["comparisons"][0]
    assert sec["a"] == "char_lr"
    assert sec["macro_f1_a_ci95"] == out["models"]["char_lr"]["macro_f1_ci95"]
    prim = out["sentences"]["aileler"]["primary"][0]
    assert test_plan.SENTENCES["birincil_kosul"] in prim
    assert prim.endswith(out["sentences"]["seed_tutarlilik"])  # karar: birincil cümlenin parçası


def test_evaluate_requires_all_preds(world):
    _predict(world)
    final_eval.preds_path("bge_m3", world["preds"]).unlink()
    with pytest.raises(FileNotFoundError, match="predict"):
        final_eval.evaluate(world["md"], world["preds"], world["md"], world["split"], B=50)


# --------------------------------------------------------------------------- cümle şablonları
def _row(p_holm, ci, n10, n01, diff):
    r = {"a": "berturk_ft", "b": "char_lr", "diff": diff, "diff_ci95": ci, "n10": n10,
         "n01": n01, "p": p_holm, "p_holm": p_holm}  # fmt: skip
    return r | decide(p_holm, ci, n10, n01)


@pytest.mark.parametrize(
    "args, verdict, phrase",
    [
        ((0.001, [0.01, 0.03], 40, 15, 0.02), "ayırt edildi", "daha yüksek olarak ayırt edildi"),
        ((0.001, [-0.03, -0.01], 15, 40, -0.02), "ayırt edildi", "daha düşük olarak ayırt edildi"),
        (
            (0.001, [-0.03, -0.01], 40, 15, -0.02),
            "ayırt edilemedi (yön çelişkisi)",
            "yön çelişkisi",
        ),
        ((0.30, [-0.01, 0.02], 30, 25, 0.005), "ayırt edilemedi", "eşdeğerlik iddiası değildir"),
        ((0.001, [-0.01, 0.02], 40, 15, 0.005), "ayırt edilemedi", "sağlanmayanlar: CI 0'ı"),
    ],
)
def test_sentence_for_each_verdict(args, verdict, phrase):
    r = _row(*args)
    assert r["verdict"] == verdict
    s = final_eval.render_comparison(r, "ikincil", 4, None)
    assert phrase in s and "{" not in s and "m=4" in s
    p = final_eval.render_comparison(r, "birincil", 1, None)
    assert p.endswith(test_plan.SENTENCES["birincil_kosul"])


def test_not_distinguished_sentence_lists_conditions_and_power():
    r = _row(0.30, [-0.01, 0.02], 30, 25, 0.005)
    pw = {"_file": "test_power_x.json", "oof_acc_diff_mean": 0.0079, "by_alpha": {
        "0.0125": {"power_at_oof_acc_diff": 0.05, "mde_acc": 0.026, "mde_macro_f1_rough": 0.025},
        "0.05": {"power_at_oof_acc_diff": 0.15, "mde_acc": 0.022, "mde_macro_f1_rough": 0.021},
    }}  # fmt: skip
    s = final_eval.render_comparison(r, "ikincil", 4, pw)
    assert "Sağlanan koşullar: hiçbiri" in s
    # CI 0'ı içerirken yön tutarlılığı "sağlanmadı" değil, "değerlendirilemez"
    assert ("sağlanmayanlar: Holm p < 0.05, CI 0'ı dışlıyor, McNemar ve CI yönleri tutarlı "
            "(CI 0'ı içerdiği için değerlendirilemez)") in s  # fmt: skip
    # her α kendi sayılarıyla, Holm adımı etiketli
    assert ("α/4 = 0.0125 (Holm'un en kötü adımı): güç 0.05, accuracy MDE 0.026, macro-F1 MDE "
            "(kaba) 0.025") in s  # fmt: skip
    assert "α = 0.05 (Holm'un en iyi adımı): güç 0.15, accuracy MDE 0.022" in s
    assert "+0.0079" in s and "yaklaşık" in s
    one = {"_file": "f", "oof_acc_diff_mean": 0.01, "by_alpha": {"0.05": pw["by_alpha"]["0.05"]}}
    s1 = final_eval.render_comparison(r, "birincil", 1, one)
    assert "α = 0.05: güç 0.15" in s1 and "Holm" not in s1.split("Test öncesi")[1]


def test_conflict_sentence_names_models():
    r = _row(0.001, [-0.03, -0.01], 40, 15, -0.02)
    s = final_eval.render_comparison(r, "ikincil", 4, None)
    assert "McNemar (accuracy) berturk_ft lehine" in s and "CI'ı ise char_lr lehine" in s


@pytest.mark.parametrize(
    "args, held, unmet",
    [
        # CI alt sınırı tam 0: "dışlıyor" sayılmaz (muhafazakâr)
        ((0.001, [0.0, 0.03], 40, 15, 0.015), ["Holm p < 0.05"], "CI 0'ı dışlıyor"),
        # n10 = n01: p = 1, McNemar yönsüz; CI 0'ı dışlasa bile ayırt edilemedi
        ((1.0, [0.001, 0.02], 20, 20, 0.01), ["CI 0'ı dışlıyor"], "Holm p < 0.05"),
    ],
)
def test_edge_cases(args, held, unmet):
    r = _row(*args)
    assert r["verdict"] == "ayırt edilemedi"
    s = final_eval.render_comparison(r, "ikincil", 4, None)
    assert f"Sağlanan koşullar: {', '.join(held)};" in s and unmet in s.split("sağlanmayanlar")[1]


def test_accuracy_bootstrap_uses_same_weights():
    """accuracy replikatları cluster_bootstrap'in w'leriyle aynı: ikisi de bootstrap_weights."""
    from ttc4900.paired_test import (
        bootstrap_weights,
        cluster_bootstrap,
        confusion,
        group_index,
        macro_f1_from_confusion,
    )

    rng = np.random.default_rng(1)
    y = rng.integers(0, K, 60)
    preds = {"a": np.where(rng.random(60) < 0.8, y, (y + 1) % K), "b": rng.integers(0, K, 60)}
    g = np.repeat(np.arange(30), 2)
    acc = final_eval.accuracy_bootstrap(y, preds, g, B=50, seed=3)
    f1 = cluster_bootstrap(y, preds, g, B=50, seed=3)
    inv, G = group_index(g)
    for start, w in bootstrap_weights(G, 50, 3):
        for i, wi in enumerate(w):
            rows = np.repeat(np.arange(60), wi[inv])
            for m, p in preds.items():
                assert acc[m][start + i] == pytest.approx((p[rows] == y[rows]).mean())
                assert f1[m][start + i] == pytest.approx(
                    macro_f1_from_confusion(confusion(y[rows], p[rows]))
                )


# --------------------------------------------------------------------------- v1 teşhisi
def _v1_setup(w):
    """v1: train'den 0, test'ten ntr+5 yer değiştirir. Bileşenler (fake split_v2):
    [ntr, ntr+1] tamamen test (temiz), [ntr+5, ntr+6, ntr+7]: ntr+5 v1 train'de -> ntr+6, ntr+7
    sızmış; [0, 1]: 0 v1 test'te, 1 train'de -> 0 sızmış."""
    tr, te = w["train"], w["test"]
    ntr = len(tr)
    tr_ids = [i for i in tr["row_id"].tolist() if i != 0] + [ntr + 5]
    te_ids = [i for i in te["row_id"].tolist() if i != ntr + 5] + [0]
    split1 = {"revision": "x", "label_names": NAMES, "train_ids": tr_ids, "test_ids": te_ids}
    p1 = w["tmp"] / "split_v1.json"
    p1.write_text(json.dumps(split1), encoding="utf-8")
    full = pd.concat([tr, te]).set_index("row_id", drop=False)
    opened = w["opened"]

    def load_v1(path):
        opened["n"] += 1
        return full.loc[tr_ids].reset_index(drop=True), full.loc[te_ids].reset_index(drop=True)

    return p1, load_v1, {0, ntr + 6, ntr + 7}


def test_leaked_rows_component_definition():
    comps = [[1, 10, 11], [12, 13], [2, 3]]
    # 10, 11: bileşende train (1) var -> sızmış (11'in 1 ile doğrudan kenarı olmasa da: zincir)
    # 12-13 yalnız test -> temiz; 14 tekil -> temiz; 2-3 yalnız train
    assert final_eval.leaked_rows([1, 2, 3], [10, 11, 12, 13, 14], comps) == {10, 11}
    g = final_eval.v1_groups([10, 11, 12, 13, 14], comps)
    assert g.tolist() == [1, 1, 12, 12, -15]


def test_group_subset_bootstrap_matches_manual_weights():
    from ttc4900.paired_test import bootstrap_weights, group_index

    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(20), [1, 2] * 10)
    n = len(groups)
    y, pred = rng.integers(0, K, n), rng.integers(0, K, n)
    pred[:15] = y[:15]
    mask = np.isin(groups, np.arange(0, 20, 3))
    d, dropped = final_eval.group_subset_bootstrap(y, pred, groups, mask, B=40, seed=5)
    inv, G = group_index(groups)
    want = []
    for _, w in bootstrap_weights(G, 40, 5):
        for wi in w:
            rows = np.repeat(np.arange(n), wi[inv])
            m = mask[rows]
            if m.any() and (~m).any():
                ok = y[rows] == pred[rows]
                want.append(ok[m].mean() - ok[~m].mean())
    assert dropped == 40 - len(want)
    assert np.allclose(d, want)


def test_group_subset_bootstrap_rejects_split_group():
    with pytest.raises(ValueError, match="bölünmüş"):
        final_eval.group_subset_bootstrap(
            [0, 1, 2], [0, 1, 2], [7, 7, 8], [True, False, False], B=5
        )


def test_v1_diagnosis_runs_only_after_main_results(world, monkeypatch):
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(final_eval, "_is_tracked", lambda p: True)
    monkeypatch.setattr(final_eval, "_untracked_inputs", lambda: [])
    p1, load_v1, leaked = _v1_setup(world)
    kw = dict(metrics_dir=world["md"], preds_dir=world["preds"], out_dir=world["md"],
              load_data=load_v1, v1_path=p1, groups_path=world["split"], B=100)  # fmt: skip
    with pytest.raises(RuntimeError, match="SONRA"):
        final_eval.v1_diagnosis(**kw)
    assert world["opened"]["n"] == 0
    _predict(world)
    final_eval.evaluate(world["md"], world["preds"], world["md"], world["split"], B=50)
    _, out = final_eval.v1_diagnosis(**kw)
    assert out["split"] == "split_v1"
    assert out["frozen_params"]["best_params"] == final_eval.V1_FROZEN
    assert out["subsets"]["leaked"]["n"] == len(leaked)
    assert out["subsets"]["leaked"]["n_groups"] == 2  # [ntr+5..7] ve [0, 1] bileşenleri
    assert sum(out["subsets"]["leaked"]["class_counts"].values()) == len(leaked)
    assert "p" not in out["diagnostic"] and "macro_f1_leaked_minus_clean" not in out["diagnostic"]
    assert "macro_f1" not in out["subsets"]["leaked"]
    assert "macro_f1_ci95" in out["overall_v1_test"] and "accuracy_ci95" in out["overall_v1_test"]
    assert out["v2_reference"]["char_lr_v2_test_macro_f1_ci95"] is not None
    with pytest.raises(FileExistsError):  # tahmin dosyası da write-once
        final_eval.v1_diagnosis(**kw)


def test_v1_diagnosis_requires_clean_tree(world, monkeypatch):
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234-dirty")
    p1, load_v1, _ = _v1_setup(world)
    with pytest.raises(RuntimeError, match="temiz"):
        final_eval.v1_diagnosis(world["md"], world["preds"], world["md"], load_v1, p1,
                                world["split"], B=10)  # fmt: skip
    assert world["opened"]["n"] == 0


# --------------------------------------------------------------------------- seed tutarlılığı
def _pair(a, ci, diff):
    return {"a": a, "b": "char_lr", "diff": diff, "diff_ci95": ci}


def test_seed_consistency_sentence():
    prim = _pair("berturk_ft", [0.005, 0.03], 0.017)
    others = [_pair("berturk_ft_s43", [0.002, 0.025], 0.013),
              _pair("berturk_ft_s44", [-0.004, 0.02], 0.008)]  # fmt: skip
    s = final_eval.render_seed_consistency(prim, others)
    assert "birincil farkın %95 CI'ı berturk_ft lehine" in s
    assert "berturk_ft_s43 − char_lr fark +0.0130 [+0.0020, +0.0250]: aynı yön" in s
    assert "berturk_ft_s44 − char_lr fark +0.0080 [-0.0040, +0.0200]: CI 0'ı içeriyor" in s
    assert "{" not in s and "karara girmez" in s
    s2 = final_eval.render_seed_consistency(prim, [_pair("berturk_ft_s43", [-0.03, -0.01], -0.02)])
    assert "ters yön" in s2
    s3 = final_eval.render_seed_consistency(_pair("berturk_ft", [-0.01, 0.02], 0.005), others)
    assert "işaretsiz" in s3 and "birincil CI işaretsiz" in s3


def test_v1_diagnosis_requires_committed_main_results(world, monkeypatch):
    monkeypatch.setattr(final_eval.report, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(final_eval, "_untracked_inputs", lambda: [])
    p1, load_v1, _ = _v1_setup(world)
    _predict(world)
    final_eval.evaluate(world["md"], world["preds"], world["md"], world["split"], B=20)
    n0 = world["opened"]["n"]
    kw = dict(metrics_dir=world["md"], preds_dir=world["preds"], out_dir=world["md"],
              load_data=load_v1, v1_path=p1, groups_path=world["split"], B=10)  # fmt: skip
    with pytest.raises(RuntimeError, match="commit"):  # tmp_path proje dışında: izlenmiyor
        final_eval.v1_diagnosis(**kw)
    monkeypatch.setattr(final_eval, "_is_tracked", lambda p: True)
    monkeypatch.setattr(final_eval, "_untracked_inputs", lambda: ["outputs/metrics/phase2_x.json"])
    with pytest.raises(RuntimeError, match="izlenmeyen"):
        final_eval.v1_diagnosis(**kw)
    assert world["opened"]["n"] == n0


# --------------------------------------------------------------------------- kapı ayrıntıları
def test_gate_filter_exempts_only_added_phase5_outputs():
    lines = [
        "A	outputs/metrics/phase5_model_check_20261016-120000.json",
        "M	outputs/metrics/phase5_model_check_20261016-110000.json",
        "D	outputs/metrics/phase5_test_preds_20261016-110000.json",
        "A	outputs/metrics/preds/test_char_lr.npz",
        "A	outputs/metrics/phase2_char_lr_20261016-120000.json",
        "R100	outputs/metrics/a.json	outputs/metrics/phase5_test_20261016-120000.json",
        "M	src/ttc4900/final_eval.py",
    ]
    assert final_eval.gate_filter(lines) == [
        "outputs/metrics/phase5_model_check_20261016-110000.json",
        "outputs/metrics/phase5_test_preds_20261016-110000.json",
        "outputs/metrics/phase2_char_lr_20261016-120000.json",
        "outputs/metrics/phase5_test_20261016-120000.json",
        "src/ttc4900/final_eval.py",
    ]


def test_predict_blocked_when_model_files_changed_after_verify(world):
    _verify(world)
    (world["models"] / "43" / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="model dosyaları"):
        _predict(world)
    assert world["opened"]["n"] == 0


def test_cluster_bootstrap_weights_unchanged_by_refactor():
    """bootstrap_weights refaktöründen önceki döngüyle (test_plan'daki şema) birebir aynı w."""
    from ttc4900.paired_test import _group_confusions, cluster_bootstrap, macro_f1_from_confusion

    rng = np.random.default_rng(4)
    y = rng.integers(0, K, 90)
    preds = {"a": rng.integers(0, K, 90), "b": np.where(rng.random(90) < 0.7, y, 0)}
    groups = rng.integers(0, 40, 90)
    B, seed, chunk = 2500, 42, 1000  # parça sınırını geçer
    _, inv = np.unique(groups, return_inverse=True)
    G = int(inv.max()) + 1
    flats = {m: _group_confusions(y, p, inv, G, K) for m, p in preds.items()}
    r = np.random.default_rng(seed)
    old = {m: np.empty(B) for m in preds}
    for start in range(0, B, chunk):
        n = min(chunk, B - start)
        w = r.multinomial(G, np.full(G, 1 / G), size=n)
        for m, F in flats.items():
            old[m][start : start + n] = macro_f1_from_confusion((w @ F).reshape(n, K, K))
    new = cluster_bootstrap(y, preds, groups, B, seed, K, chunk)
    for m in preds:
        assert np.array_equal(old[m], new[m])


def test_group_subset_bootstrap_drops_empty_replicates():
    # 3 grup, sızmış 1 grup: P(w_L = 0) = (2/3)^3 ≈ 0.30 -> bazı replikatlar düşer
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    groups = np.array([5, 5, 6, 6, 7, 7])
    mask = groups == 5
    d, dropped = final_eval.group_subset_bootstrap(y, pred, groups, mask, B=300, seed=0)
    assert dropped > 0 and len(d) == 300 - dropped and np.all(np.isfinite(d))


# --------------------------------------------------------------------------- keşifsel
def test_exploratory_after_main_results_no_p(world):
    with pytest.raises(RuntimeError, match="SONRA"):
        final_eval.exploratory(world["preds"], world["md"], world["split"], B=50)
    _predict(world)
    final_eval.evaluate(world["md"], world["preds"], world["md"], world["split"], B=50)
    _, out = final_eval.exploratory(world["preds"], world["md"], world["split"], B=50)
    (d,) = out["pairs"]
    assert (d["a"], d["b"]) == ("berturk_ft", "char_lr_c1clean")
    assert "p" not in d and "p_holm" not in d and "verdict" not in d
    assert out["label"] == "keşifsel, sonradan" and "sonradan" in d["inference"]
    assert {"diff", "diff_ci95", "acc_diff_ci95"} <= set(d)
    assert ("berturk_ft", "char_lr_c1clean") not in final_eval.DESCRIPTIVE
    with pytest.raises(ValueError, match="aynı değil"):  # farklı w'ler reddedilir
        final_eval.exploratory(world["preds"], world["md"], world["split"], B=50, seed=7)

"""Faz 4: dondurulmuş FT_CONFIG'in tutarlılığı (4a) ve eğitim kodunun saf parçaları (4b).

Ağ ve GPU gerekmez. Gerçek split ve char_lr npz'i repo dosyalarından okunur; train'in
row_id/label sütunları char_lr npz'inden kurulur (load_train() ile aynı sıra, testte doğrulanır).
test_ids yalnız id düzeyinde ayrıklık kontrolünde okunur.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ttc4900 import finetune, paired_test, report, test_plan
from ttc4900.cv_compare import latest_metrics, nb_mean_ci
from ttc4900.data import RANDOM_STATE, SPLIT_PATH, cv_groups, load_split
from ttc4900.embeddings import ENCODERS, row_ids_hash
from ttc4900.paths import METRICS_DIR, PROJECT_ROOT


def test_same_model_and_revision_as_phase3():
    c = finetune.FT_CONFIG
    assert c["model_id"] == "dbmdz/bert-base-turkish-cased"
    assert c["revision"] == ENCODERS["berturk_mean"].revision
    assert len(c["revision"]) == 40  # tam commit SHA
    assert c["max_length"] == 512


def test_batch_and_seeds_consistent():
    c = finetune.FT_CONFIG
    assert c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] == 16
    assert c["effective_batch_size"] == 16
    assert c["primary_seed"] == test_plan.PRIMARY_SEED in c["seeds"]
    assert len(set(c["seeds"])) == 3


def test_cv_matches_char_lr_repeat0():
    c = finetune.FT_CONFIG
    assert c["cv_random_state"] == RANDOM_STATE and c["cv_n_splits"] == 5
    assert c["val_fold"] < c["val_n_splits"]


def test_primary_family_names_this_model():
    assert (finetune.NAME, finetune.REFERENCE) in test_plan.FAMILIES["primary"]["comparisons"]


FAZ4A_COMMIT = "81381ee"


def test_frozen_dicts_identical_to_faz4a_commit():
    """FT_CONFIG / RULES / LITERATURE Faz 4a commit'indekiyle birebir aynı (git yoksa atla)."""
    try:
        src = subprocess.run(
            ["git", "show", f"{FAZ4A_COMMIT}:src/ttc4900/finetune.py"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, encoding="utf-8",
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git ya da Faz 4a commit'i yok (sığ klon?)")
    ns: dict = {}
    exec(compile(src, "finetune_faz4a.py", "exec"), ns)
    assert ns["FT_CONFIG"] == finetune.FT_CONFIG
    assert ns["RULES"] == finetune.RULES
    assert ns["LITERATURE"] == finetune.LITERATURE


FAZ4A_EK_COMMIT = "65aaed5"


def test_final_rules_identical_to_faz4a_ek_commit():
    """FINAL_RULES ve TEST_PLAN['primary_seed_rule'] Faz 4a eki commit'indekiyle aynı."""
    try:
        src = subprocess.run(
            ["git", "show", f"{FAZ4A_EK_COMMIT}:src/ttc4900/finetune.py"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, encoding="utf-8",
        ).stdout  # fmt: skip
        plan_src = subprocess.run(
            ["git", "show", f"{FAZ4A_EK_COMMIT}:src/ttc4900/test_plan.py"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, encoding="utf-8",
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git ya da Faz 4a eki commit'i yok (sığ klon?)")
    ns: dict = {}
    exec(compile(src, "finetune_faz4a_ek.py", "exec"), ns)
    assert ns["FINAL_RULES"] == finetune.FINAL_RULES
    plan_ns: dict = {}
    exec(compile(plan_src, "test_plan_faz4a_ek.py", "exec"), plan_ns)
    assert plan_ns["TEST_PLAN"]["primary_seed_rule"] == test_plan.TEST_PLAN["primary_seed_rule"]


# =========================================================================== Faz 4b
@pytest.fixture(scope="module")
def ref():
    try:
        return finetune.load_reference(METRICS_DIR)
    except FileNotFoundError:
        pytest.skip("char_lr referans json/npz'i yok")


@pytest.fixture(scope="module")
def split():
    return load_split(SPLIT_PATH)


@pytest.fixture(scope="module")
def train(ref, split):
    """load_train()'in row_id ve label sütunları, ağsız (metin gerekmez)."""
    assert ref["row_id"].tolist() == split["train_ids"]  # load_train() sırası
    return pd.DataFrame({"row_id": ref["row_id"], "label": ref["y_true"]})


@pytest.fixture(scope="module")
def plans(train):
    return finetune.make_plan(train)


def test_validation_split_is_leak_free(train, plans, split):
    g = cv_groups(train)
    ids = train["row_id"].to_numpy()
    test_ids = set(split["test_ids"])  # yalnız id düzeyinde
    assert [p.fold for p in plans] == [0, 1, 2, 3, 4]
    for p in plans:
        fit, val, otr, ova = (set(ids[i]) for i in (p.fit, p.val, p.outer_train, p.outer_val))
        # fit ve val'in yakın-tekrar grupları ayrık (ve dış train/val'inki de)
        assert not set(g[p.fit]) & set(g[p.val])
        assert not set(g[p.outer_train]) & set(g[p.outer_val])
        # val ⊂ dış train, fit ∪ val = dış train, fit ∩ val = ∅
        assert val <= otr and fit | val == otr and not fit & val
        assert not val & ova  # val ∩ dış val = ∅
        # Not: bu fixture split'in train_ids'inden kurulduğu için aşağıdaki kontrol yapı gereği
        # sağlanır (yalnız split dosyasının train/test ayrıklığını yeniden doğrular). Anlamlı
        # kontrol, make_plan'ın test satırı içeren girdiyi REDDETMESİDİR:
        # test_make_plan_rejects_test_rows.
        assert not (fit | val | ova) & test_ids
        assert 560 <= len(val) <= 600 and 2300 <= len(fit) <= 2345  # ~581 / ~2323


def test_make_plan_rejects_test_rows(train, split):
    """make_plan (cv_groups üzerinden) split'in train'inde olmayan satırı kabul etmez: tek bir
    test id'si eklenmiş girdide ValueError. Böylece hiçbir fold/validation parçası test satırı
    içeremez (yalnız id düzeyinde; metin ve etiket okunmaz)."""
    extra = pd.DataFrame({"row_id": [split["test_ids"][0]], "label": [0]})
    with pytest.raises(ValueError, match="train"):
        finetune.make_plan(pd.concat([train, extra], ignore_index=True))


def test_outer_folds_match_char_lr_repeat0(train, plans, ref, split):
    assert np.array_equal(finetune.fold_assignment(plans, len(train)), ref["fold_id0"])
    finetune.check_reference(train, plans, ref, split["label_names"])  # hata atmamalı
    shuffled = train.sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="satır sırası"):
        finetune.check_reference(shuffled, finetune.make_plan(shuffled), ref, split["label_names"])
    swapped = [finetune.FoldPlan((p.fold + 1) % 5, *[getattr(p, a) for a in
               ("outer_train", "outer_val", "fit", "val")]) for p in plans]  # fmt: skip
    with pytest.raises(ValueError, match="fold"):
        finetune.check_reference(train, swapped, ref, split["label_names"])


def test_label_encoding_matches_char_lr(train, plans, ref, split):
    names = split["label_names"]
    id2label, label2id = finetune.label_maps(names)
    assert names == ref["label_names"] and len(names) == finetune.FT_CONFIG["num_labels"]
    assert [id2label[i] for i in range(len(names))] == names
    assert all(label2id[id2label[i]] == i for i in id2label)
    df = train.assign(category=[names[k] for k in train["label"]])
    finetune.check_labels(df, names)  # category -> label2id == label
    with pytest.raises(ValueError):
        finetune.check_labels(df.assign(category=df["category"].iloc[::-1].to_numpy()), names)
    with pytest.raises(ValueError, match="Etiket adları"):
        finetune.check_reference(train, plans, ref, names[::-1])
    perm = (train["label"] + 1) % len(names)
    with pytest.raises(ValueError, match="kodlaması"):
        finetune.check_reference(train.assign(label=perm), plans, ref, names)


def test_smoke_subset_keeps_relations(plans):
    for p in plans:
        s = finetune.smoke_subset(p)
        assert len(s.fit) + len(s.val) + len(s.outer_val) <= 64
        assert set(s.fit) <= set(p.fit) and set(s.val) <= set(p.val)
        assert set(s.outer_val) <= set(p.outer_val)


def test_e_final_median():
    assert finetune.e_final([3] * 7 + [2] * 4 + [4] * 4) == 3  # 15 koşu
    assert finetune.e_final([1, 2, 2, 5, 5]) == 2
    assert finetune.e_final([2, 3]) == 2  # çift sayıda (yalnız smoke): alt medyan
    for bad in ([], [0], [6]):
        with pytest.raises(ValueError):
            finetune.e_final(bad)


def test_collapse_selection():
    assert finetune.is_collapsed(0.4999) and not finetune.is_collapsed(0.5)
    runs = {
        (0, 42): {"collapsed": False, "id": "a"},
        (0, 43): {"collapsed": True, "id": "b"},
        (0, 143): {"collapsed": False, "id": "c"},
        (1, 42): {"collapsed": True, "id": "d"},
    }
    used, missing, collapsed = finetune.select_used(runs, [(0, 42), (0, 43), (1, 42), (1, 43)])
    assert used[(0, 42)]["id"] == "a" and used[(0, 43)]["id"] == "c"  # yeniden koşu kullanılır
    assert missing == [(1, 142), (1, 43)]
    assert collapsed == [(0, 43), (1, 42)]
    runs[(1, 142)] = {"collapsed": True, "id": "e"}  # yeniden koşu da çöktü: yine o kullanılır
    used, missing, _ = finetune.select_used(runs, [(1, 42)])
    assert used[(1, 42)]["id"] == "e" and not missing


def test_val_curve_and_best_epoch():
    hist = [
        {"loss": 1.5, "epoch": 1.0},
        {"eval_macro_f1": 0.80, "eval_accuracy": 0.81, "eval_loss": 0.6, "epoch": 1.0},
        {"loss": 0.7, "epoch": 2.0},
        {"eval_macro_f1": 0.90, "eval_accuracy": 0.90, "eval_loss": 0.4, "epoch": 2.0},
        {"loss": 0.3, "epoch": 3.0},
        {"eval_macro_f1": 0.90, "eval_accuracy": 0.91, "eval_loss": 0.5, "epoch": 3.0},
        {"train_loss": 0.8, "train_runtime": 1.0, "epoch": 3.0},
    ]
    curve = finetune.val_curve(hist)
    assert [r["epoch"] for r in curve] == [1, 2, 3]
    assert curve[0]["train_loss"] == 1.5 and curve[2]["val_loss"] == 0.5
    assert finetune.best_epoch(curve) == (2, 0.90)  # eşitlikte ilk (Trainer'ın '>' kuralı)


# --------------------------------------------------------------------------- kayıt denetimleri
def _rec(**kw):
    base = {
        "_file": "x.json",
        "ft_config": report.to_jsonable(finetune.FT_CONFIG),
        "revision": finetune.FT_CONFIG["revision"],
        "best_val_macro_f1": 0.9,
        "trainer_best_metric": 0.9,
        "val_macro_f1_reloaded": 0.9004,
        "final_collapse_rule": finetune.FINAL_RULES["final_collapse"],
    }
    return base | kw


def test_check_config_and_best_checkpoint():
    finetune.check_config(_rec())
    bad_cfg = report.to_jsonable(finetune.FT_CONFIG) | {"learning_rate": 3e-5}
    with pytest.raises(ValueError, match="ft_config"):
        finetune.check_config(_rec(ft_config=bad_cfg))
    with pytest.raises(ValueError, match="revision"):
        finetune.check_config(_rec(revision="0" * 40))
    finetune.check_best_checkpoint(_rec())
    with pytest.raises(ValueError, match="best_metric"):
        finetune.check_best_checkpoint(_rec(trainer_best_metric=0.8))
    with pytest.raises(ValueError, match="yüklenen"):
        finetune.check_best_checkpoint(_rec(val_macro_f1_reloaded=0.85))  # son epoch modeli


def test_find_runs_and_finals_reject_duplicates(tmp_path):
    def w(name):
        (tmp_path / name).write_text(json.dumps({"smoke": False}), encoding="utf-8")

    w(f"phase4_{finetune.NAME}_f0_s42_20260101-000000.json")
    w(f"phase4_{finetune.NAME}_final_s42_20260101-000000.json")
    assert set(finetune.find_runs(tmp_path)) == {(0, 42)}
    assert set(finetune.find_finals(tmp_path)) == {42}
    w(f"phase4_{finetune.NAME}_f0_s42_20260102-000000.json")
    w(f"phase4_{finetune.NAME}_final_s42_20260102-000000.json")
    with pytest.raises(ValueError, match="birden çok"):
        finetune.find_runs(tmp_path)
    with pytest.raises(ValueError, match="birden çok"):
        finetune.find_finals(tmp_path)


# --------------------------------------------------------------------------- summarize (sahte)
NAMES = ["siyaset", "dunya", "ekonomi", "kultur", "saglik", "spor", "teknoloji"]
N_FAKE = 140
FAKE_ENV = {"transformers": "5.19.0", "torch": "2.14.1+cu126", "gpu": "Tesla T4"}


def _fake_world(d):
    """Sahte split (gruplu) + char_lr referansı; fold'lar make_plan'dan (gerçekteki gibi)."""
    row_id = np.arange(100, 100 + N_FAKE)
    y = np.arange(N_FAKE) % 7
    comps = [[100 + i, 100 + i + 7] for i in range(0, 70, 14)]  # aynı sınıftan ikili gruplar
    split = {"revision": "fake", "train_ids": row_id.tolist(), "test_ids": [1, 2, 3],
             "components": comps, "label_names": NAMES}  # fmt: skip
    split_path = d / "split_fake.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    train = pd.DataFrame({"row_id": row_id, "label": y})
    plans = finetune.make_plan(train, split_path)
    fold0 = finetune.fold_assignment(plans, N_FAKE)
    rng = np.random.default_rng(0)
    folds = [
        {"repeat": r, "fold": k, "n_train": 112, "n_val": 28, "macro_f1": 0.80 + 0.01 * k}
        for r in range(3)
        for k in range(5)
    ]
    preds_file = "preds/cv_char_lr_20260101-000000.npz"
    (d / "preds").mkdir(parents=True, exist_ok=True)
    np.savez(d / preds_file, row_id=row_id, y_true=y, y_pred=np.stack([y] * 3),
             fold_id=np.stack([fold0, rng.permutation(fold0), rng.permutation(fold0)]))  # fmt: skip
    m = {"split": "split_v2", "git_commit": "abc1234", "label_names": NAMES,
         "preds_file": preds_file, "cv": {"folds": folds}, "test": None}  # fmt: skip
    (d / "phase2_char_lr_20260101-000000.json").write_text(json.dumps(m), encoding="utf-8")
    return {"row_id": row_id, "y": y, "plans": plans, "split_path": split_path}


def _fake_run(d, w, k, seed, i, collapsed=False, reseed_of=None, err=0.1, best_epoch=None):
    row_id, y, p = w["row_id"], w["y"], w["plans"][k]
    rng = np.random.default_rng(1000 * k + seed)
    rows = p.outer_val
    yt = y[rows]
    yp = np.where(rng.random(len(rows)) < err, (yt + 1) % 7, yt)
    when = datetime(2026, 1, 1, 1, 0, i)
    name = f"phase4_{finetune.NAME}_f{k}_s{seed}"
    preds_file = f"preds/cv_{name}_{when:%Y%m%d-%H%M%S}.npz"
    logits = rng.normal(size=(len(rows), 7)).astype(np.float32)
    logits[np.arange(len(rows)), yp] += 10  # argmax = y_pred
    np.savez(d / preds_file, row_id=row_id[rows], y_true=yt, y_pred=yp, logits=logits,
             fold_id=np.full(len(rows), k))  # fmt: skip
    best = 0.3 if collapsed else 0.9
    be = best_epoch if best_epoch is not None else 2 + (k + seed) % 4  # 2..5
    obj = {
        "split": "split_v2", "git_commit": "def5678", "label_names": NAMES, "smoke": False,
        "revision": finetune.FT_CONFIG["revision"], "ft_config": finetune.FT_CONFIG,
        "env": FAKE_ENV, "fold": k, "seed": seed, "reseed_of": reseed_of,
        "collapsed": collapsed, "best_epoch": be, "best_val_macro_f1": best,
        "trainer_best_metric": best, "val_macro_f1_reloaded": best, "epochs_run": min(be + 2, 5),
        "stopped_early": be + 2 < 5, "n_fit": len(p.fit), "n_val": len(p.val),
        "n_oof": len(rows), "val_row_ids": sorted(int(r) for r in row_id[p.val]),
        "fit_row_ids_sha256": row_ids_hash(np.sort(row_id[p.fit])),
        "schedule": {"num_train_epochs": 5, "max_steps": 35, "warmup_steps": 4},
        "oof": finetune.oof_metrics(yt, yp, NAMES), "preds_file": preds_file, "test": None,
    }  # fmt: skip
    path = report.write_metrics(obj, name, d, when=when)
    return report.to_jsonable(obj) | {"_file": path.name}


def _fake_runs(d, collapse=((2, 43), (3, 42))):
    w = _fake_world(d)
    runs, i = {}, 0
    for k in range(5):
        for s in (42, 43, 44):
            runs[(k, s)] = _fake_run(d, w, k, s, i, collapsed=(k, s) in collapse)
            i += 1
    runs[(2, 143)] = _fake_run(d, w, 2, 143, i, reseed_of=43, err=0.3)
    runs[(3, 142)] = _fake_run(d, w, 3, 142, i + 1, reseed_of=42, collapsed=True)
    return w, runs


def _used(runs):
    return {
        (k, s): runs[(k, s + 100)] if runs[(k, s)]["collapsed"] else runs[(k, s)]
        for k in range(5)
        for s in (42, 43, 44)
    }


def _edit(d, pattern, **kw):
    f = next(d.glob(pattern))
    rec = json.loads(f.read_text(encoding="utf-8")) | kw
    f.write_text(json.dumps(rec), encoding="utf-8")


def test_summarize_with_fake_runs(tmp_path):
    w, runs = _fake_runs(tmp_path)
    used = _used(runs)
    p_sum, p_oof = finetune.summarize(
        tmp_path, tmp_path, when=datetime(2026, 1, 2), split_path=w["split_path"]
    )
    s = json.loads(p_sum.read_text(encoding="utf-8"))
    assert s["split"] == "split_v2" and s["test"] is None
    assert s["env"] == FAKE_ENV
    # çöküş raporu: orijinal ve yeniden koşu ayrıntıları
    cs = {(c["fold"], c["seed"]): c for c in s["collapsed_slots"]}
    assert set(cs) == {(2, 43), (3, 42)}
    assert cs[(2, 43)]["reseed"] == 143 and not cs[(2, 43)]["reseed_also_collapsed"]
    assert cs[(3, 42)]["reseed_also_collapsed"] is True
    assert cs[(2, 43)]["original"]["file"] == runs[(2, 43)]["_file"]
    assert cs[(2, 43)]["reseed_run"]["oof_macro_f1"] == runs[(2, 143)]["oof"]["macro_f1"]
    assert cs[(2, 43)]["original"]["best_val_macro_f1"] == 0.3
    used_seeds = {(r["fold"], r["slot_seed"]): r["seed"] for r in s["runs"]}
    assert used_seeds[(2, 43)] == 143 and used_seeds[(3, 42)] == 142 and len(used_seeds) == 15
    # skorlar
    f1 = np.array([[used[(k, sd)]["oof"]["macro_f1"] for sd in (42, 43, 44)] for k in range(5)])
    assert np.allclose(s["cv"]["macro_f1_folds"], f1.mean(axis=1))
    assert np.isclose(s["cv"]["macro_f1_fold_std"], f1.mean(axis=1).std(ddof=1))
    assert np.allclose(s["cv"]["macro_f1_seed_std_by_fold"], f1.std(axis=1, ddof=1))
    assert np.isclose(s["cv"]["macro_f1_by_seed_std"], f1.mean(axis=0).std(ddof=1))
    ratio = 28 / 112
    fm = f1.mean(axis=1)
    ratio_fit = np.mean([used[key]["n_oof"] / used[key]["n_fit"] for key in sorted(used)])
    assert s["cv"]["ratio_char_lr"] == pytest.approx(ratio)
    assert s["cv"]["macro_f1_nb_ci95_ratio_char_lr"] == pytest.approx(nb_mean_ci(fm, ratio))
    assert s["cv"]["ratio_bert_fit"] == pytest.approx(ratio_fit)
    assert s["cv"]["macro_f1_nb_ci95_ratio_bert_fit"] == pytest.approx(nb_mean_ci(fm, ratio_fit))
    # havuzlanmış seed std: sqrt(ortalama s²), df = 5·2 = 10, σ aralığı χ²'den
    sp = np.sqrt(np.mean(f1.std(axis=1, ddof=1) ** 2))
    assert s["cv"]["macro_f1_seed_std_pooled"] == pytest.approx(sp)
    assert s["cv"]["macro_f1_seed_std_pooled_df"] == 10
    lo10 = np.sqrt(10 / stats.chi2.ppf(0.975, 10))
    hi10 = np.sqrt(10 / stats.chi2.ppf(0.025, 10))
    assert s["cv"]["macro_f1_seed_std_pooled_sigma_ci95"] == pytest.approx([sp * lo10, sp * hi10])
    assert "0.886" in s["notes"]["macro_f1_seed_std_mean"]
    assert s["final_rules"] == finetune.FINAL_RULES
    assert s["superseded_files"] == [] and s["failed_attempts"] == []
    ref_f1 = 0.80 + 0.01 * np.arange(5)
    vs = s["vs_reference"]
    assert np.allclose(vs["d_folds"], f1.mean(axis=1) - ref_f1)
    assert vs["reference_macro_f1_mean_repeat0"] == pytest.approx(ref_f1.mean())
    assert vs["val_train_ratio"] == pytest.approx(ratio)
    assert vs["nadeau_bengio"]["J"] == 5 and vs["nadeau_bengio"]["df"] == 4
    assert s["notes"]["cv_comparison"] == finetune.RULES["cv_comparison"]
    assert s["notes"]["input_text"] == finetune.RULES["input_text"]
    lo, hi = s["notes"]["seed_std_sigma_ci95_factors"]
    assert lo == pytest.approx(0.52, abs=0.005) and hi == pytest.approx(6.29, abs=0.01)
    # best epoch dağılımı, sansür, erken durma
    best = [used[(k, sd)]["best_epoch"] for k in range(5) for sd in (42, 43, 44)]
    assert s["e_final"] == int(np.median(best))
    assert s["best_epoch_distribution"] == {str(e): best.count(e) for e in range(1, 6)}
    assert s["n_best_at_max_epoch"] == best.count(5) > 0
    assert "frac_stopped_early" not in s
    assert s["frac_epochs_run_lt_max"] == pytest.approx(np.mean([b + 2 < 5 for b in best]))
    assert "≤ 7" in s["notes"]["best_epoch_censoring"]
    assert s["finals"]["slots"] == {} and s["finals"]["missing_slots"] == [42, 43, 44]

    # seed 42 OOF: paired_test --power'ın beklediği ad ve biçim; 142 kullanılan fold işaretli
    assert latest_metrics(finetune.NAME, tmp_path, phase=4)["_file"] == p_oof.name
    A = paired_test.load_oof(finetune.NAME, tmp_path)
    B = paired_test.load_oof("char_lr", tmp_path)
    assert A["y_pred"].shape == (1, N_FAKE) and A["fold_id"].shape == (1, N_FAKE)
    paired_test._check_pair(A, B, 1)  # aynı satırlar, etiketler ve repeat 0 fold'ları
    o = json.loads(p_oof.read_text(encoding="utf-8"))
    assert o["seed"] == test_plan.PRIMARY_SEED and o["test"] is None
    assert o["reseeded_folds"] == [3]
    assert np.allclose(o["cv"]["macro_f1_folds"], f1[:, 0])


@pytest.mark.parametrize(
    "pattern, change, match",
    [
        ("f1_s43", {"ft_config": {"learning_rate": 5e-5}}, "ft_config"),
        ("f1_s43", {"revision": "0" * 40}, "revision"),
        ("f1_s43", {"env": FAKE_ENV | {"transformers": "4.46.0"}}, "ortam"),
        ("f1_s43", {"env": FAKE_ENV | {"gpu": "NVIDIA L4"}}, "ortam"),
        # CV simetrisi (RULES['collapse']): orijinali çökmemiş yeniden koşu, yanlış reseed_of,
        # FT_CONFIG seed'li koşuda reseed_of, json fold'u adla uyuşmuyor
        ("f2_s43", {"collapsed": False}, "çökmemiş"),
        ("f2_s143", {"reseed_of": 44}, "slotla uyuşmuyor"),
        ("f0_s42", {"reseed_of": 42}, "reseed_of"),
        ("f0_s42", {"fold": 1}, "adla uyuşmuyor"),
        # fit/val denetimi kullanılmayan (çöken) orijinallere de uygulanır
        ("f2_s43", {"val_row_ids": [100]}, "val_row_ids"),
        ("f3_s42", {"fit_row_ids_sha256": "0" * 64}, "fit"),
        ("f0_s44", {"trainer_best_metric": 0.8}, "best_metric"),
        ("f0_s44", {"val_macro_f1_reloaded": 0.7}, "yüklenen"),
        ("f4_s42", {"val_row_ids": [100]}, "val_row_ids"),
        ("f4_s42", {"fit_row_ids_sha256": "0" * 64}, "fit"),
        ("f2_s143", {"ft_config": {}}, "ft_config"),  # kullanılan yeniden koşu da denetlenir
    ],
)
def test_summarize_rejects_inconsistent_runs(tmp_path, pattern, change, match):
    w, _ = _fake_runs(tmp_path)
    _edit(tmp_path, f"phase4_{finetune.NAME}_{pattern}_*.json", **change)
    with pytest.raises(ValueError, match=match):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])


def test_summarize_rejects_val_oof_overlap_and_missing(tmp_path):
    w, runs = _fake_runs(tmp_path)
    # val ∩ OOF ≠ ∅ (aynı zamanda planla uyuşmaz): her iki denetimden biri yakalar
    oof_ids = sorted(int(r) for r in w["row_id"][w["plans"][1].outer_val])
    _edit(tmp_path, f"phase4_{finetune.NAME}_f1_s42_*.json", val_row_ids=oof_ids[:3])
    with pytest.raises(ValueError, match="val_row_ids|kesişiyor"):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])
    next(tmp_path.glob(f"phase4_{finetune.NAME}_f4_s44_*.json")).unlink()
    with pytest.raises(SystemExit, match="Eksik"):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])


COLAB_MD = "/content/drive/MyDrive/ttc4900/faz4/models"  # başka işletim sisteminde yazılmış yol


def _fake_final(d, seed, e_fin, used, f1, reseed_of=None, i=0, model_dir=COLAB_MD):
    """train_final'ın yazdığı json'un sahtesi; FINAL_RULES alanları gerçek final_meta'dan."""
    md = str(model_dir).rstrip("/\\")
    sep = "/" if "/" in md else "\\"
    obj = {
        "split": "split_v2", "smoke": False, "revision": finetune.FT_CONFIG["revision"],
        "ft_config": finetune.FT_CONFIG, "seed": seed, "e_final": e_fin, "epochs_run": e_fin,
        "e_final_sources": finetune.e_final_sources(used), "train_loss_last_epoch": 0.1,
        "train_in_sample_macro_f1": f1, "train_in_sample_accuracy": f1,
        "model_dir": f"{md}{sep}{seed}", "runtime_seconds": 1.0, "test": None,
        "final_collapse_rule": finetune.FINAL_RULES["final_collapse"], "env": FAKE_ENV,
        "weights_sha256": {"model.safetensors": f"{seed:064d}"},
        **finetune.final_meta(seed, reseed_of, f1, model_dir),
    }  # fmt: skip
    if "/" in md:  # final_meta yerel Path ile birleştirir; Colab'da yazılmış gibi posix yol
        if obj["primary_model_dir"] is not None:
            obj["primary_model_dir"] = f"{md}/{Path(obj['primary_model_dir']).name}"
    path = report.write_metrics(obj, f"phase4_{finetune.NAME}_final_s{seed}", d,
                                when=datetime(2026, 1, 1, 3, 0, i))  # fmt: skip
    return report.to_jsonable(obj) | {"_file": path.name}


def test_summarize_cross_checks_finals(tmp_path):
    w, runs = _fake_runs(tmp_path)
    used = _used(runs)
    e_fin = finetune.e_final([r["best_epoch"] for r in used.values()])
    _fake_final(tmp_path, 42, e_fin, used, 0.04, i=0)  # slot 42 çöktü
    _fake_final(tmp_path, 142, e_fin, used, 0.97, reseed_of=42, i=1)
    _fake_final(tmp_path, 43, e_fin, used, 0.99, i=2)
    _fake_final(tmp_path, 44, e_fin, used, 0.98, i=3)
    p_sum, _ = finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])
    fin = json.loads(p_sum.read_text(encoding="utf-8"))["finals"]
    assert fin["slots"]["42"]["used_seed"] == 142 and fin["slots"]["42"]["collapsed"]
    assert fin["slots"]["42"]["reseed_also_collapsed"] is False
    assert fin["slots"]["42"]["original"]["train_in_sample_macro_f1"] == 0.04
    assert fin["slots"]["43"]["used_seed"] == 43 and fin["slots"]["43"]["reseed"] is None
    assert fin["primary_seed"] == 142 and fin["primary_model_dir"] == f"{COLAB_MD}/142"
    assert fin["collapsed_slots"] == [42] and fin["missing_slots"] == []
    assert fin["rule"] == finetune.FINAL_RULES["final_collapse"]
    assert fin["primary_model_sha256"] == {"model.safetensors": f"{142:064d}"}

    # tutarsızlıklar: e_final, e_final_sources, primary bayrağı, eksik yeniden eğitim
    pat42 = f"phase4_{finetune.NAME}_final_s42_*.json"
    _edit(tmp_path, pat42, e_final=e_fin + 1)
    with pytest.raises(ValueError, match="e_final"):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])
    _edit(tmp_path, pat42, e_final=e_fin, e_final_sources=finetune.e_final_sources(used)[1:])
    with pytest.raises(ValueError, match="e_final_sources"):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])
    _edit(tmp_path, pat42, e_final_sources=finetune.e_final_sources(used), primary_for_test=True)
    with pytest.raises(ValueError, match="FINAL_RULES|primary"):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])
    _edit(tmp_path, pat42, primary_for_test=False)
    finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])  # yeniden tutarlı
    next(tmp_path.glob(f"phase4_{finetune.NAME}_final_s142_*.json")).unlink()
    with pytest.raises(ValueError, match="yeniden eğitimi eksik"):
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])


def test_final_rules_text_and_frozen_sources(monkeypatch):
    c, txt = finetune.FT_CONFIG, finetune.FINAL_RULES["final_collapse"]
    off, thr, p = c["collapse_reseed_offset"], c["collapse_threshold"], test_plan.PRIMARY_SEED
    assert f"< {thr}" in txt and f"seed + {off}" in txt and f"seed {p + off} modelidir" in txt
    assert "FINAL_RULES['final_collapse']" in test_plan.TEST_PLAN["primary_seed_rule"]
    assert f"seed {p}+100" in test_plan.TEST_PLAN["primary_seed_rule"]  # belge metniyle aynı
    # eşik ve offset FT_CONFIG'ten okunur (yeni sayı yok): değiştirilince davranış izler
    monkeypatch.setitem(c, "collapse_threshold", 0.95)
    monkeypatch.setitem(c, "collapse_reseed_offset", 7)
    m = finetune.final_meta(42, None, 0.9, Path("m"))
    assert m["collapsed"] and m["reseed_seed"] == 49 and m["primary_model_dir"].endswith("49")
    finals = {42: {"collapsed": True}, 49: {"collapsed": False, "id": "r"}}
    assert finetune.select_finals(finals, [42])[0][42]["id"] == "r"


def test_final_meta_cases():
    md = Path("models")
    m = finetune.final_meta(42, None, 0.9, md)  # çökmedi
    assert m["primary_for_test"] and m["used_for_slot"] and m["reseed_seed"] is None
    assert m["primary_model_dir"] == str(md / "42") and m["reseed_also_collapsed"] is None
    m = finetune.final_meta(42, None, 0.04, md)  # çöktü: kullanılan 142 olacak
    assert m["collapsed"] and not m["primary_for_test"] and not m["used_for_slot"]
    assert m["reseed_seed"] == 142 and m["primary_model_dir"] == str(md / "142")
    m = finetune.final_meta(142, 42, 0.95, md)  # yeniden eğitim
    assert m["slot_seed"] == 42 and m["primary_for_test"] and m["reseed_also_collapsed"] is False
    m = finetune.final_meta(142, 42, 0.03, md)  # o da çöktü: yine o kullanılır, işaretlenir
    assert m["collapsed"] and m["primary_for_test"] and m["reseed_also_collapsed"] is True
    m = finetune.final_meta(143, 43, 0.03, md)  # birincil olmayan slot
    assert not m["primary_for_test"] and m["primary_model_dir"] is None
    with pytest.raises(ValueError):
        finetune.final_meta(150, 42, 0.9, md)  # yeniden eğitim seed'i slot + offset değil


def test_cmd_final_reseed_and_restart(tmp_path, monkeypatch):
    """cmd_final: çöken slot için yalnız bir kez seed+100 eğitilir; tekrar başlatmada yalnız
    eksik olan (yeniden eğitim) koşulur. Eğitim monkeypatch'li (GPU/ağ yok)."""
    _, runs = _fake_runs(tmp_path)
    md = tmp_path / "models"
    ctx = {"out_dir": tmp_path, "smoke": False, "model_dir": md}
    in_sample = {42: 0.04, 142: 0.03, 43: 0.99, 44: 0.98}  # 42 ve yeniden eğitimi çöker
    calls = []

    def fake_train_final(ctx_, seed, e_fin, used, reseed_of=None):
        calls.append((seed, reseed_of))
        return _fake_final(tmp_path, seed, e_fin, used, in_sample[seed], reseed_of,
                           i=len(calls), model_dir=md)  # fmt: skip

    monkeypatch.setattr(finetune, "_context", lambda args: ctx)
    monkeypatch.setattr(finetune, "train_final", fake_train_final)
    args = finetune._parser().parse_args(["final"])
    finetune.cmd_final(args)
    assert calls == [(42, None), (142, 42), (43, None), (44, None)]
    f = finetune.find_finals(tmp_path)
    assert {s: r["primary_for_test"] for s, r in f.items()} == {
        42: False, 142: True, 43: False, 44: False
    }  # fmt: skip
    assert f[142]["reseed_also_collapsed"] is True and f[42]["reseed_seed"] == 142
    assert f[42]["primary_model_dir"] == f[142]["primary_model_dir"] == str(md / "142")

    calls.clear()
    finetune.cmd_final(args)  # hepsi var: hiçbir eğitim yok
    assert calls == []
    next(tmp_path.glob(f"phase4_{finetune.NAME}_final_s142_*.json")).unlink()
    finetune.cmd_final(args)  # orijinal çökmüş, yeniden eğitim yok: yalnız o koşulur
    assert calls == [(142, 42)]
    with pytest.raises(SystemExit):
        finetune.main(["final", "--seeds", "142"])  # 142 yalnız kural yoluyla
    # var olan json başka bir model dizinine göre yazılmışsa atlanmaz: hata
    monkeypatch.setitem(ctx, "model_dir", tmp_path / "baska")
    with pytest.raises(ValueError, match="FINAL_RULES|primary"):
        finetune.cmd_final(args)


def test_check_final_on_skip():
    used = {(0, 42): {"seed": 42, "_file": "a.json", "best_epoch": 3}}
    md = Path("models")
    rec = _rec(
        smoke=False, e_final=3, e_final_sources=finetune.e_final_sources(used), seed=43,
        train_in_sample_macro_f1=0.9, model_dir=str(md / "43"),
        **finetune.final_meta(43, None, 0.9, md),
    )  # fmt: skip
    finetune.check_final(rec, 3, used, smoke=False)
    finetune.check_final(rec, 3, used, smoke=False, model_dir=md)
    with pytest.raises(ValueError, match="e_final"):
        finetune.check_final(rec, 2, used, smoke=False)
    with pytest.raises(ValueError, match="smoke"):
        finetune.check_final(rec, 3, used, smoke=True)
    with pytest.raises(ValueError, match="ft_config"):
        finetune.check_final(rec | {"ft_config": {}}, 3, used, smoke=False)
    with pytest.raises(ValueError, match="FINAL_RULES"):  # çökmüş ama çökmedi diye kaydedilmiş
        finetune.check_final(rec | {"train_in_sample_macro_f1": 0.1}, 3, used, smoke=False)


# --------------------------------------------------------------------------- CLI ve HF (ağsız)
def test_gpu_gate_and_smoke_dirs(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="GPU"):
        finetune.require_device(smoke=False)
    assert finetune.require_device(smoke=True) == "cpu"
    args = finetune._parser().parse_args(["cv", "--smoke", "--out-dir", str(METRICS_DIR)])
    with pytest.raises(SystemExit):
        finetune.resolve_dirs(args)  # smoke gerçek outputs/metrics'e yazmaz
    out, ckpt, _ = finetune.resolve_dirs(finetune._parser().parse_args(["cv", "--smoke"]))
    assert METRICS_DIR.resolve() not in out.resolve().parents and out != METRICS_DIR


def _tiny(tmp_path, n=16):
    """Ağsız küçük tokenizer + model + veri (yalnız Trainer mekaniğini sınamak için)."""
    from datasets import Dataset
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(50)]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"))
    rng = np.random.default_rng(0)
    texts = [" ".join(f"w{j}" for j in rng.integers(0, 50, rng.integers(3, 12))) for _ in range(n)]
    ds = Dataset.from_dict({"text": texts, "label": rng.integers(0, 7, n).tolist()})
    ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True, remove_columns=["text"])

    def model():
        import torch

        torch.manual_seed(0)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32, num_labels=7,
                         hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)  # fmt: skip
        return BertForSequenceClassification(cfg)

    return tok, ds, model


def test_training_args_follow_ft_config(tmp_path):
    c = finetune.FT_CONFIG
    a = finetune.training_args(tmp_path / "o", 43, smoke=True)
    assert a.seed == 43 and a.learning_rate == c["learning_rate"]
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == 16
    assert a.get_warmup_steps(100) == 10  # warmup oranı 0.1 (4.x warmup_ratio / 5.x warmup_steps)
    assert a.lr_scheduler_type == "linear" and a.optim == "adamw_torch"
    assert a.weight_decay == c["weight_decay"] and a.max_grad_norm == c["max_grad_norm"]
    assert a.eval_strategy == "epoch" and a.save_strategy == "epoch" and a.load_best_model_at_end
    assert a.metric_for_best_model == "macro_f1" and a.greater_is_better
    assert not a.fp16 and a.num_train_epochs == finetune.SMOKE_EPOCHS  # smoke: CPU, fp16 yok
    f = finetune.training_args(tmp_path / "f", 42, smoke=True, final=True)
    assert f.eval_strategy == "no" and f.save_strategy == "no" and not f.load_best_model_at_end


def test_gradient_accumulation_equals_batch16_mean(tmp_path):
    """8 x 2 birikimin gradyanı = 16 örneklik batch'in ortalama kaybının gradyanı.

    transformers 5.x varsayılanında BertForSequenceClassification için gradyan 2 katı çıkıyordu
    (kayıp birikim adımına bölünmüyordu); make_trainer bunu düzeltir.
    """
    import torch
    from transformers import DataCollatorWithPadding, TrainerCallback

    tok, ds, model = _tiny(tmp_path)
    grads = []

    class Grab(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if not grads:
                grads.append(torch.cat([p.grad.flatten().clone() for p in model.parameters()
                                        if p.grad is not None]))  # fmt: skip

    args = finetune.training_args(tmp_path / "o", 0, smoke=True, final=True)
    args.max_grad_norm = 1e9  # kırpmasız gradyan
    trainer = finetune.make_trainer(model(), args, ds, None, tok, [Grab()])
    assert trainer.model_accepts_loss_kwargs is False
    trainer.train()
    m = model()
    batch = DataCollatorWithPadding(tok)([ds[i] for i in range(len(ds))])
    m(**batch).loss.backward()
    ref = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    assert torch.allclose(grads[0], ref, atol=1e-6, rtol=1e-4)


def test_stop_at_epoch_keeps_full_schedule(tmp_path):
    """Final: schedule num_train_epochs üzerinden kurulur, eğitim E_final'da durur."""
    tok, ds, model = _tiny(tmp_path, n=32)
    args = finetune.training_args(tmp_path / "o", 0, smoke=True, final=True)
    args.num_train_epochs = 3
    trainer = finetune.make_trainer(
        model(), args, ds, None, tok, [finetune.stop_at_epoch_callback(2)]
    )
    trainer.train()
    steps_per_epoch = 2  # 32 örnek / (8 x 2)
    assert trainer.state.max_steps == 3 * steps_per_epoch
    assert trainer.state.global_step == 2 * steps_per_epoch
    assert round(trainer.state.epoch) == 2
    assert trainer.lr_scheduler.get_last_lr()[0] > 0  # lr henüz sıfıra inmedi


def test_make_trainer_guards_loss_scaling(tmp_path, monkeypatch):
    """Sürüm koruması: Trainer kaybı birikime bölmeyecek durumdaysa (model_accepts_loss_kwargs
    True) make_trainer hata verir."""
    import transformers

    tok, ds, model = _tiny(tmp_path)
    args = finetune.training_args(tmp_path / "o", 0, smoke=True, final=True)
    assert (
        finetune.make_trainer(model(), args, ds, None, tok, []).model_accepts_loss_kwargs is False
    )

    class Patched(transformers.Trainer):
        def __init__(self, *a, processing_class=None, **kw):  # imza processing_class'ı korur
            super().__init__(*a, processing_class=processing_class, **kw)
            self.model_accepts_loss_kwargs = True  # gelecekteki bir sürümün davranışı

    monkeypatch.setattr(transformers, "Trainer", Patched)
    with pytest.raises(RuntimeError, match="model_accepts_loss_kwargs"):
        finetune.make_trainer(model(), args, ds, None, tok, [])


def test_tokenizer_revision_check(tmp_path, monkeypatch):
    """Ağsız: sahte HF önbelleği snapshots/<revision>/vocab.txt; sözlük birebir aynı olmalı."""
    import huggingface_hub
    from transformers import BertTokenizerFast

    rev = finetune.FT_CONFIG["revision"]
    snap = tmp_path / "snapshots" / rev
    snap.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "haber", "spor"]
    (snap / "vocab.txt").write_text("".join(t + "\n" for t in vocab), encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("".join(t + "\n" for t in [*vocab, "ekstra"]), encoding="utf-8")
    cached = {"path": str(snap / "vocab.txt")}
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: cached["path"])

    info = finetune.check_tokenizer_revision(BertTokenizerFast(str(snap / "vocab.txt")))
    assert info["commit"] == rev and info["vocab_size"] == len(vocab)
    with pytest.raises(ValueError, match="sözlüğü"):  # farklı sözlük
        finetune.check_tokenizer_revision(BertTokenizerFast(str(other)))
    cached["path"] = None  # bu revision önbellekte yok
    with pytest.raises(ValueError, match="bulunamadı"):
        finetune.check_tokenizer_revision(BertTokenizerFast(str(snap / "vocab.txt")))


# --------------------------------------------------------------------------- ikinci denetim
def test_summarize_env_includes_finals_and_lists_extras(tmp_path):
    w, runs = _fake_runs(tmp_path)
    used = _used(runs)
    e_fin = finetune.e_final([r["best_epoch"] for r in used.values()])
    for i, sd in enumerate((42, 43, 44)):
        _fake_final(tmp_path, sd, e_fin, used, 0.99, i=i)
    # alt dizindeki phase4 dosyaları listelenir (hata değil), başarısız denemeler özete girer
    sup = tmp_path / "superseded"
    sup.mkdir()
    (sup / f"phase4_{finetune.NAME}_f0_s42_20250101-000000.json").write_text("{}", "utf-8")
    finetune.record_failed_attempt(
        tmp_path, {"kind": "cv", "fold": 0, "seed": 42}, RuntimeError("OOM"), {}
    )
    p_sum, _ = finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])
    s = json.loads(p_sum.read_text(encoding="utf-8"))
    assert s["superseded_files"] == [
        f"superseded/phase4_{finetune.NAME}_f0_s42_20250101-000000.json"
    ]
    assert len(s["failed_attempts"]) == 1 and "OOM" in s["failed_attempts"][0]["error"]
    _edit(tmp_path, f"phase4_{finetune.NAME}_final_s43_*.json", env=FAKE_ENV | {"gpu": "A100"})
    with pytest.raises(ValueError, match="ortam"):  # final json'ları da ortam denetimine girer
        finetune.summarize(tmp_path, tmp_path, split_path=w["split_path"])


def test_check_env_and_phase4_records(tmp_path):
    env = {"transformers": "5.19.0", "torch": "2.14.1", "gpu": "Tesla T4", "python": "3.12"}
    (tmp_path / f"phase4_{finetune.NAME}_f0_s42_20260101-000000.json").write_text(
        json.dumps({"env": env | {"python": "3.11"}}), "utf-8"
    )
    (tmp_path / f"phase4_{finetune.NAME}_20260101-000000.json").write_text("{}", "utf-8")  # OOF
    recs = finetune.phase4_records(tmp_path)
    assert [r["_file"] for r in recs] == [f"phase4_{finetune.NAME}_f0_s42_20260101-000000.json"]
    finetune.check_env(recs, env)  # python sürümü ortam anahtarında değil
    with pytest.raises(ValueError, match="f0_s42.*gpu"):
        finetune.check_env(recs, env | {"gpu": "NVIDIA L4"})
    with pytest.raises(ValueError, match="torch"):
        finetune.check_env(recs, env | {"torch": "2.15.0"})


def test_failed_attempt_is_recorded_and_reraised(tmp_path, monkeypatch):
    class FakeTrainer:
        class state:  # noqa: N801 - Trainer.state taklidi
            log_history = [
                {"loss": 1.9, "epoch": 1.0},
                {"eval_macro_f1": 0.31, "eval_accuracy": 0.4, "eval_loss": 1.5, "epoch": 1.0},
            ]

    def boom(ctx, k, seed, reseed_of, state):
        state["trainer"] = FakeTrainer()
        raise ValueError("best checkpoint yüklenmemiş")

    monkeypatch.setattr(finetune, "_train_cv_run", boom)
    ctx = {"out_dir": tmp_path, "smoke": False}
    with pytest.raises(ValueError, match="best checkpoint"):
        finetune.train_cv_run(ctx, 2, 43)
    monkeypatch.setattr(finetune, "_train_final", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        finetune.train_final(ctx, 42, 3, {}, None)
    rows = finetune.read_failed_attempts(tmp_path)
    assert [(r["kind"], r["fold"], r["seed"]) for r in rows] == [("cv", 2, 43), ("final", None, 42)]
    assert rows[0]["val_curve"][0]["val_macro_f1"] == 0.31 and rows[1]["val_curve"] is None
    assert rows[0]["error"].startswith("ValueError") and rows[0]["git_commit"]
    assert not list(tmp_path.glob(f"phase4_{finetune.NAME}_f*_s*_*.json"))  # koşu sayılmaz


def test_weights_sha256_and_no_overwrite(tmp_path):
    import hashlib

    md = tmp_path / "models" / "42"
    finetune.check_model_dir_free(md)  # yok: serbest
    md.mkdir(parents=True)
    finetune.check_model_dir_free(md)  # boş: serbest
    (md / "model.safetensors").write_bytes(b"agirliklar")
    assert finetune.weights_sha256(md) == {
        "model.safetensors": hashlib.sha256(b"agirliklar").hexdigest()
    }
    with pytest.raises(FileExistsError):
        finetune.check_model_dir_free(md)
    with pytest.raises(FileNotFoundError):
        finetune.weights_sha256(tmp_path)
    args = finetune._parser().parse_args(
        ["final", "--smoke", "--model-dir", str(finetune.DEFAULT_MODEL_DIR)]
    )
    with pytest.raises(SystemExit):
        finetune.resolve_dirs(args)  # smoke gerçek final model dizinine yazmaz


def test_check_final_rule_text_and_stat_helpers():
    used = {(0, 42): {"seed": 42, "_file": "a.json", "best_epoch": 3}}
    md = Path("models")
    rec = _rec(
        smoke=False, e_final=3, e_final_sources=finetune.e_final_sources(used), seed=43,
        train_in_sample_macro_f1=0.9, model_dir=str(md / "43"),
        **finetune.final_meta(43, None, 0.9, md),
    )  # fmt: skip
    finetune.check_final(rec, 3, used, smoke=False)
    with pytest.raises(ValueError, match="final_collapse_rule"):
        finetune.check_final(rec | {"final_collapse_rule": "eski metin"}, 3, used, smoke=False)
    assert finetune.c4_factor(3) == pytest.approx(0.8862, abs=1e-4)
    assert finetune.sigma_ci_factors(2) == pytest.approx([0.5206, 6.2847], abs=1e-3)

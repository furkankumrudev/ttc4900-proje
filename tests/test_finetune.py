"""Faz 4a: dondurulmuş FT_CONFIG'in tutarlılığı (eğitim kodu yok, ağ/GPU gerekmez)."""

from ttc4900 import finetune, test_plan
from ttc4900.data import RANDOM_STATE
from ttc4900.embeddings import ENCODERS


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

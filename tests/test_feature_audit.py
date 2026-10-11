"""feature_audit kuralları: ağsız, elle kurulmuş örnekler."""

from collections import Counter

import numpy as np
import pandas as pd

from ttc4900 import feature_audit as fa


def test_word_flags():
    assert fa.word_flags("ekonomik") == {"category_name"}
    assert fa.word_flags("sporcu") == {"category_name"}
    assert fa.word_flags("dha") == {"source_signature"}
    assert fa.word_flags("enflasyon") == set()


def test_char_flags_prefix_and_examples():
    assert "category_name" in fa.char_flags(" spo", [])  # kelime başı, "spor" öneki
    assert "category_name" in fa.char_flags("kono", [])  # >=4 harf, "ekonomi" alt dizesi
    assert fa.char_flags("por", []) == set()  # kelime başı değil, 3 harf: "rapor" da içerir
    assert "source_signature" in fa.char_flags(" dh", ["dha"])


def test_example_words_uses_char_wb_padding():
    freq = Counter({"spor": 5, "sporcu": 3, "rapor": 9})
    assert fa.example_words(" spo", freq) == ["spor", "sporcu"]
    assert fa.example_words("por ", freq) == ["rapor", "spor"]


def test_audit_model_flags_template_and_concentration():
    """Sınıf 1'in her belgesi 'kaynak dha' ile bitiyor: template + source + concentrated."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(2):
        for i in range(30):
            body = " ".join(f"k{k}w{j}" for j in rng.integers(0, 8, 12))
            rows.append({"label": k, "text": body + (" kaynak dha" if k == 1 else f" son{i}")})
    train = pd.DataFrame(rows)
    pipe = fa.SPECS["word_lr"].make().set_params(clf__C=10.0, tfidf__ngram_range=(1, 1))
    pipe.fit(train["text"], train["label"])
    df = fa.audit_model("word_lr", pipe, train, ["a", "b"], top_k=20)
    dha = df[(df["class"] == "b") & (df["feature"] == "dha")].iloc[0]
    assert set(dha["flags"].split(";")) == {"concentrated", "source_signature", "template"}
    assert dha["df_in_class"] == 30 and dha["df_other"] == 0
    assert np.isclose(dha["boundary_share_in_class"], 1.0)

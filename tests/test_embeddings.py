"""embeddings: ağsız testler (sahte tensörler ve config'ten kurulan minik BERT)."""

import numpy as np
import pandas as pd
import pytest
import torch

from ttc4900 import embeddings as emb
from ttc4900.data import make_split_v2


def test_mean_pool_ignores_padding_and_normalizes():
    h = torch.tensor([[[1.0, 0.0], [3.0, 4.0], [100.0, 100.0]]])  # son token padding
    m = torch.tensor([[1, 1, 0]])
    e = emb.pool(h, m, "mean")
    ref = np.array([2.0, 2.0]) / np.linalg.norm([2.0, 2.0])  # (1+3)/2, (0+4)/2
    assert np.allclose(e.numpy()[0], ref)
    assert np.allclose(emb.pool(h, m, "cls").numpy()[0], [1.0, 0.0])
    with pytest.raises(ValueError):
        emb.pool(h, m, "max")


@pytest.fixture
def tiny(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = "a b c d e f g h".split()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=13, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=16, max_position_embeddings=32)  # fmt: skip
    return tok, BertModel(cfg).eval()


def test_embed_texts_independent_of_batch_and_order(tiny):
    """Mask'li pooling + attention mask: bir metnin vektörü batch'teki diğerlerinden bağımsız."""
    tok, model = tiny
    spec = emb.EncoderSpec("x", "y", "mean", "", 6, "test")
    texts = ["a b", "c d e f g h a b c", "e"]
    E, n_trunc = emb.embed_texts(texts, tok, model, spec, batch_size=3)
    alone = np.vstack([emb.embed_texts([t], tok, model, spec, batch_size=1)[0] for t in texts])
    assert np.allclose(E, alone, atol=1e-5)
    assert n_trunc == 1  # 9 kelime + [CLS][SEP] = 11 > 6
    assert np.allclose(np.linalg.norm(E, axis=1), 1)


def test_prefix_is_prepended(tiny):
    tok, _ = tiny
    spec = emb.EncoderSpec("x", "y", "mean", "a ", 512, "test")
    assert emb.token_lengths(tok, spec, ["b c"]).tolist() == [5]  # [CLS] a b c [SEP]


def test_assert_train_only_rejects_test_rows(tmp_path):
    clean = pd.DataFrame(
        {"row_id": range(40), "label": [i % 2 for i in range(40)], "category": "c", "text": "t"}
    )
    p = tmp_path / "s.json"
    s = make_split_v2(clean, np.arange(40), path=p, n_raw=40)
    emb.assert_train_only(clean[clean["row_id"].isin(s["train_ids"])], p)
    with pytest.raises(ValueError, match="train'inde değil"):
        emb.assert_train_only(clean, p)


def test_encoder_specs_pinned():
    for name, spec in emb.ENCODERS.items():
        assert len(spec.revision) == 40 and int(spec.revision, 16) >= 0, name
        assert spec.pooling in {"mean", "cls"} and spec.max_length == 512, name


@pytest.fixture
def real_train_ids():
    from ttc4900.data import load_split

    ids = load_split()["train_ids"][:25]
    return pd.DataFrame({"row_id": ids, "text": [f"metin {i}" for i in ids]})


def test_embed_train_resumes_and_validates(tmp_path, monkeypatch, real_train_ids):
    """Yarıda kesilen üretim biten parçaları yeniden hesaplamaz; sonuç bütün koşuyla aynı."""
    monkeypatch.setattr(emb, "CACHE_DIR", tmp_path)
    train = real_train_ids
    calls: list[int] = []

    def enc(texts):
        calls.append(len(texts))
        if len(calls) == 3:
            raise KeyboardInterrupt  # 3. parçada kesinti
        return np.array([[float(t.split()[1]), 1.0] for t in texts]), 0

    with pytest.raises(KeyboardInterrupt):
        emb.embed_train("e5_base", train, chunk=10, encode=enc)
    assert not emb.cache_path("e5_base").exists()
    calls.clear()
    good = lambda texts: (np.array([[float(t.split()[1]), 1.0] for t in texts]), 0)  # noqa: E731
    seen: list[int] = []
    emb.embed_train("e5_base", train, chunk=10, encode=lambda t: (seen.append(len(t)), good(t))[1])
    assert seen == [5]  # yalnızca kalan 3. parça (25 = 10 + 10 + 5)
    X = emb.load_embeddings("e5_base", train)
    assert np.array_equal(X[:, 0], train["row_id"].to_numpy())
    assert not emb.cache_path("e5_base").with_suffix(".parts").exists()
    # satır sırası farklı train -> hata (embedding'ler yanlış satırlara gitmesin)
    with pytest.raises(ValueError):
        emb.load_embeddings("e5_base", train.iloc[::-1].reset_index(drop=True))
    # c1clean ayrı dosya, metin temizlenerek
    got: list[str] = []
    emb.embed_train("e5_base", train.assign(text=train["text"] + " \u0092"), c1clean=True,
                    chunk=50, encode=lambda t: (got.extend(t), good(t))[1])  # fmt: skip
    assert all("\u0092" not in t for t in got)
    assert emb.cache_path("e5_base", c1clean=True).exists()

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

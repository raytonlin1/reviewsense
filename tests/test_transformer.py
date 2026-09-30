"""Part 2 tests: prove each brick does what the math says. No downloads, <2 s on CPU."""
import torch

from reviewsense.scratch.transformer import (DecoderOnlyLM, MultiHeadAttention, PositionalEncoding, Seq2SeqTransformer,
                                             causal_mask, padding_mask, scaled_dot_product_attention)

torch.manual_seed(0)


def test_attention_rows_are_distributions_and_causal():
    q = k = v = torch.randn(2, 3, 5, 8)                        # (batch, heads, seq, d_head)
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    assert out.shape == (2, 3, 5, 8)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5))     # softmax rows sum to 1
    assert torch.all(w.triu(1) == 0)                            # zero weight on the future


def test_multihead_shapes():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 7, 32)
    assert mha(x, x, x).shape == (2, 7, 32)
    assert mha.last_attn.shape == (2, 4, 7, 7)                 # one 7x7 attention map per head


def test_positional_encoding_is_bounded_and_position_specific():
    pe = PositionalEncoding(16, max_len=50).pe[0]
    assert pe.shape == (50, 16) and pe.abs().max() <= 1.0     # sin/cos stay in [-1, 1]
    assert not torch.allclose(pe[3], pe[4])                    # every position gets a different vector


def test_padding_does_not_change_real_tokens():
    m = Seq2SeqTransformer(20, 20, d_model=32, n_heads=4, n_layers=2).eval()
    short = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])                   # 0 = PAD
    a, _ = m.encode(short)
    b, _ = m.encode(padded)
    assert torch.allclose(a, b[:, :3], atol=1e-5)              # padding mask works
    assert padding_mask(padded, 0).tolist() == [[[[True, True, True, False, False]]]]


def test_gpt_cannot_see_the_future():
    lm = DecoderOnlyLM(30, d_model=32, n_heads=4, n_layers=2, max_len=16).eval()
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    y = x.clone(); y[0, 4:] = torch.tensor([9, 9])             # change only tokens 4 and 5
    la, _ = lm(x)
    lb, _ = lm(y)
    assert torch.allclose(la[0, :4], lb[0, :4], atol=1e-5)     # positions 0-3 unaffected
    assert not torch.allclose(la[0, 4:], lb[0, 4:])


def test_seq2seq_and_generation_shapes():
    m = Seq2SeqTransformer(20, 30, d_model=32, n_heads=4, n_layers=1)
    assert m(torch.randint(3, 20, (2, 7)), torch.randint(3, 30, (2, 5))).shape == (2, 5, 30)
    lm = DecoderOnlyLM(25, d_model=32, n_heads=4, n_layers=1, max_len=16)
    logits, loss = lm(torch.randint(0, 25, (2, 10)), torch.randint(0, 25, (2, 10)))
    assert logits.shape == (2, 10, 25) and loss.item() > 0
    assert lm.generate(torch.zeros(1, 1, dtype=torch.long), 5, top_k=5, top_p=0.9).shape == (1, 6)


def test_untrained_gpt_loss_is_near_uniform():
    """Sanity check every LM should pass: before training, loss ~ ln(vocab) (uniform guessing).
    A much larger value means bad initialisation (this caught a real bug: loss 84 with vocab ~100)."""
    import math

    V = 100
    lm = DecoderOnlyLM(V, d_model=128, n_heads=4, n_layers=2, max_len=32).eval()
    _, loss = lm(torch.randint(0, V, (4, 32)), torch.randint(0, V, (4, 32)))
    assert abs(loss.item() - math.log(V)) < 0.5
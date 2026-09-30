"""Chapter 9: a Transformer built from LEGO bricks, in plain PyTorch.

Bricks (bottom-up):
  scaled_dot_product_attention -> MultiHeadAttention -> (+ residual, LayerNorm, FeedForward)
  -> EncoderLayer / DecoderLayer -> Encoder / Decoder -> Seq2SeqTransformer (translation)
  and DecoderOnlyLM (GPT-style generative LLM, ch 10.2) re-uses DecoderBlock with a causal mask.

Why attention instead of recurrence (RNN/LSTM)?
  * RNN: token t depends on hidden state t-1 -> O(n) sequential steps, vanishing gradients over long spans.
  * Attention: every token looks at every other token in ONE matrix multiply -> parallel on GPU,
    path length between any two tokens is 1.  Cost: O(n^2) memory in sequence length.
  * Attention has no notion of order -> we must add positional encodings.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (batch, heads, seq, d_head). mask: broadcastable bool, True = keep.

    scores = Q K^T / sqrt(d_k): dividing by sqrt(d_k) keeps the dot products' variance ~1 so
    softmax doesn't saturate (tiny gradients) when d_k is large.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """h heads each attend in a d_model/h subspace -> different heads learn different relations
    (syntax, coreference, position...). Outputs are concatenated and mixed by W_o."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_head = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None  # kept for visualisation

    def _split(self, x):  # (B, T, D) -> (B, H, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_head).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q, k, v = self._split(self.w_q(query)), self._split(self.w_k(key)), self._split(self.w_v(value))
        out, self.last_attn = scaled_dot_product_attention(q, k, v, mask)
        B, H, T, dh = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.drop(self.w_o(out))


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding from 'Attention Is All You Need':
        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    Each dimension is a sine wave of a different wavelength, so relative offsets are linear
    functions of the encodings. (BERT instead *learns* a position embedding; modern LLMs use RoPE.)"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))


class EncoderLayer(nn.Module):
    """Pre-LayerNorm variant (more stable than the original post-LN): x + Sublayer(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, src_mask)            # self-attention: bidirectional (like BERT)
        return x + self.ff(self.ln2(x))


class DecoderLayer(nn.Module):
    """Masked self-attention -> cross-attention over encoder memory -> feed-forward."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, cross_attention=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout) if cross_attention else None
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))

    def forward(self, x, tgt_mask, memory=None, mem_mask=None):
        h = self.ln1(x)
        x = x + self.self_attn(h, h, h, tgt_mask)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.ln2(x), memory, memory, mem_mask)   # Q from decoder, K/V from encoder
        return x + self.ff(self.ln3(x))


def padding_mask(ids, pad_id):          # (B, T) -> (B, 1, 1, T)
    return (ids != pad_id)[:, None, None, :]


def causal_mask(T, device=None):        # (1, 1, T, T) lower-triangular: token t sees only <= t
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))[None, None]


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder for translation (ch 9.2 'Transformer translation')."""

    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id, self.d_model = pad_id, d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln_enc, self.ln_dec = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src):
        m = padding_mask(src, self.pad_id)
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, m)
        return self.ln_enc(x), m

    def decode(self, tgt, memory, mem_mask):
        m = padding_mask(tgt, self.pad_id) & causal_mask(tgt.size(1), tgt.device)
        x = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, m, memory, mem_mask)
        return self.out(self.ln_dec(x))

    def forward(self, src, tgt_in):
        memory, mem_mask = self.encode(src)
        return self.decode(tgt_in, memory, mem_mask)

    @torch.no_grad()
    def greedy_decode(self, src, bos_id, eos_id, max_len=50):
        memory, mem_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            next_tok = self.decode(ys, memory, mem_mask)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return ys


class DecoderOnlyLM(nn.Module):
    """GPT-style generative LM (ch 10.2 'Creating your own generative LLM').
    Same DecoderLayer, no cross-attention, trained on next-token prediction."""

    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_len=256, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)                    # learned positions, like GPT-2
        self.blocks = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout, cross_attention=False) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight                           # weight tying
        self.max_len = max_len
        # GPT-2 init: N(0, 0.02). nn.Embedding defaults to N(0, 1); with weight tying the output logits would
        # then have std ~ sqrt(d_model) ~ 11 and the untrained model is confidently wrong: loss 84 instead of
        # ln(vocab) ~ 5. Small init = near-uniform predictions at step 0.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, ids, targets=None):
        T = ids.size(1)
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        mask = causal_mask(T, ids.device)
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        """Sampling knobs: temperature flattens/sharpens the distribution; top-k keeps the k best tokens;
        top-p (nucleus) keeps the smallest set whose probability mass >= p. Greedy = temperature -> 0."""
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.max_len:])
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p:
                sorted_logits, idx = logits.sort(descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                remove = cum - sorted_logits.softmax(-1) > top_p
                logits = logits.scatter(1, idx, sorted_logits.masked_fill(remove, float("-inf")))
            ids = torch.cat([ids, torch.multinomial(logits.softmax(-1), 1)], dim=1)
        return ids

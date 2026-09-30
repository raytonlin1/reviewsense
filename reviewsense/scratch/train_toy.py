"""Train the from-scratch models on CPU in minutes.

  python -m reviewsense.scratch.train_toy translate   # seq2seq: "march 5 2021" -> "2021-03-05"
  python -m reviewsense.scratch.train_toy gpt         # char-level GPT on review text
"""
from __future__ import annotations

import random
import sys

import torch
import torch.nn.functional as F

from .transformer import DecoderOnlyLM, Seq2SeqTransformer

MONTHS = "january february march april may june july august september october november december".split()
PAD, BOS, EOS = 0, 1, 2


class CharVocab:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.itos = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s, bos=False, eos=False):
        ids = [self.stoi[c] for c in s if c in self.stoi]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i > EOS)

    def __len__(self):
        return len(self.itos)


def random_date_pair():
    y, m, d = random.randint(1950, 2049), random.randint(1, 12), random.randint(1, 28)
    src = random.choice([f"{MONTHS[m-1]} {d} {y}", f"{d} {MONTHS[m-1]} {y}", f"{MONTHS[m-1][:3]} {d}, {y}"])
    return src, f"{y}-{m:02d}-{d:02d}"


def pad_batch(seqs):
    T = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (T - len(s)) for s in seqs])

def train_translation(steps=3000, batch=64, device="cpu"):
    random.seed(0)
    torch.manual_seed(0)
    vocab = CharVocab(MONTHS + ["0123456789 ,-"])
    model = Seq2SeqTransformer(len(vocab), len(vocab), d_model=128, n_heads=4, n_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    # Warmup + decay: large early updates destabilise attention; this is the schedule from the paper, simplified.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.1, 1 - s / steps))
    for step in range(steps):
        pairs = [random_date_pair() for _ in range(batch)]
        src = pad_batch([vocab.encode(s) for s, _ in pairs]).to(device)
        tgt = pad_batch([vocab.encode(t, bos=True, eos=True) for _, t in pairs]).to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]                       # teacher forcing: shift right
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                               ignore_index=PAD, label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}")
    model.eval()
    tests = [random_date_pair() for _ in range(200)]
    src = pad_batch([vocab.encode(s) for s, _ in tests]).to(device)
    out = model.greedy_decode(src, BOS, EOS, max_len=12)
    preds = [vocab.decode(o.tolist()[1:o.tolist().index(EOS)] if EOS in o.tolist() else o.tolist()[1:]) for o in out]
    acc = sum(p == t for p, (_, t) in zip(preds, tests)) / len(tests)
    for (s, t), p in list(zip(tests, preds))[:5]:
        print(f"{s:>22} -> {p}   (gold {t})")
    print(f"exact-match accuracy: {acc:.1%}")
    return model, acc

def train_gpt(text: str, steps=2000, block=128, batch=32, device="cpu"):
    torch.manual_seed(0)
    vocab = CharVocab([text])
    data = torch.tensor(vocab.encode(text))
    model = DecoderOnlyLM(len(vocab), d_model=128, n_heads=4, n_layers=4, max_len=block).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(steps):
        ix = torch.randint(len(data) - block - 1, (batch,))
        x = torch.stack([data[i:i + block] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + block + 1] for i in ix]).to(device)   # next-token targets
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}  perplexity {loss.exp().item():.1f}")
    model.eval()
    seed = torch.tensor([vocab.encode("The food was")], device=device)
    print(vocab.decode(model.generate(seed, 200, temperature=0.8, top_k=20)[0].tolist()))
    return model


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "translate"
    if task == "translate":
        train_translation()
    else:
        from ..data import load_hf_yelp

        corpus = "\n".join(load_hf_yelp("train", 5000)["text"])
        train_gpt(corpus)
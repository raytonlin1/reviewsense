import torch
from transformers import AutoTokenizer
from reviewsense.data import load_sample

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = [r.text for r in load_sample()[:2]]
enc = tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
print(enc["input_ids"].shape)          # torch.Size([2, 32])  -> (batch, seq)
print(tok.convert_ids_to_tokens(enc["input_ids"][0])[:12])
# ['[CLS]', 'the', 'hand', '-', 'pulled', 'noodles', 'at', 'golden', 'dragon', 'noodle', 'house', 'are']
print(enc["attention_mask"][1])        # 1 = real token, 0 = padding

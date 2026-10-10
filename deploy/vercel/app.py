"""ReviewSense star-rating API for Vercel: the Part 14 INT8 model with only onnxruntime + tokenizers (no PyTorch).

Gives the same stars as reviewsense/serving/onnx_model.py (tested in tests/integration/test_vercel_app.py), but small
enough for a Vercel function. Vercel finds this file (app.py) and its `app` variable automatically. See README.md.
"""
from pathlib import Path

import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

MODEL = Path(__file__).parent / "model"
tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
session = onnxruntime.InferenceSession(str(MODEL / "model-int8.onnx"), providers=["CPUExecutionProvider"])
STARS = np.arange(1, 6)

app = FastAPI(title="ReviewSense star rating")


class StarsRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32, description="review texts (at most 32)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/stars")
def stars(body: StarsRequest) -> dict:
    encodings = tokenizer.encode_batch(body.texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    logits = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))     # softmax (without scipy: smaller bundle)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {"stars": [round(float(value), 2) for value in probabilities @ STARS]}

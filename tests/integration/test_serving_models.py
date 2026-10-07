"""The served app with real models (needs artifacts/stars-onnx: python -m reviewsense.serving.onnx_model export).
Run: python -m pytest -m integration"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reviewsense.config import get_settings
from reviewsense.serving.onnx_model import compare

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    if not (Path(get_settings().star_onnx_dir) / "model-int8.onnx").exists():
        pytest.skip("export first: python -m reviewsense.serving.onnx_model export")
    from reviewsense.serving.api import app                       # loads every model, including the chatbot
    with TestClient(app) as test_client:
        yield test_client


def test_served_endpoints(client):
    good, bad = client.post("/stars", json={"texts": ["Best tacos in town!", "Cold pizza, rude manager."]}).json()["stars"]
    assert good - bad > 2
    assert client.post("/search", json={"query": "food poisoning", "k": 1}).json()["results"][0]["review_id"] == "r12"
    reply = client.post("/chat", json={"session_id": "it", "message": "Book a table at Golden Dragon for 2 tomorrow at 7 pm"})
    assert reply.json()["action"] == "confirm"
    assert client.get("/ui/").status_code == 200


def test_int8_model_is_as_accurate_as_pytorch():
    if not (Path(get_settings().star_onnx_dir) / "model-int8.onnx").exists():
        pytest.skip("export first")
    scores = compare(n=300)
    assert scores["onnx"]["same_as_pytorch"] == 1.0                      # export changes nothing
    assert scores["onnx_int8"]["accuracy"] >= scores["pytorch"]["accuracy"] - 0.02   # measured: +0.8 points (n=1000)

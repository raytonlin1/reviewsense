"""The Vercel star-rating app gives the same answers as the project's ONNX model. Needs the model copied into
deploy/vercel/model/ (see deploy/vercel/README.md). Run: python -m pytest -m integration"""
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration
APP = Path("deploy/vercel/app.py")


def test_vercel_app_matches_the_project_model():
    if not (APP.parent / "model" / "model-int8.onnx").exists():
        pytest.skip("copy the model first: see deploy/vercel/README.md")
    spec = importlib.util.spec_from_file_location("vercel_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)
    texts = ["Best tacos in town, friendly staff!", "Cold pizza, rude manager, never again."]
    from reviewsense.serving.onnx_model import OnnxStarModel
    assert client.post("/stars", json={"texts": texts}).json()["stars"] == OnnxStarModel().stars(texts)
    assert client.post("/stars", json={"texts": []}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}

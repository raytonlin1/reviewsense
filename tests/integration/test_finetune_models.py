"""The fine-tuning path end to end with the seconds-long smoke configs (CI / after library upgrades), and the quality
gate for a real fine-tuned model if one has been trained. Run: python -m pytest -m integration"""
import json
from pathlib import Path

import pytest

from reviewsense.finetune import dpo, sft
from reviewsense.finetune.evaluate import compare

pytestmark = pytest.mark.integration


def test_sft_then_dpo_produce_loadable_models():
    if not Path("artifacts/llm-data/sft.jsonl").exists():
        pytest.skip("build the data first: python -m reviewsense.finetune.build_data")
    sft.main(["--config", "configs/llm/smoke-sft.yaml"])
    dpo.main(["--config", "configs/llm/smoke-dpo.yaml"])
    for folder in ["artifacts/llm-smoke-sft", "artifacts/llm-smoke-dpo"]:
        record = json.loads((Path(folder) / "run.json").read_text())
        assert (Path(folder) / "model.safetensors").exists() and record["configs"]["model"]["use_peft"]


def test_fine_tuned_model_quality_does_not_regress():
    model = "artifacts/qwen3-0.6b-sft"                                   # the released model (DPO was not released)
    if not Path(model).exists():
        pytest.skip("train it first: python -m reviewsense.finetune.sft --config configs/llm/sft-qwen3-0.6b.yaml")
    scores = compare([model])[model]
    assert scores["accuracy"] >= 0.8 and scores["unsupported_sentences"] <= 0.2     # measured 0.86 and 0.11

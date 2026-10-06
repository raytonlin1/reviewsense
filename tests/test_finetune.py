"""Fast fine-tuning tests: configs, settings and data handling (no training, no model downloads)."""
import json
from dataclasses import fields
from pathlib import Path

import pytest
from trl import DPOConfig, ModelConfig, SFTConfig, TrlParser

from reviewsense.finetune.common import TaskArgs, is_refusal, load_split
from reviewsense.rag.assistant import DECLINE

CONFIGS = sorted(Path("configs/llm").glob("*.yaml"))


@pytest.mark.parametrize("config", CONFIGS, ids=[c.stem for c in CONFIGS])
def test_every_llm_config_parses_and_trains_on_cpu_safely(config):
    training = DPOConfig if "dpo" in config.stem else SFTConfig
    task, args, model = TrlParser((TaskArgs, training, ModelConfig)).parse_args_and_config(["--config", str(config)])
    assert model.use_peft and task.export_dir != args.output_dir          # checkpoints never mixed into the release
    assert not args.bf16 and not args.gradient_checkpointing               # TRL's GPU defaults switched off


def test_our_settings_never_shadow_library_settings():
    ours = {f.name for f in fields(TaskArgs)}
    for library in (SFTConfig, DPOConfig, ModelConfig):
        assert not ours & {f.name for f in fields(library)}, library.__name__


def test_refusals_are_capped(tmp_path):
    rows = [{"prompt": [{"role": "user", "content": f"q{i}"}],
             "completion": [{"role": "assistant", "content": DECLINE if i < 50 else f"Answer [1] {i}."}]}
            for i in range(100)]                                               # 50 refusals, 50 answers
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    split = load_split(TaskArgs(data_file=str(path), max_refusal_fraction=0.25, val_fraction=0.1))
    examples = list(split["train"]) + list(split["test"])
    assert len(examples) == 66 and sum(is_refusal(e) for e in examples) == 16       # 50 answers + 16 refusals

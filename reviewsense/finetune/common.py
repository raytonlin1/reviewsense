"""What the SFT and DPO scripts share: their data/export settings, the data split, and the release artifact."""
from __future__ import annotations

from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

from ..rag.assistant import DECLINE

from ..runinfo import write_run_record


@dataclass
class TaskArgs:
    data_file: str = "artifacts/llm-data/sft.jsonl"   # from python -m reviewsense.finetune.build_data
    export_dir: str = "artifacts/llm-model"           # the release artifact: the merged model, ready to serve
                                                      # (output_dir in the training config holds checkpoints only)
    val_fraction: float = 0.1
    split_seed: int = 42
    resume: bool = False                              # continue from the latest checkpoint after a crash
    max_refusal_fraction: float | None = None         # cap the share of examples whose target is a refusal


def is_refusal(example: dict) -> bool:
    """The target answer is "The reviews don't say." (SFT: the completion; DPO: the chosen answer)."""
    target = example["completion"] if "completion" in example else example["chosen"]
    return target[0]["content"] == DECLINE


def is_answer(example: dict) -> bool:
    return not is_refusal(example)


def load_split(task: TaskArgs):
    data = load_dataset("json", data_files=task.data_file, split="train")
    if task.max_refusal_fraction is not None:
        # Refusals were 50% of the SFT data (the fact check rejected many teacher answers, no refusals) and 60% of
        # the DPO pairs, while only ~25% of questions can't be answered. Trained on that, a model refuses too often.
        answers = data.filter(is_answer)
        refusals = data.filter(is_refusal).shuffle(seed=task.split_seed)
        keep = int(len(answers) * task.max_refusal_fraction / (1 - task.max_refusal_fraction))
        data = concatenate_datasets([answers, refusals.select(range(min(keep, len(refusals))))])
    return data.train_test_split(test_size=task.val_fraction, seed=task.split_seed)


def export(trainer, task: TaskArgs, configs: dict) -> dict:
    """Merge the LoRA weights into the model and save it with its tokenizer: one folder that loads like any model
    (settings.llm_model can point at it), with run.json recording how it was made."""
    metrics = trainer.evaluate()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(task.export_dir)
    trainer.processing_class.save_pretrained(task.export_dir)
    write_run_record(task.export_dir, configs, metrics)
    return metrics

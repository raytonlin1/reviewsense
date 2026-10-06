"""Supervised fine-tuning (SFT) with LoRA: teach the small LLM to answer like the verified teacher answers.

LoRA (low-rank adaptation) freezes the model and trains small extra matrices next to each layer: 1.7% of the
weights here, so it fits on a laptop CPU and the result is a small adapter. The data is prompt / completion pairs,
so the loss is computed on the answer only (the model isn't trained to write prompts).

  python -m reviewsense.finetune.sft --config configs/llm/sft-qwen3-0.6b.yaml
  python -m reviewsense.finetune.sft --config configs/llm/sft-qwen3-0.6b.yaml --learning_rate 1e-4   # override
"""
from __future__ import annotations

from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_peft_config

from .common import TaskArgs, export, load_split


def main(argv: list[str] | None = None) -> dict:
    # The YAML config fills three parts: our TaskArgs, TRL's training settings, and the model + LoRA settings.
    task, args, model_args = TrlParser((TaskArgs, SFTConfig, ModelConfig)).parse_args_and_config(argv)
    data = load_split(task)
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        peft_config=get_peft_config(model_args),        # LoRA: rank, alpha, which layers
    )
    trainer.train(resume_from_checkpoint=task.resume)
    return export(trainer, task, {"task": task, "training_args": args, "model": model_args})


if __name__ == "__main__":
    print(main())

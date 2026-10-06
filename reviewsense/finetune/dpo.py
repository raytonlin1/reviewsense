"""Direct preference optimization (DPO) with LoRA: after SFT, teach the model which of two answers is better.

Each example is a prompt with a chosen answer (verified by the fact check) and a rejected one (it cited a review
that doesn't say that, or answered a question the reviews can't answer). DPO raises the probability of the chosen
answer relative to the rejected one, compared with a frozen reference model (here: the SFT model with the new LoRA
weights switched off), and beta limits how far it may move away from that reference.
This is the same idea as RLHF (learning from preferences), without training a separate reward model.

  python -m reviewsense.finetune.dpo --config configs/llm/dpo-qwen3-0.6b.yaml
"""
from __future__ import annotations

from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser, get_peft_config

from .common import TaskArgs, export, load_split


def main(argv: list[str] | None = None) -> dict:
    task, args, model_args = TrlParser((TaskArgs, DPOConfig, ModelConfig)).parse_args_and_config(argv)
    data = load_split(task)
    trainer = DPOTrainer(
        model=model_args.model_name_or_path,             # the SFT model from the previous step
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        peft_config=get_peft_config(model_args),
    )
    trainer.train(resume_from_checkpoint=task.resume)
    return export(trainer, task, {"task": task, "training_args": args, "model": model_args})


if __name__ == "__main__":
    print(main())

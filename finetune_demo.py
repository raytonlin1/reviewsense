"""Part 11 demo: a small LLM fine-tuned (LoRA SFT + DPO) on answers distilled from a bigger one.

    python finetune_demo.py
Needs the data and models first (about 40 minutes on a laptop CPU):
    python -m reviewsense.finetune.build_data --n_examples 400
    python -m reviewsense.finetune.sft --config configs/llm/sft-qwen3-0.6b.yaml
    python -m reviewsense.finetune.build_pairs
    python -m reviewsense.finetune.dpo --config configs/llm/dpo-qwen3-0.6b.yaml
"""
import json
from pathlib import Path

from transformers.utils import logging as transformers_logging

from reviewsense.data import load_sample
from reviewsense.finetune.evaluate import compare
from reviewsense.rag.assistant import ReviewAssistant, load_llm
from reviewsense.search.engine import SearchEngine

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable
transformers_logging.set_verbosity_error()          # and generation-settings notices

BASE, TUNED = "Qwen/Qwen3-0.6B", "artifacts/qwen3-0.6b-sft"           # SFT is the released model
if not Path("artifacts/qwen3-0.6b-dpo").exists():
    raise SystemExit(__doc__)

# 1. The training data: what the model learned from.
sft_rows = [json.loads(line) for line in open("artifacts/llm-data/sft.jsonl")]
dpo_rows = [json.loads(line) for line in open("artifacts/llm-data/dpo.jsonl")]
print(f"SFT examples: {len(sft_rows)}   DPO pairs (the SFT model's own mistakes): {len(dpo_rows)}\n")
pair = next(row for row in dpo_rows if row["chosen"][0]["content"] != "The reviews don't say.")
print("A DPO pair (same prompt; rejected = what the SFT model wrote):")
print("  chosen:  ", pair["chosen"][0]["content"][:200])
print("  rejected:", pair["rejected"][0]["content"][:200], "\n")

# 2. The same questions answered by the base model and the fine-tuned one.
engine = SearchEngine(load_sample())
for model in [BASE, TUNED]:
    assistant = ReviewAssistant(engine, load_llm(model))
    print(f"== {model}")
    for question in ["Does Sunrise Cafe have vegan options?", "Is the coffee good at Sunrise Cafe?",
                     "What did the customer eat before getting sick?"]:
        print(f"Q: {question}\n   {assistant.ask(question)['answer'][:220]}")
    print()

# 3. Measured on the Part 9 questions (about reviews the training never saw). p = McNemar test against the base
#    model: below 0.05 means the accuracy difference is real, not noise.
print("Comparison (unsupported = share of answer sentences the Part 10 fact check would remove):")
for model, scores in compare([BASE, TUNED, "artifacts/qwen3-0.6b-dpo", "Qwen/Qwen3-1.7B"]).items():
    p_value = f"  p={scores['mcnemar_p']}" if "mcnemar_p" in scores else "  (baseline)"
    print(f"  {model:28} correct {scores['accuracy']:.0%}{p_value}  cited {scores['supported_by_citation']:.0%}  "
          f"unsupported {scores['unsupported_sentences']:.0%}  {scores['seconds_per_answer']}s/answer")

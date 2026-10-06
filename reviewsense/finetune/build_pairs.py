"""Build DPO preference pairs from the model being trained ("on-policy"): its own wrong answers are the rejected ones.

For each SFT prompt, the small model answers. If that answer fails the checks (a sentence the fact check removes,
no citation, or an answer to a question the reviews can't answer), the pair is:
    chosen = the SFT target (a verified teacher answer, or "The reviews don't say.")   rejected = the model's answer
Measured, in order:
  1. Pairs where "chosen" was the teacher's answer with unsupported sentences cut out: "chosen" was shorter in 178 of
     178 pairs, so DPO learned that shorter is better and refused more (correct 86% -> 67%).
  2. Pairs from the base model's mistakes: lessons SFT had already taught (71%).
  3. Pairs from the SFT model's own mistakes (this default): only 44 pairs, 35 of them refusals; capped to 12.
None beat SFT alone, and the differences were within noise (see finetune/evaluate.py), so DPO was not released.
With more data, the next step would be more prompts, not more tuning of these few pairs.

  python -m reviewsense.finetune.build_pairs --model artifacts/qwen3-0.6b-sft
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from haystack import Document
from haystack.dataclasses import ChatMessage

from ..rag.assistant import DECLINE, citations, load_llm
from ..safety.fact_check import FactChecker

# The prompt shows passages as "[3] (Restaurant 7) text"; turn them back into documents for the fact check.
PASSAGE = r"^\[(\d+)\] \((.+?)\) (.+)$"


def prompt_documents(prompt: list[dict]) -> list[Document]:
    documents = []
    for line in prompt[1]["content"].splitlines():
        match = re.match(PASSAGE, line)
        if match:
            documents.append(Document(content=match.group(3), meta={"business": match.group(2)}))
    return documents


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build on-policy DPO pairs from the SFT data.")
    parser.add_argument("--model", default="artifacts/qwen3-0.6b-sft", help="the model DPO will train")
    parser.add_argument("--sft_file", default="artifacts/llm-data/sft.jsonl")
    parser.add_argument("--out_file", default="artifacts/llm-data/dpo.jsonl")
    args = parser.parse_args(argv)

    policy = load_llm(args.model)
    fact_checker = FactChecker()
    rows = [json.loads(line) for line in Path(args.sft_file).read_text().splitlines()]
    pairs = []
    for number, row in enumerate(rows, start=1):
        target = row["completion"][0]["content"]
        documents = prompt_documents(row["prompt"])
        messages = [ChatMessage.from_system(row["prompt"][0]["content"]), ChatMessage.from_user(row["prompt"][1]["content"])]
        answer = policy.run(messages=messages)["replies"][0].text.strip()
        if target == DECLINE:
            wrong = bool(citations(answer, documents))                   # answered what the reviews can't answer
        else:
            wrong = not citations(answer, documents) or bool(fact_checker.check(answer, documents)["removed"])
        if wrong:
            pairs.append({"prompt": row["prompt"], "chosen": row["completion"],
                          "rejected": [{"role": "assistant", "content": answer}]})
        print(f"{number}/{len(rows)}  pairs {len(pairs)}", flush=True)
    Path(args.out_file).write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    print(f"wrote {len(pairs)} pairs to {args.out_file}")


if __name__ == "__main__":
    main()

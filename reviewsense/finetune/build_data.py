"""Build fine-tuning data for the small LLM by distillation: the bigger model (teacher) answers, the fact check keeps
only the answers it can verify.

For each training example:
  1. Take 5 passages from real Yelp reviews (never the 12 sample reviews the assistant is evaluated on).
  2. The teacher (Qwen3-1.7B) writes a question that ONE of the passages answers.
     Answerable example: the 5 passages include that one. Unanswerable example: they don't.
  3. The teacher answers with Part 9's prompt. Part 10's fact check verifies every sentence.
Output: sft.jsonl in prompt / completion format (training learns the answer, not the prompt). The target is a fully
verified teacher answer, or "The reviews don't say." when unanswerable. Answers that fail the checks are not used.
DPO pairs are built afterwards from the trained model's own mistakes (build_pairs.py).

  python -m reviewsense.finetune.build_data --n_examples 300 --out_dir artifacts/llm-data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from haystack import Document
from haystack.dataclasses import ChatMessage

from haystack.components.builders import ChatPromptBuilder

from ..data import clean, load_hf_yelp, split_sentences
from ..rag.assistant import ANSWER_PROMPT, DECLINE, citations, load_llm
from ..safety.fact_check import FactChecker

QUESTION_PROMPT = """Here is a sentence from a restaurant review:
{passage}
Write one short question a customer could ask that this sentence answers. Do not mention the review.
Reply with only the question."""


def passages_from_yelp(n_reviews: int, seed: int) -> list[Document]:
    """2-sentence passages, each labelled with a made-up restaurant name (the Yelp download has no names)."""
    passages = []
    for number, text in enumerate(load_hf_yelp("train", n_reviews)["text"]):
        sentences = split_sentences(clean(text))[:4]
        for start in range(0, len(sentences), 2):
            passages.append(Document(content=" ".join(sentences[start:start + 2]),
                                     meta={"business": f"Restaurant {number % 40 + 1}"}))
    random.Random(seed).shuffle(passages)
    return passages


def as_dicts(messages: list[ChatMessage]) -> list[dict]:
    """Haystack messages -> the {"role", "content"} format TRL and Hugging Face chat templates use."""
    return [{"role": message.role.value, "content": message.text} for message in messages]


def verified(answer: str, documents: list[Document], fact_checker: FactChecker) -> bool:
    """Every sentence supported by the passage it cites, and at least one citation."""
    return bool(citations(answer, documents)) and not fact_checker.check(answer, documents)["removed"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build SFT data by distillation from the bigger LLM.")
    parser.add_argument("--n_examples", type=int, default=300)
    parser.add_argument("--unanswerable_fraction", type=float, default=0.25)
    parser.add_argument("--out_dir", default="artifacts/llm-data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    teacher = load_llm()
    fact_checker = FactChecker()
    prompt_builder = ChatPromptBuilder(template=ANSWER_PROMPT)
    pool = passages_from_yelp(args.n_examples * 2, args.seed)
    sft = []
    for i in range(args.n_examples):
        # 1. A question from one passage, and 5 passages that do or don't contain its answer.
        target = pool[i]
        question = teacher.run(messages=[ChatMessage.from_user(QUESTION_PROMPT.format(passage=target.content))],
                               generation_kwargs={"max_new_tokens": 40})["replies"][0].text.strip()
        others = rng.sample(pool[args.n_examples:], 5)
        answerable = rng.random() >= args.unanswerable_fraction
        documents = others[:4] + [target] if answerable else others
        rng.shuffle(documents)
        messages = prompt_builder.run(documents=documents, question=question)["prompt"]
        prompt = as_dicts(messages)

        # 2. The teacher answers; the checks decide what is a good answer and what is a bad one.
        reply = teacher.run(messages=messages)["replies"][0].text.strip()
        if answerable:
            if not verified(reply, documents, fact_checker) or target not in citations(reply, documents):
                continue                                       # not verified: never used as a training target
            ideal = reply
        else:
            ideal = DECLINE
        sft.append({"prompt": prompt, "completion": [{"role": "assistant", "content": ideal}]})
        print(f"{i + 1}/{args.n_examples}  examples {len(sft)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sft))
    print(f"wrote {len(sft)} SFT examples to {out}")


if __name__ == "__main__":
    main()

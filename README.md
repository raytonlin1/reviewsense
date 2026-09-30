---
title: ReviewSense
emoji: 🍜
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
---

# ReviewSense: a voice-enabled conversational assistant for customer reviews

Ask about restaurants by typing or speaking, in any of 200 languages. ReviewSense turns reviews into business metrics
(aspect sentiment, emotions, star prediction), searches them with a hybrid engine, answers questions with
grounded RAG and a knowledge graph, and learns from 👍/👎 feedback with DPO.
ReviewSense turns thousands of unstructured customer reviews into answers and business metrics. Users can ask "Is the service slow at Golden Dragon?" by voice in any language. The system transcribes and translates the question, identifies the intent, and retrieves evidence with hybrid search. It then answers with a small LLM grounded in cited reviews and checks the answer for hallucination. It also produces per-business dashboards of aspect-level sentiment and emotions. Every component, from attention to the dialog manager, is implemented and evaluated. User feedback is logged and used for preference tuning.

**Course:** see [COURSE.md](COURSE.md). It explains every component, the theory behind it, and what interviewers ask about it.

```
 🎤 Whisper ASR ──┐                                           ┌── 🔊 MMS-TTS
 ⌨️  text ────────┼─► language ID ─► NLLB translate ─► guard rails (PII · injection · semantic router · toxicity)
                  │                                                  │
                  ▼                                                  ▼
         Dialog engine (ch 12): multi-label intents · slots · conversation graph in SQLite · confirmations
            │             │                │                        │                         │
            ▼             ▼                ▼                        ▼                         ▼
     Text mining     Knowledge graph   Hybrid search           RAG (Qwen 0.5B              Summaries
     BERT stars,     (ch 11) coref →   autocorrect → BM25 +     + LoRA/DPO adapters)        TextRank +
     GoEmotions,     NER → dep-parse   FAISS-HNSW + LSI → RRF   → NLI groundedness          DistilBART;
     aspects via     → relations →     → cross-encoder rerank    check                      RoBERTa extractive QA
     dependencies    networkx
                                        ▲
     Transformer + GPT from scratch (ch 9) · CYK · WSD · discourse · edit distance: the "show you understand it" layer
```

## Quickstart
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
pytest -q                                          # fast tests, no model downloads
python -m reviewsense.scratch.train_toy translate  # from-scratch Transformer, ~3 min on CPU
python app.py                                      # UI at http://localhost:7860, REST docs at /docs
```
Train your own BERT (GPU recommended; a free Colab T4 works):
```bash
python -m reviewsense.text_mining.train_sentiment --n_train 20000 --epochs 2
```
Use the real Yelp Open Dataset (business names make the knowledge graph and search more interesting):
```bash
YELP_REVIEWS=yelp_academic_dataset_review.json YELP_BUSINESSES=yelp_academic_dataset_business.json python app.py
```

## Deploy
**Hugging Face Spaces (free):** create a Space with the Docker SDK, then `git push` this repo. The YAML header above configures it.
The free CPU tier has 16 GB of RAM, which fits every model listed here.
**AWS (it's an Amazon role, so this is worth mentioning):** `docker build -t reviewsense .`, push to ECR, and run on App Runner or ECS Fargate
(4 vCPU / 16 GB), or host the BERT model on a SageMaker endpoint and swap `llm/generator.py` for a Bedrock call.

## Layout
| Path | Paradigm / chapter |
|---|---|
| `reviewsense/scratch/` | Transformer, attention, positional encoding, GPT (ch 9, 10.2) |
| `reviewsense/text_mining/` | BERT fine-tuning (ch 9.3), sentiment, emotions, aspects, business KPIs |
| `reviewsense/linguistics/` | POS, NER, dependency/constituency parsing, CYK, WSD, discourse, coreference |
| `reviewsense/ir/` | edit distance, autocorrect, BM25, LSI, FAISS ANN + PQ, hybrid search, rerank, IR metrics |
| `reviewsense/nlp/` | summarization, extractive QA, translation |
| `reviewsense/llm/` | RAG, hallucination check, guard rails, semantic routing, red teaming, LoRA SFT, DPO (ch 10) |
| `reviewsense/kg/` | information extraction pipeline and knowledge graph QA (ch 11) |
| `reviewsense/chatbot/` | NLU, dialog engine, evaluation (ch 12) |
| `reviewsense/speech/` | Whisper ASR, MMS TTS |
| `integrations/` | Haystack, LangChain, and Rasa versions (ch 10.3, 12.5) |
| `experiments/` | bootstrap CIs, McNemar, permutation tests |
| `deploy/` | INT8 quantization, ONNX export, model cache warmup |

## Resume bullets (replace the brackets with your own measured numbers)
- Built and deployed **ReviewSense**, a voice-enabled conversational NLP system (PyTorch, Hugging Face, FastAPI, Docker) that combines
  ASR (Whisper), multilingual translation (NLLB-200), RAG with a knowledge graph, and TTS. Live at [link].
- Fine-tuned **BERT** with a multi-task head (5-class and regression) on Yelp reviews, reaching [X]% accuracy / [Y] star MAE. Aspect-based
  sentiment over dependency parses surfaced service and wait time as the main drivers of negative reviews.
- Designed a **hybrid retrieval** engine (BM25 + FAISS-HNSW + LSI, reciprocal rank fusion, cross-encoder reranking). It improved
  nDCG@5 from [a] to [b] over BM25 alone; the evaluation used bootstrap confidence intervals.
- Built a **human-in-the-loop** feedback pipeline: 👍/👎 ratings become preference pairs used for **DPO** (TRL + LoRA). Also added guard rails
  (semantic routing, PII redaction, NLI groundedness) and a red-team suite that cut attack success from [p]% to [q]%.
- Implemented a Transformer encoder-decoder and a GPT from scratch in PyTorch, plus CYK parsing, Lesk/BERT word-sense disambiguation, and
  noisy-channel spelling correction.
  

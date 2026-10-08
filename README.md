# ReviewSense

A conversational NLP system for customer reviews, built part by part with the libraries companies use
(Hugging Face, spaCy, scikit-learn, Haystack). All parts (0–15) are in this repository.

## Setup (Python 3.14)
```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python -m pytest                  # fast tests (seconds)
python -m pytest -m integration   # tests that download and run real models (minutes)
```

## Train and gate a model (Part 2)
```bash
python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/tfidf-baseline.yaml
python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/distilbert-20k.yaml
python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k --gate
```
Every run writes `export_dir/` with the model, `run.json` (configs, metrics, git commit, library versions) and
`eval.json` (per-review predictions for the gate). Any config key can be overridden on the command line,
e.g. `--learning_rate 3e-5`.

## Layout
| Path | Part | Demo |
|---|---|---|
| `reviewsense/config.py`, `reviewsense/__init__.py` | 0: settings from environment variables / `.env` | |
| `reviewsense/data.py` | 0–1: loaders, cleaning (ftfy, BeautifulSoup), sentence splitting (spaCy) | `test.py` |
| `reviewsense/text_mining/train_sentiment.py`, `configs/sentiment/` | 2: TF-IDF baseline and BERT fine-tuning | |
| `reviewsense/runinfo.py`, `experiments/significance.py` | 2: run records and the release gate | |
| `reviewsense/text_mining/analyzer.py` | 3: stars, emotions, aspect sentiment per business | `analyze_demo.py` |
| `reviewsense/text_mining/topics.py` | 4: topic discovery (TF-IDF + NMF) | `topics_demo.py` |
| `reviewsense/linguistics/` | 5: shared spaCy pipeline with entity rules, NER evaluation | `linguistics_demo.py` |
| `reviewsense/kg/` | 6: fact extraction into a knowledge graph, question answering | `kg_demo.py` |
| `reviewsense/search/` | 7: hybrid search (BM25 + embeddings + reranker), spelling, evaluation | `search_demo.py` |
| `reviewsense/nlp/` | 8: review highlights (LexRank), question answering (extractive reader), translation | `nlp_demo.py` |
| `reviewsense/rag/` | 9: LLM answers and summaries from retrieved reviews, with citations (RAG) | `rag_demo.py` |
| `reviewsense/safety/` | 10: personal data removal, injection and harm guard, fact check, red team | `safety_demo.py` |
| `reviewsense/chat/` | Conversations with LangChain + LangGraph: chat memory, follow-up questions | `chat_demo.py` |
| `reviewsense/finetune/`, `configs/llm/` | 11: distillation data, LoRA SFT and DPO of a small LLM, model comparison | `finetune_demo.py` |
| `reviewsense/chat/` (`intents`, `slots`, `dialog`, `bookings`) | 12: chatbot: intents (SetFit), slots, dialog manager, table booking | `chatbot_demo.py` |
| `reviewsense/speech/` | 13: speech to text (faster-whisper + hotwords), text to speech (VITS), voice assistant, WER | `voice_demo.py` |
| `reviewsense/serving/`, `Dockerfile` | 14: FastAPI service, Gradio page at /ui, ONNX INT8 star model, container | `serve_demo.py` |

## Serve it
```bash
python -m reviewsense.serving.onnx_model export      # once: the star model as ONNX / INT8
uvicorn reviewsense.serving.api:app --port 8000      # API docs: http://localhost:8000/docs, web page: /ui
```
| `experiments/` | 2 and 15: release gate (McNemar, bootstrap) and A/B tests (power, assignment, SRM, Holm) | `experiments_demo.py` |

## Results (all measured in this repository; how each was measured is in the module docstrings)
| Part | Task | Result |
|---|---|---|
| 2 | Star rating (Yelp, 2k test reviews) | TF-IDF 57.3% -> DistilBERT 61.0% accuracy (+3.7 points, 95% CI [+1.3, +6.0], McNemar p = 0.003) |
| 5 | Named entities (hand-labelled reviews) | F1 0.19 (standard spaCy) -> 0.88 (rules + catalog) -> 0.93 (+ short names) |
| 6 | Fact extraction for the knowledge graph | F1 0.97 |
| 7 | Search (hand-labelled queries) | nDCG@5 0.77 (BM25) -> 0.93 (hybrid + reranker), MRR 1.0 |
| 8 | Extractive QA with "no answer" | 8/14 -> 13/14 correct (passage count and confidence threshold tuned) |
| 8 | Translation to English | chrF 70 -> 88 (per-language models instead of one multilingual model) |
| 9 | RAG answers with citations (21 questions) | 90% correct, every correct answer backed by a cited review |
| 10 | Safety: fact check / input guard | 8/8 unsupported claims caught; held-out attacks 89% blocked, 0% normal questions blocked |
| chat | Follow-up questions in conversations | 80% -> 100% resolved (rewrite check + restaurant memory) |
| 11 | Fine-tuned Qwen3-0.6B (LoRA SFT on distilled, fact-checked answers) | 86% correct, unsupported sentences 44% -> 11%, 3x faster than the 1.7B teacher (90%, 59%) |
| 12 | Intents (SetFit) / booking dialogs | 90% vs 69% TF-IDF; 100% turn accuracy, 100% booking success |
| 13 | Speech recognition (faster-whisper small.en) | 4.6% WER on LibriSpeech; restaurant names right 15/15 (11/15 without vocabulary) |
| 14 | Serving the star model | ONNX 2x faster (8.7 vs 16.1 ms); INT8 4x smaller (67 MB), same accuracy (p = 0.40) |
| 15 | A/B testing (simulated) | Daily peeking: 23% false wins vs 6% when analyzed once at the planned size |

Small test sets (10-30 items) were used for several parts: their numbers show direction, and several differences
were tested and found not significant (noted in the code). Real traffic would need larger sets.

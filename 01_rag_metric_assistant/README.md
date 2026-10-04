# 01 RAG Metric Assistant

A small assistant that answers questions about business metrics using only approved definitions. It measures retrieval quality with a test set, checks its choices on held-out questions, and refuses to answer when the definitions don't cover the question.

## Problem

Teams often define the same metric in different ways. If an AI system answers from memory, it can give a confident but wrong definition. This project grounds every answer in a library of approved definitions and cites the definition it used.

## How it works

1. **Load** 30 metric definitions from `data/definitions.json` (made-up, generic customer service metrics).
2. **Embed** each definition (metric name plus text) with `all-mpnet-base-v2` from sentence-transformers.
3. **Store** the vectors in a FAISS index. Vectors are normalized, so inner product equals cosine similarity.
4. **Retrieve** the top 3 definitions for a question.
5. **Generate** an answer with an LLM (Claude Haiku through the Anthropic API). The prompt says: answer only from the context, cite definition ids, and say "I don't know" if the context lacks the answer.
6. **Evaluate** retrieval on 15 test questions written in different words from the definitions, then confirm the final model choice on 5 held-out questions.

## Project structure

```
01_rag_metric_assistant/
  data/
    definitions.json              30 made-up metric definitions
    test_questions.json           15 questions with expected definition ids
    holdout_questions.json        5 held-out questions
    out_of_scope_questions.json   3 questions the definitions can't answer
  rag.py            retrieval and prompt building
  llm.py            LLM call (Anthropic or OpenAI, set in .env)
  ask.py            command line question answering
  evaluate.py       hit rate and in-scope vs out-of-scope scores
  experiments.py    compares embedding models and text inputs
  tests/test_rag.py unit tests
```

## Run it

From the repo root, create a `.env` file from `.env.example` and add your API key and model name. Then:

```bash
cd 01_rag_metric_assistant
python rag.py                 # retrieval demo, no API key needed
python evaluate.py            # retrieval metrics
python experiments.py         # model and input comparison
python ask.py                 # interactive Q&A (needs an API key)
python -m pytest              # unit tests
```

The first run downloads the embedding model. A Hugging Face warning about unauthenticated requests is harmless.

## Results

Final setup: `all-mpnet-base-v2`, metric name plus text, top 3 retrieval.

| Metric | Value |
|---|---|
| Hit rate @1 | 0.73 (11 of 15) |
| Hit rate @3 | 1.00 (15 of 15) |
| Average top score, in-scope questions | 0.51 |
| Average top score, out-of-scope questions | 0.23 |

### Experiments

Each setup was scored on the 15-question test set.

| Setup | Hit @1 | Hit @3 | Hit @5 |
|---|---|---|---|
| MiniLM, name + text (baseline) | 0.53 | 0.73 | 0.80 |
| MiniLM, text only | 0.53 | 0.73 | 0.73 |
| MiniLM, name only | 0.33 | 0.47 | 0.60 |
| **MPNet, name + text (final)** | **0.73** | **1.00** | **1.00** |

Holdout check on 5 new questions, used to confirm the final choice:

| Setup | Hit @1 | Hit @3 |
|---|---|---|
| MiniLM, name + text | 0.80 | 1.00 |
| MPNet, name + text | 1.00 | 1.00 |

**Takeaways**

- The embedding model had the largest effect. Moving from MiniLM to MPNet raised hit@3 from 0.73 to 1.00.
- The metric name helps. Embedding the name alone performed worst, and name plus text was the safer input.
- I chose the model using the test set, so I checked it on held-out questions, where MPNet matched or beat MiniLM. With only 5 questions, the holdout confirms direction, not the size of the gain.
- MPNet is about 5 times larger than MiniLM (roughly 440 MB vs 90 MB). That is negligible for 30 definitions and would matter at scale.

### Failure analysis

The baseline model missed 4 of 15 questions entirely. Two examples:

- "Solved without having to call back" ranked Repeat Contact Rate above First Contact Resolution. The two metrics describe the same idea from opposite sides, so their definitions sit close together.
- "Hang up before anyone picks up" pulled speed-of-answer metrics ahead of Abandonment Rate, because "picks up" overlaps with how long customers wait.

With the final model, every expected definition appears in the top 3. Four questions still rank it second or third, which is why retrieving 3 definitions matters for generation: the LLM reads all three and cites the one that fits.

### Example

```
Question: What share of people hang up in queue?
Retrieved: m09 (0.51), m10 (0.43), m11 (0.36)
Answer: The share of people who hang up in queue is measured by Abandonment
Rate [m09], which is the percentage of customers who hang up or disconnect
while waiting in the queue, before reaching an employee.
```

Out-of-scope question:

```
Question: What is the weather in Toronto?
Retrieved: m29 (0.10), m19 (0.09), m28 (0.07)
Answer: I don't know based on the provided definitions.
```

## Design choices

- **Local embeddings.** Retrieval runs on my machine, and only the retrieved definitions and the question go to the LLM.
- **Strict system prompt.** It limits answers to the retrieved context, requires citations, and defines the refusal sentence. The model also declines to invent formulas that the definition doesn't contain.
- **Provider-agnostic LLM function.** `llm.py` reads the provider and model from `.env`, so the model can be swapped without touching retrieval.
- **Held-out check.** A second question set guards against tuning the pipeline to the test set.

## Limitations

- 30 short definitions is a small corpus, so chunking wasn't needed and results may not carry over to long documents.
- The test set has 15 questions, the holdout has 5, and the out-of-scope set has 3. Each question moves a score by 7 to 33 points, so results are directional.
- A score cutoff alone would be fragile for refusals. In an earlier run with the smaller model, a customer-themed out-of-scope question scored higher than some valid questions because it shared vocabulary with real definitions. The prompt-level refusal carries most of the weight.
- No keyword search, reranking, or metadata filtering yet.

## Next steps

- Hybrid search (keyword plus vector) and a reranker for near-neighbor metrics
- A larger test set and harder out-of-scope questions
- An LLM-based groundedness check on generated answers
- Serve behind a FastAPI endpoint with logging and monitoring
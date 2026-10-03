# 01 RAG Metric Assistant

A small assistant that answers questions about business metrics using only approved definitions. It measures retrieval quality with a test set and refuses to answer when the definitions don't cover the question.

## Problem

Teams often define the same metric in different ways. If an AI system answers from memory, it can give a confident but wrong definition. This project grounds every answer in a library of approved definitions and cites the definition it used.

## How it works

1. **Load** 30 metric definitions from `data/definitions.json` (made-up, generic customer service metrics).
2. **Embed** each definition with `all-MiniLM-L6-v2` (sentence-transformers).
3. **Store** the vectors in a FAISS index (inner product on normalized vectors, which equals cosine similarity).
4. **Retrieve** the top 3 definitions for a question.
5. **Generate** an answer with an LLM, using a prompt that says: answer only from the context, cite ids, and say "I don't know" if the context lacks the answer.
6. **Evaluate** retrieval on 15 test questions that are phrased differently from the definitions.
from fastapi import FastAPI
from pydantic import BaseModel

from llm import ask_llm
from rag import build_index, build_prompt, load_docs, retrieve

app = FastAPI(title="Metric Assistant")

docs = load_docs()
index = build_index(docs)  # built once at startup, reused for every request


class Question(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "definitions": len(docs)}


@app.post("/ask")
def ask(body: Question):
    hits = retrieve(body.question, docs, index, body.k)
    system, user = build_prompt(body.question, hits)
    return {
        "answer": ask_llm(system, user),
        "sources": [{"id": d["id"], "score": round(s, 3)} for d, s in hits],
    }
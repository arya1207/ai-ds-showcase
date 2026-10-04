import json
from functools import lru_cache
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer("all-mpnet-base-v2")

def load_docs(path: Path = DATA_DIR / "definitions.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_index(docs: list[dict]) -> faiss.Index:
    texts = [f"{d['name']}: {d['text']}" for d in docs]
    vecs = get_model().encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index    

def retrieve(question: str, docs: list[dict], index: faiss.Index, k: int = 3):
    q = get_model().encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [(docs[i], float(s)) for i, s in zip(ids[0], scores[0])]


def build_prompt(question: str, hits) -> tuple[str, str]:
    context = "\n".join(f"[{d['id']}] {d['name']}: {d['text']}" for d, _ in hits)
    system = (
        "You answer questions about business metrics. "
        "Answer only from the context. Cite definition ids in brackets, like [m01]. "
        "If the context does not contain the answer, say: I don't know based on the provided definitions."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return system, user

if __name__ == "__main__":
    docs = load_docs()
    index = build_index(docs)
    question = "How long does a typical call take?"
    for doc, score in retrieve(question, docs, index):
        print(f"{score:.3f} [{doc['id']}] {doc['name']}")
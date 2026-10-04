import json

import faiss
from sentence_transformers import SentenceTransformer

from rag import DATA_DIR, load_docs

K_VALUES = (1, 3, 5)

CONFIGS = [
    ("baseline: MiniLM, name + text", "all-MiniLM-L6-v2", lambda d: f"{d['name']}: {d['text']}"),
    ("MiniLM, text only", "all-MiniLM-L6-v2", lambda d: d["text"]),
    ("MiniLM, name only", "all-MiniLM-L6-v2", lambda d: d["name"]),
    ("MPNet, name + text", "all-mpnet-base-v2", lambda d: f"{d['name']}: {d['text']}"),
]


def run(model_name: str, text_fn, test_file: str = "test_questions.json") -> dict[int, float]:
    docs = load_docs()
    with open(DATA_DIR / test_file, encoding="utf-8") as f:
        tests = json.load(f)

    model = SentenceTransformer(model_name)
    doc_vecs = model.encode([text_fn(d) for d in docs], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)

    q_vecs = model.encode([t["q"] for t in tests], normalize_embeddings=True).astype("float32")
    _, ids = index.search(q_vecs, max(K_VALUES))

    results = {}
    for k in K_VALUES:
        hits = sum(
            t["expected_id"] in [docs[i]["id"] for i in row[:k]]
            for t, row in zip(tests, ids)
        )
        results[k] = hits / len(tests)
    return results


if __name__ == "__main__":
    print(f"{'config':<34} hit@1  hit@3  hit@5")
    for label, model_name, text_fn in CONFIGS:
        r = run(model_name, text_fn)
        print(f"{label:<34} {r[1]:.2f}   {r[3]:.2f}   {r[5]:.2f}")

    print("\nHoldout (5 new questions)")
    for label, model_name, text_fn in [CONFIGS[0], CONFIGS[3]]:
        r = run(model_name, text_fn, "holdout_questions.json")
        print(f"{label:<34} {r[1]:.2f}   {r[3]:.2f}   {r[5]:.2f}")
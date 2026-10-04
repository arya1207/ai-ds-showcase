import json

from rag import DATA_DIR, build_index, load_docs, retrieve


def load_json(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def evaluate(k: int = 3) -> None:
    docs = load_docs()
    index = build_index(docs)
    tests = load_json("test_questions.json")

    hit_at_1 = hit_at_k = 0
    top_scores, misses = [], []

    for t in tests:
        hits = retrieve(t["q"], docs, index, k)
        ids = [d["id"] for d, _ in hits]
        top_scores.append(hits[0][1])
        hit_at_1 += ids[0] == t["expected_id"]
        if t["expected_id"] in ids:
            hit_at_k += 1
        else:
            misses.append((t["q"], t["expected_id"], ids))

    n = len(tests)
    print(f"hit@1: {hit_at_1 / n:.2f}   hit@{k}: {hit_at_k / n:.2f}")
    print(f"avg top score (in scope): {sum(top_scores) / n:.3f}")

    for q, expected, got in misses:
        print(f"MISS: {q}\n  expected {expected}, got {got}")

    out_scores = []
    for t in load_json("out_of_scope_questions.json"):
        out_scores.append(retrieve(t["q"], docs, index, 1)[0][1])
    print(f"avg top score (out of scope): {sum(out_scores) / len(out_scores):.3f}")


if __name__ == "__main__":
    evaluate()
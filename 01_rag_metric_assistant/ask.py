from llm import ask_llm
from rag import build_index, build_prompt, load_docs, retrieve


def main() -> None:
    docs = load_docs()
    index = build_index(docs)
    print("Ask about a metric (blank line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        hits = retrieve(question, docs, index, k=3)
        system, user = build_prompt(question, hits)
        print("\nRetrieved:", ", ".join(f"{d['id']} ({s:.2f})" for d, s in hits))
        print("\n" + ask_llm(system, user))


if __name__ == "__main__":
    main()
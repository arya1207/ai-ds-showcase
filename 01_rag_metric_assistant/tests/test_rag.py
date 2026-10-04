import pytest

from rag import build_index, load_docs, retrieve


@pytest.fixture(scope="module")
def setup():
    docs = load_docs()
    return docs, build_index(docs)


def test_returns_k_results(setup):
    docs, index = setup
    assert len(retrieve("average call length", docs, index, k=3)) == 3


def test_known_question_finds_expected_definition(setup):
    docs, index = setup
    hits = retrieve("What fraction of calls are answered within the target number of seconds?", docs, index, k=3)
    assert "m08" in [d["id"] for d, _ in hits]


def test_scores_are_sorted_high_to_low(setup):
    docs, index = setup
    scores = [s for _, s in retrieve("employee turnover", docs, index, k=3)]
    assert scores == sorted(scores, reverse=True)
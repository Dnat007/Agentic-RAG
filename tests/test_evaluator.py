from langchain_core.documents import Document

from app.retrieval.evaluator import ContextEvaluator


def create_document():
    return Document(
        page_content="Azure AI Search supports hybrid search.",
        metadata={"chunk_id": "chunk_1"},
    )


def test_context_is_relevant():

    evaluator = ContextEvaluator(
        min_documents=1,
        min_score=0.5,
    )

    results = [
        (create_document(), 0.85),
    ]

    evaluation = evaluator.evaluate(
        "What is hybrid search?",
        results,
    )

    assert evaluation["is_relevant"] is True
    assert evaluation["score"] == 0.85


def test_context_is_not_relevant():

    evaluator = ContextEvaluator(
        min_documents=1,
        min_score=0.5,
    )

    results = [
        (create_document(), 0.2),
    ]

    evaluation = evaluator.evaluate(
        "What is hybrid search?",
        results,
    )

    assert evaluation["is_relevant"] is False


def test_no_results():

    evaluator = ContextEvaluator()

    evaluation = evaluator.evaluate(
        "What is hybrid search?",
        [],
    )

    assert evaluation["is_relevant"] is False
    assert evaluation["score"] == 0.0


def test_insufficient_documents():

    evaluator = ContextEvaluator(
        min_documents=2,
        min_score=0.5,
    )

    results = [
        (create_document(), 0.9),
    ]

    evaluation = evaluator.evaluate(
        "What is hybrid search?",
        results,
    )

    assert evaluation["is_relevant"] is False


def test_empty_query():

    evaluator = ContextEvaluator()

    try:
        evaluator.evaluate("", [])

        assert False

    except ValueError:
        assert True

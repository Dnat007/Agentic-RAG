from langchain_core.documents import Document

from app.retrieval.rrf import ReciprocalRankFusion


def test_rrf_fusion():

    doc_a = Document(
        page_content="Document A",
        metadata={"chunk_id": "A"},
    )

    doc_b = Document(
        page_content="Document B",
        metadata={"chunk_id": "B"},
    )

    doc_c = Document(
        page_content="Document C",
        metadata={"chunk_id": "C"},
    )

    bm25_results = [
        (doc_a, 10.0),
        (doc_b, 8.0),
        (doc_c, 5.0),
    ]

    vector_results = [
        (doc_c, 0.95),
        (doc_a, 0.90),
    ]

    rrf = ReciprocalRankFusion(k=60)

    results = rrf.fuse(
        rankings=[
            bm25_results,
            vector_results,
        ],
        top_k=3,
    )

    assert len(results) == 3

    result_ids = [
        document.metadata["chunk_id"]
        for document, _ in results
    ]

    assert result_ids[0] in {"A", "C"}
    assert "B" in result_ids
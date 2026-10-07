from langchain_core.documents import Document

from app.retrieval.bm25 import BM25Retrieval


def test_bm25_search():

    documents = [
        Document(
            page_content=(
                "Azure AI Search supports hybrid search "
                "using BM25 and vector search."
            )
        ),
        Document(
            page_content=(
                "Python is widely used for machine learning."
            )
        ),
        Document(
            page_content=(
                "Redis is commonly used for caching."
            )
        ),
    ]

    retriever = BM25Retrieval()

    retriever.build(documents)

    results = retriever.search(
        "Azure AI Search BM25",
        top_k=2,
    )

    assert len(results) == 2

    top_document, score = results[0]

    assert "Azure AI Search" in top_document.page_content
    assert score > 0

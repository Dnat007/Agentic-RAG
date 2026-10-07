from langchain_core.documents import Document

from app.retrieval.reranker import CrossEncoderReranker


def test_reranker():

    documents = [
        Document(
            page_content=(
                "The company provides 30 days of annual leave "
                "to eligible employees."
            )
        ),
        Document(
            page_content=(
                "The company uses Python for machine learning."
            )
        ),
        Document(
            page_content=(
                "Employees can apply for annual leave through HR."
            )
        ),
    ]

    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query="What is the company's annual leave policy?",
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0].page_content
    assert isinstance(results[0][1], float)

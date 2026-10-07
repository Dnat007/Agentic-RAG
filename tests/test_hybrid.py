from langchain_core.documents import Document

from app.retrieval.hybrid import HybridRetriever


def create_documents():
    return [
        Document(
            page_content=(
                "Azure AI Search supports hybrid search "
                "using BM25 and vector search."
            ),
            metadata={
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "user_id": "user_1",
                "department": "Engineering",
            },
        ),
        Document(
            page_content=(
                "Redis provides fast in-memory caching "
                "for applications."
            ),
            metadata={
                "chunk_id": "chunk_2",
                "document_id": "doc_2",
                "user_id": "user_2",
                "department": "Finance",
            },
        ),
        Document(
            page_content=(
                "Python is commonly used for "
                "machine learning."
            ),
            metadata={
                "chunk_id": "chunk_3",
                "document_id": "doc_3",
                "user_id": "user_1",
                "department": "Engineering",
            },
        ),
    ]

def test_hybrid_search():

    documents = create_documents()

    retriever = HybridRetriever()

    retriever.build(documents)

    results = retriever.search(
        "Azure AI Search hybrid search",
        top_k=2,
    )

    assert len(results) <= 2

    for document, score in results:
        assert isinstance(document, Document)
        assert isinstance(score, float)

def test_hybrid_search_with_user_acl():

    documents = create_documents()

    retriever = HybridRetriever()

    retriever.build(documents)

    results = retriever.search(
        "Redis caching",
        top_k=2,
        user_id="user_1",
    )

    for document, _ in results:
        assert document.metadata["user_id"] == "user_1"

def test_hybrid_search_with_department_filter():

    documents = create_documents()

    retriever = HybridRetriever()

    retriever.build(documents)

    results = retriever.search(
        "machine learning",
        top_k=2,
        allowed_departments=["Engineering"],
    )

    for document, _ in results:
        assert document.metadata["department"] == "Engineering"

def test_hybrid_search_with_document_filter():

    documents = create_documents()

    retriever = HybridRetriever()

    retriever.build(documents)

    results = retriever.search(
        "Azure search",
        top_k=2,
        allowed_document_ids=["doc_1"],
    )

    for document, _ in results:
        assert document.metadata["document_id"] == "doc_1"

def test_hybrid_search_empty_query():

    documents = create_documents()

    retriever = HybridRetriever()

    retriever.build(documents)

    try:
        retriever.search("")
        assert False
    except ValueError:
        assert True


def test_hybrid_search_without_authorized_documents():

    documents = create_documents()

    retriever = HybridRetriever()

    retriever.build(documents)

    results = retriever.search(
        "Azure search",
        user_id="unknown_user",
    )

    assert results == []
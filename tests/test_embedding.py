from langchain_core.documents import Document

from app.retrieval.embeddings import EmbeddingService


def test_embedding_generation():
    service = EmbeddingService()

    documents = [
        Document(page_content="What is retrieval augmented generation?")
    ]

    vectors = service.embed_doc(documents)

    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_query_embedding():
    service = EmbeddingService()

    vector = service.embed_query("What is RAG?")

    assert len(vector) > 0
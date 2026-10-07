from langchain_core.documents import Document

from app.retrieval.vector import VectorEmbedding


def create_documents():
    return [
        Document(
            page_content=(
                "Azure AI Search supports hybrid search "
                "using BM25 and vector search."
            ),
            metadata={
                "chunk_id": "chunk_1",
            },
        ),
        Document(
            page_content=(
                "Redis provides fast in-memory caching "
                "for applications."
            ),
            metadata={
                "chunk_id": "chunk_2",
            },
        ),
        Document(
            page_content=(
                "Python is commonly used for "
                "machine learning."
            ),
            metadata={
                "chunk_id": "chunk_3",
            },
        ),
    ]


def test_vector_hnsw_search():

    documents = create_documents()

    vector_store = VectorEmbedding()

    vector_store.build(documents)
    results = vector_store.search("What is Azure AI Search hybrid search?",top_k=2)

    assert len(results) == 2

    top_document, score = results[0]

    assert (
        top_document.metadata["chunk_id"]
        == "chunk_1"
    )

    assert isinstance(score, float)


def test_vector_hnsw_index_created():

    documents = create_documents()

    vector_store = VectorEmbedding()

    vector_store.build(documents)

    assert vector_store.index is not None

    assert vector_store.index.ntotal == len(documents)


def test_vector_hnsw_configuration():

    documents = create_documents()

    vector_store = VectorEmbedding(
        hnsw_m=32,
        ef_construction=200,
        ef_search=64,
    )

    vector_store.build(documents)

    assert vector_store.index.hnsw.nb_neighbors(0) > 0

    assert (
        vector_store.index.hnsw.efConstruction
        == 200
    )

    assert (
        vector_store.index.hnsw.efSearch
        == 64
    )


def test_vector_search_empty_query():

    documents = create_documents()

    vector_store = VectorEmbedding()

    vector_store.build(documents)

    try:
        vector_store.search(
            "",
            top_k=2,
        )

        assert False

    except ValueError:
        assert True


def test_vector_search_invalid_top_k():

    documents = create_documents()

    vector_store = VectorEmbedding()

    vector_store.build(documents)

    try:
        vector_store.search(
            "Azure search",
            top_k=0,
        )

        assert False

    except ValueError:
        assert True


def test_vector_build_empty_documents():

    vector_store = VectorEmbedding()

    try:
        vector_store.build([])

        assert False

    except ValueError:
        assert True
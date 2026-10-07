from langchain_core.documents import Document

from app.ingestion.chunker import DocumentsChunker


def test_document_chunking():
    document = Document(
        page_content="This is a test document. " * 200,
        metadata={
            "document_id": "test_doc",
            "document_name": "test.pdf",
            "page": 1,
        },
    )

    chunker = DocumentsChunker()

    chunks = chunker.split([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["document_id"] == "test_doc"
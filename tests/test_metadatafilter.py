from langchain_core.documents import Document

from app.security.filter import MetadataFilter


def test_department_filter():

    documents = [
        Document(
            page_content="HR leave policy",
            metadata={
                "document_id": "hr_001",
                "department": "HR",
            },
        ),
        Document(
            page_content="Finance policy",
            metadata={
                "document_id": "finance_001",
                "department": "Finance",
            },
        ),
    ]

    metadata_filter = MetadataFilter()

    results = metadata_filter.filter(
        documents,
        allowed_departments=["HR"],
    )

    assert len(results) == 1
    assert results[0].metadata["department"] == "HR"


def test_document_filter():

    documents = [
        Document(
            page_content="Document A",
            metadata={"document_id": "A"},
        ),
        Document(
            page_content="Document B",
            metadata={"document_id": "B"},
        ),
    ]

    metadata_filter = MetadataFilter()

    results = metadata_filter.filter(
        documents,
        allowed_document_ids=["B"],
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "B"

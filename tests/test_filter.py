from langchain_core.documents import Document

from app.security.filter import MetadataFilter


def test_filter_by_user_id():

    documents = [
        Document(
            page_content="HR document",
            metadata={
                "document_id": "doc_1",
                "user_id": "user_1",
                "department": "HR",
            },
        ),
        Document(
            page_content="Finance document",
            metadata={
                "document_id": "doc_2",
                "user_id": "user_2",
                "department": "Finance",
            },
        ),
    ]

    filter_service = MetadataFilter()

    results = filter_service.filter(
        documents,
        user_id="user_1",
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc_1"


def test_filter_by_department():

    documents = [
        Document(
            page_content="HR document",
            metadata={
                "document_id": "doc_1",
                "department": "HR",
            },
        ),
        Document(
            page_content="Finance document",
            metadata={
                "document_id": "doc_2",
                "department": "Finance",
            },
        ),
    ]

    filter_service = MetadataFilter()

    results = filter_service.filter(
        documents,
        allowed_departments=["Finance"],
    )

    assert len(results) == 1
    assert results[0].metadata["department"] == "Finance"


def test_filter_by_document_id():

    documents = [
        Document(
            page_content="Document 1",
            metadata={"document_id": "doc_1"},
        ),
        Document(
            page_content="Document 2",
            metadata={"document_id": "doc_2"},
        ),
    ]

    filter_service = MetadataFilter()

    results = filter_service.filter(
        documents,
        allowed_document_ids=["doc_2"],
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc_2"


def test_filter_multiple_conditions():

    documents = [
        Document(
            page_content="HR document",
            metadata={
                "document_id": "doc_1",
                "user_id": "user_1",
                "department": "HR",
            },
        ),
        Document(
            page_content="Finance document",
            metadata={
                "document_id": "doc_2",
                "user_id": "user_1",
                "department": "Finance",
            },
        ),
        Document(
            page_content="HR document",
            metadata={
                "document_id": "doc_3",
                "user_id": "user_2",
                "department": "HR",
            },
        ),
    ]

    filter_service = MetadataFilter()

    results = filter_service.filter(
        documents,
        user_id="user_1",
        allowed_departments=["Finance"],
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc_2"


def test_filter_empty_documents():

    filter_service = MetadataFilter()

    results = filter_service.filter(
        [],
        user_id="user_1",
    )

    assert results == []
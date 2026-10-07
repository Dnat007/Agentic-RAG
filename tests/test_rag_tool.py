from langchain_core.documents import Document

from app.tools.rag import RAGTool


class FakeRetriever:
    def __init__(self):
        self.built_documents = None

    def build(self, documents):
        self.built_documents = documents

    def search(
        self,
        query,
        top_k=None,
        user_id=None,
        allowed_departments=None,
        allowed_document_ids=None,
    ):
        return [
            (
                self.built_documents[0],
                0.95,
            )
        ]


def test_rag_tool_build_and_search():
    tool = RAGTool()

    fake_retriever = FakeRetriever()
    tool.retriever = fake_retriever

    documents = [
        Document(
            page_content="Company leave policy provides 20 annual leaves.",
            metadata={
                "document_id": "doc-1",
            },
        )
    ]

    tool.build(documents)

    assert tool.is_built is True
    assert fake_retriever.built_documents == documents

    results = tool.search(
        query="How many annual leaves?"
    )

    assert len(results) == 1
    assert results[0][0].page_content.startswith(
        "Company leave policy"
    )
    assert results[0][1] == 0.95
from langchain_core.documents import Document
from app.retrieval.hybrid import HybridRetriever

class RAGTool:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.is_built = False

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build RAG tool with empty documents.")

        self.retriever.build(documents)
        self.is_built = True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        allowed_departments: list[str] | None = None,
        allowed_document_ids: list[str] | None = None,
    ) -> list[tuple[Document, float]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not self.is_built:
            raise RuntimeError("RAG tool has not been built.")

        return self.retriever.search(
            query=query,
            top_k=top_k,
            user_id=user_id,
            allowed_departments=allowed_departments,
            allowed_document_ids=allowed_document_ids,
        )

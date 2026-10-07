from langchain_core.documents import Document

from app.config.settings import get_settings
from app.retrieval.bm25 import BM25Retrieval
from app.retrieval.rrf import ReciprocalRankFusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector import VectorEmbedding
from app.security.filter import MetadataFilter


class HybridRetriever:
    """
    Production-ready hybrid retriever.

    Retrieval flow:

    Query
        ↓
    Metadata / ACL Filtering
        ↓
    BM25 Search + Vector Search
        ↓
    Reciprocal Rank Fusion (RRF)
        ↓
    Cross Encoder Reranking
        ↓
    Final Documents
    """

    def __init__(self):
        settings = get_settings()
        self.top_k = settings.retrieval_top_k
        self.reranker_top_k = settings.reranker_top_k
        self.bm25 = BM25Retrieval()
        self.vector = VectorEmbedding()
        self.rrf = ReciprocalRankFusion()
        self.reranker = CrossEncoderReranker()
        self.metadata_filter = MetadataFilter()
        self.documents: list[Document] = []

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build hybrid retriever from empty documents.")

        self.documents = documents

        # Build keyword retrieval index
        self.bm25.build(documents)

        # Build vector retrieval index
        self.vector.build(documents)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        allowed_departments: list[str] | None = None,
        allowed_document_ids: list[str] | None = None,
    ):
      
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not self.documents:
            raise RuntimeError(
                "Hybrid retriever has not been built."
            )

        final_top_k = top_k or self.reranker_top_k

        allowed_documents = self.metadata_filter.filter(
            self.documents,
            user_id=user_id,
            allowed_departments=allowed_departments,
            allowed_document_ids=allowed_document_ids,
        )

        if not allowed_documents:
            return []

        bm25_results = self.bm25.search(query,top_k=self.top_k)

        bm25_results = [(document, score)
            for document, score in bm25_results
            if self._is_allowed(document,allowed_documents)
        ]

        vector_results = self.vector.search(query,top_k=self.top_k)

        vector_results = [
            (document, score)
            for document, score in vector_results
            if self._is_allowed(
                document,
                allowed_documents,
            )
        ]

        fused_results = self.rrf.fuse([bm25_results,vector_results],top_k=self.top_k)

        if not fused_results:
            return []

        fused_documents = [document for document, _ in fused_results]

        reranked_results = self.reranker.rerank(
            query=query,
            documents=fused_documents,
            top_k=final_top_k,
        )

        return reranked_results

    @staticmethod
    def _is_allowed(
        document: Document,
        allowed_documents: list[Document],
    ) -> bool:
       
        allowed_ids = {HybridRetriever._document_id(document) for document in allowed_documents}

        return (HybridRetriever._document_id(document) in allowed_ids )

    @staticmethod
    def _document_id(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)
        document_id = document.metadata.get("document_id")

        if document_id:
            return str(document_id)
        return document.page_content
from collections import defaultdict
from langchain_core.documents import Document

class ReciprocalRankFusion:

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError("RRF k must be greater than zero.")
        self.k = k

    def fuse(self,rankings: list[list[tuple[Document, float]]], top_k: int = 10):
        if not rankings:
            return []

        scores = defaultdict(float)
        documents = {}

        for ranking in rankings:
            for rank, (document, _) in enumerate(ranking,start=1):
                
                document_id = self._document_id(document)
                scores[document_id] += (1.0 / (self.k + rank))
                documents[document_id] = document

        ranked_documents = sorted(scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            (documents[document_id], score)
            for document_id, score in ranked_documents[:top_k]
        ]

    @staticmethod
    def _document_id(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)

        document_id = document.metadata.get("document_id")
        if document_id:
            return str(document_id)

        return document.page_content
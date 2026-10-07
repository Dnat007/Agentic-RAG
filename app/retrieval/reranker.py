from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.config.settings import get_settings

class CrossEncoderReranker:
    def __init__(self):
        settings = get_settings()
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self,query: str,documents: list[Document],top_k: int) :

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        # Query-document pairs
        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        scores = self.model.predict(pairs)
        results = list(zip(documents, scores))
        results.sort(key=lambda item: float(item[1]),reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return [
            (document, float(score))
            for document, score in results
        ]
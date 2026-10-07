from langchain_core.documents import Document


class ContextEvaluator:

    def __init__(
        self,
        min_documents: int = 1,
        min_score: float = -10.0,
    ):
        self.min_documents = min_documents
        self.min_score = min_score

    def evaluate(
        self,
        query: str,
        results: list[tuple[Document, float]],
    ) -> dict:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not results:
            return {
                "is_relevant": False,
                "score": 0.0,
                "reason": "No documents were retrieved.",
            }

        if len(results) < self.min_documents:
            return {
                "is_relevant": False,
                "score": 0.0,
                "reason": "Insufficient documents retrieved.",
            }

        scores = [float(score) for _, score in results]

        best_score = max(scores)

        is_relevant = best_score >= self.min_score

        if is_relevant:
            reason = "Retrieved context passed relevance threshold."
        else:
            reason = "Retrieved context did not pass relevance threshold."

        return {
            "is_relevant": is_relevant,
            "score": best_score,
            "reason": reason,
        }

import faiss
import numpy as np
from langchain_core.documents import Document
from app.retrieval.embeddings import EmbeddingService

class VectorEmbedding:
   
    def __init__(
        self,
        dimension: int | None = None,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        self.embedding_service = EmbeddingService()
        self.dimension = dimension
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None
        self.documents: list[Document] = []

    def build(self, documents: list[Document]) -> None:
    
        if not documents:
            raise ValueError("Cannot build vector index from empty documents.")

        self.documents = documents

        # Generate document embeddings
        embeddings = self.embedding_service.embed_doc(documents)
        if not embeddings:
            raise ValueError("No embeddings generated for documents.")

        vectors = np.asarray(embeddings,dtype=np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        # Detect embedding dimension automatically
        dimension = vectors.shape[1]
        self.dimension = dimension

        self.index = faiss.IndexHNSWFlat(dimension,self.hnsw_m,faiss.METRIC_INNER_PRODUCT)

        # Construction-time accuracy / graph quality
        self.index.hnsw.efConstruction = self.ef_construction

        # Search-time accuracy
        self.index.hnsw.efSearch = self.ef_search

        # Add vectors to HNSW index
        self.index.add(vectors)

    def search(self,query: str,top_k: int = 5):

        if self.index is None:
            raise RuntimeError("Vector index has not been built.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        query_vector = np.asarray([query_embedding],dtype=np.float32)

        # Normalize query vector
        faiss.normalize_L2(query_vector)

        # Search HNSW
        scores, indices = self.index.search(query_vector,min(top_k, len(self.documents)))

        results = []

        for score, index in zip(scores[0],indices[0]):
            # FAISS can return -1 for invalid indices
            if index == -1:
                continue
            document = self.documents[index]
            results.append((document,float(score),))

        return results
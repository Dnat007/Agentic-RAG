from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import get_settings


class EmbeddingService:
    
    def __init__(self):
        setting = get_settings()
        self.model = HuggingFaceEmbeddings(
            model_name = setting.embedding_model,
            model_kwargs = {"device":"cpu"},
            encode_kwargs = {"normalize_embeddings": True}
        )    
    
    def embed_doc(self,documents: list[Document]):
        text = [doc.page_content for doc in documents]
        
        if not text :
            return []
        return self.model.embed_documents(text)
    
    def embed_query(self,query:str):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        return self.model.embed_query(query)
    
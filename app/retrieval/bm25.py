import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class BM25Retrieval:
    def __init__(self):
        self.documents: list[Document] = []
        self.bm25 = None

    def build(self,documents:list[Document]):
        if not documents:
            raise ValueError("cant build bm25 on empty documents")
        self.documents = documents
        
        tokenized_documents = [ self._tokenize(document.page_content)for document in documents]
        self.bm25 = BM25Okapi(tokenized_documents)
        
    def search(self,query:str,top_k:int=5):
        if self.bm25 is None:
            raise RuntimeError("Bm25 is not built right now brother")
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        scores = self.bm25.get_scores(self._tokenize(query))
        
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index],reverse=True)
        
        result = []
        
        for index in ranked_indices[:top_k]:
            result.append((
                self.documents[index],
                float(scores[index])
            ))
        
        return result
    
    @staticmethod
    def _tokenize(text:str):
        return re.findall(r"\b\w+\b",text.lower())
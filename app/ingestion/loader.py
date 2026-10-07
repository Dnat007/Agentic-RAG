from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

class PDFloader:
    def load(self,file_path:str | Path):
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError
        if path.suffix.lower() !=".pdf":
            raise ValueError(f"expected pdf file but got {path.suffix}")

        loader = PyPDFLoader(str(path))
        documents = loader.load()
        
        for doc in documents:
            doc.metadata.update({
                "document_id": path.stem,
                "document_name":path.name,
                "source": path
            })

        return documents
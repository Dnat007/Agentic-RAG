from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings


class DocumentsChunker:
    def __init__(self) -> None:
        settings = get_settings()
        self.splitters = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", "",]
        )

    def split(self, documents: list[Document]):
        chunks = self.splitters.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata["document_id"]}_{index}")

        return chunks

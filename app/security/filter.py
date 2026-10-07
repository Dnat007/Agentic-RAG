from langchain_core.documents import Document

class MetadataFilter:

    def filter(
        self,
        documents: list[Document],
        user_id: str | None = None,
        allowed_departments: list[str] | None = None,
        allowed_document_ids: list[str] | None = None,
    ) -> list[Document]:

        filtered = []

        for document in documents:
            metadata = document.metadata

            # User-level access
            if user_id is not None:
                document_user = metadata.get("user_id")
                if (document_user is not None and document_user != user_id):
                    continue

            # Department-level access
            if allowed_departments is not None:
                department = metadata.get("department")
                if department not in allowed_departments:
                    continue

            # Document-level access
            if allowed_document_ids is not None:
                document_id = metadata.get("document_id")
                if document_id not in allowed_document_ids:
                    continue

            filtered.append(document)

        return filtered
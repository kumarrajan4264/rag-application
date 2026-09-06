from pathlib import Path
import hashlib

import pymupdf

from app.models.document import Document


class PDFLoader:

    def generate_document_id(self, file_path: str) -> str:
        """
        Generate a stable document ID from the file name.
        """

        file_name = Path(file_path).name

        return hashlib.md5(
            file_name.encode("utf-8")
        ).hexdigest()

    def load(self, file_path: str) -> list[Document]:
        """
        Extract text from a PDF page by page.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {file_path}"
            )

        document_id = self.generate_document_id(
            str(file_path)
        )

        pdf = pymupdf.open(file_path)

        documents = []

        for page_index, page in enumerate(pdf):

            text = page.get_text("text").strip()

            if not text:
                continue

            document = Document(
                document_id=document_id,
                source=file_path.name,
                source_type="pdf",
                content=text,
                metadata={
                    "page_number": page_index + 1,
                    "total_pages": len(pdf),
                    "file_name": file_path.name,
                },
            )

            documents.append(document)

        pdf.close()

        return documents
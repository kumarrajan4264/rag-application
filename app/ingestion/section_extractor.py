import re
from uuid import NAMESPACE_URL, uuid5

from app.models.document import Document, DocumentSection


class SectionExtractor:
    """Lightweight heading-aware section extractor for text PDFs."""

    def extract_sections(self, document: Document) -> list[DocumentSection]:
        lines = [line.strip() for line in document.content.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return []

        sections: list[DocumentSection] = []
        current_title = "DOCUMENT"
        current_lines: list[str] = []

        for line in lines:
            if self._is_heading(line) and current_lines:
                sections.append(
                    self._build_section(document, current_title, current_lines)
                )
                current_title = line
                current_lines = []
            elif self._is_heading(line) and not current_lines:
                current_title = line
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                self._build_section(document, current_title, current_lines)
            )

        return sections

    def _build_section(
        self,
        document: Document,
        title: str,
        lines: list[str],
    ) -> DocumentSection:
        content = " ".join(lines).strip()
        page_number = document.metadata.get("page_number")
        section_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{document.document_id}:{page_number}:{title}:{content}",
            )
        )
        return DocumentSection(
            section_id=section_id,
            document_id=document.document_id,
            title=title,
            content=content,
            page_number=page_number,
            metadata={
                "source": document.source,
                "source_type": document.source_type,
            },
        )

    @staticmethod
    def _is_heading(line: str) -> bool:
        if len(line) > 80 or len(line.split()) > 8:
            return False
        if re.search(r"\d", line):
            return False
        letters = [char for char in line if char.isalpha()]
        if not letters:
            return False
        return line.upper() == line and len(letters) >= 3

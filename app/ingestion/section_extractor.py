import re
from uuid import uuid5, NAMESPACE_URL

from app.models.document import DocumentSection


class SectionExtractor:
    """
    Extracts semantic sections from a document.

    The extractor identifies section headings and groups the
    following content under those headings.

    Designed to work with:
        - PDFs
        - Resumes
        - Technical documents
        - Reports
        - Guides
        - Plain text documents
    """

    def __init__(self):
        self.default_section_title = "DOCUMENT"

    # =========================================================
    # PUBLIC API
    # =========================================================

    def extract_sections(self, document) -> list[DocumentSection]:
        """
        Extract sections from a Document.
        """

        text = self._clean_text(document.content)

        if not text:
            return []

        lines = text.split("\n")

        sections = []

        current_title = self.default_section_title
        current_content = []

        page_number = document.metadata.get("page_number")

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if self._is_heading(line):

                # Save previous section.
                if current_content:

                    sections.append(
                        self._create_section(
                            document=document,
                            title=current_title,
                            content="\n".join(
                                current_content
                            ),
                            page_number=page_number,
                        )
                    )

                current_title = self._clean_heading(line)
                current_content = []

            else:
                current_content.append(line)

        # Save final section.
        if current_content:

            sections.append(
                self._create_section(
                    document=document,
                    title=current_title,
                    content="\n".join(
                        current_content
                    ),
                    page_number=page_number,
                )
            )

        return sections

    # =========================================================
    # TEXT CLEANING
    # =========================================================

    def _clean_text(self, text: str) -> str:

        if not text:
            return ""

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Normalize bullets.
        text = text.replace("•", "●")
        text = text.replace("▪", "●")
        text = text.replace("◦", "●")

        # Remove zero-width characters.
        text = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            text,
        )

        # Normalize spaces but preserve newlines.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Normalize blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # =========================================================
    # HEADING DETECTION
    # =========================================================

    def _is_heading(self, line: str) -> bool:

        cleaned = line.strip()

        if not cleaned:
            return False

        # -----------------------------------------------------
        # Heading containing separator underscores.
        #
        # Example:
        # EXPERIENCE _____________
        # -----------------------------------------------------

        if re.search(r"_{5,}", cleaned):

            return True

        # -----------------------------------------------------
        # Common document section names.
        # -----------------------------------------------------

        normalized = re.sub(
            r"[^a-zA-Z ]",
            "",
            cleaned,
        ).strip().lower()

        common_headings = {
            "experience",
            "skills",
            "education",
            "projects",
            "certifications",
            "achievements",
            "summary",
            "profile",
            "objective",
            "work experience",
            "professional experience",
            "technical skills",
            "responsibilities",
            "publications",
            "references",
            "initiatives",
            "in college initiatives",
        }

        if normalized in common_headings:
            return True

        # -----------------------------------------------------
        # Uppercase headings.
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Uppercase headings.
        # -----------------------------------------------------

        letters = re.sub(
            r"[^A-Za-z]",
            "",
            cleaned,
        )

        # Don't classify numeric/score lines as headings.
        has_digit = bool(re.search(r"\d", cleaned))

        if (
            len(letters) >= 3
            and len(cleaned) <= 100
            and letters.isupper()
            and not has_digit
        ):
            return True

    # =========================================================
    # HEADING CLEANING
    # =========================================================

    def _clean_heading(self, heading: str) -> str:

        heading = heading.strip()

        # Remove underscore separators.
        heading = re.sub(
            r"_+",
            "",
            heading,
        )

        # Normalize whitespace.
        heading = re.sub(
            r"\s+",
            " ",
            heading,
        )

        return heading.strip()

    # =========================================================
    # SECTION CREATION
    # =========================================================

    def _create_section(
        self,
        document,
        title: str,
        content: str,
        page_number,
    ) -> DocumentSection:

        section_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{document.document_id}:{title}:{content}",
            )
        )

        return DocumentSection(
            section_id=section_id,
            document_id=document.document_id,
            title=title,
            content=content.strip(),
            page_number=page_number,
            metadata={
                "source": document.source,
                "source_type": document.source_type,
                "page_number": page_number,
                "content_length": len(
                    content.strip()
                ),
            },
        )
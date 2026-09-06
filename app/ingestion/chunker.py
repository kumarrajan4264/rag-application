import re
from uuid import uuid5, NAMESPACE_URL

from app.models.document import Document
from app.models.document import DocumentSection
from app.models.document import DocumentChunk


class TextChunker:
    """
    Section-aware semantic text chunker.

    Pipeline:

        Document
            ↓
        DocumentSection
            ↓
        DocumentChunk

    The chunker preserves section context in every chunk.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def chunk_document(
        self,
        document: Document,
        sections: list[DocumentSection] | None = None,
    ) -> list[DocumentChunk]:
        """
        Create chunks from a document.

        If sections are provided, chunking is performed
        independently inside each section.
        """

        if sections is None:
            return self._chunk_legacy_document(document)

        chunks = []

        for section in sections:
            section_chunks = self._chunk_section(
                document=document,
                section=section,
            )

            chunks.extend(section_chunks)

        return self._reindex_chunks(chunks)

    def create_chunks(
        self,
        document: Document,
        sections: list[DocumentSection] | None = None,
    ) -> list[DocumentChunk]:
        """
        Alias for chunk_document().
        """
        return self.chunk_document(
            document=document,
            sections=sections,
        )

    # ---------------------------------------------------------
    # SECTION-AWARE CHUNKING
    # ---------------------------------------------------------

    def _chunk_section(
        self,
        document: Document,
        section: DocumentSection,
    ) -> list[DocumentChunk]:

        text = self._normalize_text(section.content)

        if not text:
            return []

        blocks = self._split_into_blocks(text)

        if not blocks:
            return []

        raw_chunks = self._pack_blocks(blocks)

        # Add semantic overlap between chunks.
        raw_chunks = self._apply_overlap(raw_chunks)

        # Merge very small chunks where possible.
        raw_chunks = self._merge_small_chunks(raw_chunks)

        chunks = []

        for index, content in enumerate(raw_chunks):

            chunk_id = self._generate_chunk_id(
                document.document_id,
                section.section_id,
                index,
                content,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    content=content,
                    metadata={
                        "source": document.source,
                        "source_type": document.source_type,
                        "section_id": section.section_id,
                        "section_title": section.title,
                        "page_number": section.page_number,
                        "chunk_index": index,
                        "content_length": len(content),
                    },
                )
            )

        return chunks

    # ---------------------------------------------------------
    # TEXT NORMALIZATION
    # ---------------------------------------------------------

    def _normalize_text(self, text: str) -> str:

        if not text:
            return ""

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = text.replace("•", "●")
        text = text.replace("▪", "●")
        text = text.replace("◦", "●")

        text = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            text,
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ---------------------------------------------------------
    # BLOCK DETECTION
    # ---------------------------------------------------------

    def _split_into_blocks(
        self,
        text: str,
    ) -> list[str]:

        lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        blocks = []
        current_block = []

        for line in lines:

            if self._is_bullet(line):

                if current_block:
                    blocks.append(
                        " ".join(current_block)
                    )
                    current_block = []

                blocks.append(line)

            elif self._looks_like_heading(line):

                if current_block:
                    blocks.append(
                        " ".join(current_block)
                    )
                    current_block = []

                blocks.append(line)

            else:
                current_block.append(line)

        if current_block:
            blocks.append(
                " ".join(current_block)
            )

        return [
            block.strip()
            for block in blocks
            if block.strip()
        ]

    def _is_bullet(self, text: str) -> bool:

        return bool(
            re.match(
                r"^(●|-|\*|•)\s*",
                text,
            )
        )

    def _looks_like_heading(self, text: str) -> bool:

        cleaned = text.strip()

        if not cleaned:
            return False

        # Separator-style heading
        if re.search(r"_{5,}", cleaned):
            return True

        # Short title-like line
        if (
            len(cleaned) <= 80
            and len(cleaned.split()) <= 8
            and not cleaned.endswith(".")
        ):
            letters = re.sub(
                r"[^A-Za-z]",
                "",
                cleaned,
            )

            has_digit = bool(
                re.search(r"\d", cleaned)
            )

            if (
                len(letters) >= 3
                and letters.isupper()
                and not has_digit
            ):
                return True

        return False

    # ---------------------------------------------------------
    # CHUNK PACKING
    # ---------------------------------------------------------

    def _pack_blocks(
        self,
        blocks: list[str],
    ) -> list[str]:

        chunks = []
        current = ""

        for block in blocks:

            if len(block) > self.chunk_size:

                if current:
                    chunks.append(current.strip())
                    current = ""

                chunks.extend(
                    self._split_large_block(block)
                )

                continue

            candidate = (
                f"{current} {block}".strip()
                if current
                else block
            )

            if len(candidate) <= self.chunk_size:

                current = candidate

            else:

                if current:
                    chunks.append(current.strip())

                current = block

        if current:
            chunks.append(current.strip())

        return chunks

    def _split_large_block(
        self,
        block: str,
    ) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            block,
        )

        chunks = []
        current = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            candidate = (
                f"{current} {sentence}".strip()
                if current
                else sentence
            )

            if len(candidate) <= self.chunk_size:

                current = candidate

            else:

                if current:
                    chunks.append(current)

                # Extremely long sentence fallback.
                if len(sentence) > self.chunk_size:

                    chunks.extend(
                        self._split_by_words(sentence)
                    )

                    current = ""

                else:
                    current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _split_by_words(
        self,
        text: str,
    ) -> list[str]:

        words = text.split()

        chunks = []
        current_words = []
        current_length = 0

        for word in words:

            additional_length = (
                len(word)
                + (1 if current_words else 0)
            )

            if (
                current_length
                + additional_length
                <= self.chunk_size
            ):

                current_words.append(word)
                current_length += additional_length

            else:

                if current_words:
                    chunks.append(
                        " ".join(current_words)
                    )

                current_words = [word]
                current_length = len(word)

        if current_words:
            chunks.append(
                " ".join(current_words)
            )

        return chunks

    # ---------------------------------------------------------
    # SEMANTIC OVERLAP
    # ---------------------------------------------------------

    def _apply_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:

        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for index in range(1, len(chunks)):

            previous = chunks[index - 1]
            current = chunks[index]

            overlap = self._get_overlap_text(previous)

            if overlap:
                combined = f"{overlap} {current}".strip()

                if len(combined) <= self.chunk_size:
                    current = combined

            result.append(current)

        return result

    def _get_overlap_text(
        self,
        text: str,
    ) -> str:
        """
        Return complete semantic blocks from the end
        of the previous chunk.

        Bullets are treated as independent blocks.
        """

        if not text:
            return ""

        # Extract complete bullet blocks.
        bullet_matches = re.findall(
            r"●\s*.*?(?=\s+●|\Z)",
            text,
        )

        if bullet_matches:
            selected = []
            length = 0

            for bullet in reversed(bullet_matches):

                bullet = bullet.strip()

                additional = (
                    len(bullet)
                    + (1 if selected else 0)
                )

                if (
                    selected
                    and length + additional > self.chunk_overlap
                ):
                    break

                selected.insert(0, bullet)
                length += additional

            if selected:
                return " ".join(selected)

        # Fallback for normal prose.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        selected = []
        length = 0

        for sentence in reversed(sentences):

            sentence = sentence.strip()

            if not sentence:
                continue

            additional = (
                len(sentence)
                + (1 if selected else 0)
            )

            if (
                selected
                and length + additional > self.chunk_overlap
            ):
                break

            selected.insert(0, sentence)
            length += additional

        return " ".join(selected)

    # ---------------------------------------------------------
    # SMALL CHUNK HANDLING
    # ---------------------------------------------------------

    def _merge_small_chunks(
        self,
        chunks: list[str],
    ) -> list[str]:

        if len(chunks) <= 1:
            return chunks

        result = []

        for chunk in chunks:

            if (
                result
                and len(chunk) < self.min_chunk_size
            ):

                candidate = (
                    f"{result[-1]} {chunk}"
                )

                if len(candidate) <= self.chunk_size:
                    result[-1] = candidate
                    continue

            result.append(chunk)

        return result

    # ---------------------------------------------------------
    # LEGACY FALLBACK
    # ---------------------------------------------------------

    def _chunk_legacy_document(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        section = DocumentSection(
            section_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"{document.document_id}:DOCUMENT",
                )
            ),
            document_id=document.document_id,
            title="DOCUMENT",
            content=document.content,
            page_number=document.metadata.get(
                "page_number"
            ),
            metadata=document.metadata.copy(),
        )

        return self._chunk_section(
            document=document,
            section=section,
        )

    # ---------------------------------------------------------
    # IDS
    # ---------------------------------------------------------

    def _generate_chunk_id(
        self,
        document_id: str,
        section_id: str,
        index: int,
        content: str,
    ) -> str:

        return str(
            uuid5(
                NAMESPACE_URL,
                f"{document_id}:{section_id}:{index}:{content}",
            )
        )

    # ---------------------------------------------------------
    # REINDEX
    # ---------------------------------------------------------

    def _reindex_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:

        for index, chunk in enumerate(chunks):

            chunk.metadata["global_chunk_index"] = index

        return chunks
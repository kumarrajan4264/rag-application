from app.embeddings.embedding_model import EmbeddingModel
from app.vector_store.chroma_store import ChromaVectorStore
from app.llm.gemini_provider import GeminiProvider


class RAGPipeline:

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.vector_store = ChromaVectorStore(
            persist_directory="data/chroma",
            collection_name="rag_documents",
        )

        self.llm = GeminiProvider()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ):

        query_vector = self.embedding_model.embed_text(
            question
        )

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            metadata_filter=metadata_filter,
        )

        return results

    def build_context(self, results):

        context_parts = []

        for index, result in enumerate(results, start=1):

            content = result.content

            metadata = result.metadata

            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get(
                "page_number",
                "Unknown"
            )

            context_parts.append(
                f"""
SOURCE {index}
Source: {source}
Page: {page}

Content:
{content}
"""
            )

        return "\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        context: str,
    ):

        prompt = f"""
You are an enterprise document analysis assistant.

Answer the user's question ONLY using the
provided context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I don't have enough information in the
provided documents."

Always cite the source and page number
when answering.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

        return self.llm.generate(prompt)

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ):

        results = self.retrieve(
            question,
            top_k,
        )

        context = self.build_context(
            results
        )

        answer = self.generate_answer(
            question,
            context,
        )

        return {
            "question": question,
            "answer": answer,
            "results": results,
        }